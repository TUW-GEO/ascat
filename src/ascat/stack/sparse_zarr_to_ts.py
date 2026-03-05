# Copyright (c) 2025, TU Wien
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Convert sparse swath-time Zarr cubes to dense (incomplete multidimensional)
time-series Zarr stores.

Transforms a Zarr store with dimensions (swath_time, spacecraft, [beam,] gpi) into
one with dimensions (gpi, obs, [beam]), where observations are packed densely
per GPI and sorted by measurement time.

Supports appending new data to an existing dense store.

Optionally supports sharded arrays (Zarr v3 sharding_indexed codec) along
the gpi and/or obs dimensions.  When sharding is enabled the unit of
parallelism is automatically promoted to the shard boundary so that no two
workers ever write to the same shard file concurrently.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time as timer
from typing import Optional

import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import (
    BloscCodec,
    BloscShuffle,
)

from ascat.utils import dtype_to_nan


# ---------------------------------------------------------------------------
# Sharding configuration (internal helper, not part of the public API)
# ---------------------------------------------------------------------------

@dataclass
class _ShardingConfig:
    """Resolved sharding parameters, built inside ``sparse_to_dense``."""

    shard_size_gpi: int
    inner_chunk_gpi: int
    shard_size_obs: Optional[int]
    inner_chunk_obs: int

    def __post_init__(self):
        if self.shard_size_gpi % self.inner_chunk_gpi != 0:
            raise ValueError(
                f"shard_size_gpi ({self.shard_size_gpi}) must be a multiple of "
                f"chunk_size_gpi ({self.inner_chunk_gpi})"
            )
        if self.shard_size_obs is not None:
            if self.shard_size_obs % self.inner_chunk_obs != 0:
                raise ValueError(
                    f"shard_size_obs ({self.shard_size_obs}) must be a multiple of "
                    f"chunk_size_obs ({self.inner_chunk_obs})"
                )

    @property
    def obs_alignment(self) -> int:
        """Unit to which the obs dimension is padded/aligned."""
        return self.shard_size_obs if self.shard_size_obs is not None else self.inner_chunk_obs

    @property
    def gpi_alignment(self) -> int:
        """Worker granularity: one full shard per worker."""
        return self.shard_size_gpi


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sparse_to_dense(
    sparse_path,
    out_path,
    chunk_size_gpi=None,
    chunk_size_obs=300,
    n_workers=1,
    gpi_mask=None,
    shard_size_gpi=None,
    shard_size_obs=None,
):
    """Convert a sparse swath-time Zarr cube to a dense time-series store.

    Parameters
    ----------
    sparse_path : str or Path
        Path to the sparse Zarr store with dims (swath_time, spacecraft, [beam,] gpi).
    out_path : str or Path
        Path for the output dense Zarr store with dims (gpi, obs, [beam]).
    chunk_size_gpi : int, optional
        Chunk size along the gpi dimension.  When ``shard_size_gpi`` is also
        given this becomes the *inner* (compressed) chunk size within each
        shard.  Defaults to the sparse store's GPI chunk size.
    chunk_size_obs : int, optional
        Chunk size along the obs dimension.  When ``shard_size_obs`` is also
        given this becomes the *inner* (compressed) chunk size within each
        shard.  Default 300.
    n_workers : int, optional
        Number of parallel workers for processing GPI chunks. Default 1.
        When ``shard_size_gpi`` is set each worker is assigned a full shard's
        worth of GPIs so that no two workers write to the same shard file.
    gpi_mask : np.ndarray of bool, optional
        Boolean array of shape (n_gpi,). GPIs where gpi_mask is True will be
        skipped entirely — no data is read or written for them. Default None
        (process all GPIs).
    shard_size_gpi : int, optional
        Number of GPIs per shard.  Must be a multiple of ``chunk_size_gpi``.
        When provided, output arrays use the sharding_indexed codec along the
        gpi dimension and parallelism is aligned to shard boundaries.
        Default None (no gpi sharding).
    shard_size_obs : int, optional
        Number of obs per shard.  Must be a multiple of ``chunk_size_obs``.
        When provided, output arrays use the sharding_indexed codec along the
        obs dimension.  Default None (no obs sharding).
    """
    sparse_path = Path(sparse_path)
    out_path = Path(out_path)

    sparse_root = zarr.open(sparse_path, mode="r")

    has_beams = "beam" in sparse_root
    n_gpi = sparse_root["gpi"].shape[0]
    n_spacecraft = sparse_root["spacecraft"].shape[0]
    n_swath_time = sparse_root["swath_time"].shape[0]
    sparse_gpi_chunk_size = sparse_root["time"].chunks[-1]
    chunk_size_gpi = chunk_size_gpi or sparse_gpi_chunk_size

    # Build internal sharding config if shard sizes were requested.
    # chunk_size_gpi/obs become the inner chunk sizes within each shard.
    if shard_size_gpi is not None:
        sharding = _ShardingConfig(
            shard_size_gpi=shard_size_gpi,
            inner_chunk_gpi=chunk_size_gpi,
            shard_size_obs=shard_size_obs,
            inner_chunk_obs=chunk_size_obs,
        )
    else:
        sharding = None

    # Resolve alignment and worker granularity.
    obs_alignment = sharding.obs_alignment if sharding else chunk_size_obs
    worker_gpi_size = sharding.gpi_alignment if sharding else chunk_size_gpi

    if gpi_mask is not None:
        gpi_mask = np.asarray(gpi_mask, dtype=bool)
        if gpi_mask.shape != (n_gpi,):
            raise ValueError(
                f"gpi_mask shape {gpi_mask.shape} does not match n_gpi ({n_gpi},)"
            )
        n_masked = int(gpi_mask.sum())
        print(f"GPI mask: skipping {n_masked}/{n_gpi} GPIs")

    beam_vars, scalar_vars = _classify_variables(sparse_root, has_beams)
    print(f"Beam variables: {sorted(beam_vars)}")
    print(f"Scalar variables: {sorted(scalar_vars)}")

    # --- First pass: find all populated chunks across entire store ---
    print("Scanning for populated chunks...")
    scan_start = timer()
    populated_map = _scan_all_populated_chunks(
        sparse_path, n_swath_time, n_spacecraft, n_gpi, sparse_gpi_chunk_size
    )
    print(f"Scan complete in {timer() - scan_start:.6f}s, "
          f"found {sum(len(v) for v in populated_map.values())} populated chunk slots")

    max_new_obs = max((len(slots) for slots in populated_map.values()), default=0)

    # --- Create or open output store, pre-expanding if needed ---
    if not (out_path / "zarr.json").exists():
        obs_dim_size = _round_up_to_chunk(max_new_obs, obs_alignment)
        print(f"Creating dense Zarr structure at {out_path} "
              f"with obs_dim_size={obs_dim_size}")
        _create_dense_structure(
            out_path=out_path,
            sparse_root=sparse_root,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            obs_dim_size=obs_dim_size,
            chunk_size_gpi=chunk_size_gpi,
            chunk_size_obs=chunk_size_obs,
            sharding=sharding,
        )
    else:
        out_root_check = zarr.open(out_path, mode="a")
        current_obs_size = out_root_check["obs"].shape[0]
        existing_max_nobs = int(out_root_check["n_obs"][:].max())
        worst_case = existing_max_nobs + max_new_obs
        if worst_case > current_obs_size:
            _expand_obs_dimension(
                out_root_check, worst_case, obs_alignment,
                beam_vars, scalar_vars, has_beams,
            )

    # --- Process GPI chunks in parallel ---
    gpi_chunks = _build_gpi_chunk_ranges(n_gpi, worker_gpi_size)
    print(f"Processing {len(gpi_chunks)} GPI chunks with {n_workers} workers")

    process_func = partial(
        _process_gpi_chunk,
        sparse_path=str(sparse_path),
        out_path=str(out_path),
        beam_vars=beam_vars,
        scalar_vars=scalar_vars,
        has_beams=has_beams,
        populated_map=populated_map,
        sparse_gpi_chunk_size=sparse_gpi_chunk_size,
        inner_chunk_gpi=chunk_size_gpi,
        chunk_size_obs=chunk_size_obs,
        obs_alignment=obs_alignment,
        gpi_mask=gpi_mask,
    )

    start = timer()
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in gpi_chunks]
            futures = tqdm(futures, total=len(gpi_chunks), desc="GPI chunks")
            sum(1 for future in futures if future.result())
    else:
        gpi_chunks = tqdm(gpi_chunks, total=len(gpi_chunks), desc="GPI chunks")
        for chunk in gpi_chunks:
            process_func(chunk)

    elapsed = timer() - start
    print(f"Done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------

def _classify_variables(sparse_root, has_beams):
    """Classify data variables into beam and scalar categories."""
    coord_names = {"swath_time", "spacecraft", "beam", "gpi", "longitude", "latitude"}
    beam_vars = set()
    scalar_vars = set()

    for name in sparse_root:
        if name in coord_names:
            continue
        arr = sparse_root[name]
        if not hasattr(arr, "ndim"):
            continue
        if has_beams and arr.ndim == 4:
            beam_vars.add(name)
        elif arr.ndim == 3:
            scalar_vars.add(name)

    return beam_vars, scalar_vars


# ---------------------------------------------------------------------------
# Populated-chunk scanning
# ---------------------------------------------------------------------------

def _scan_all_populated_chunks(sparse_path, n_swath_time, n_spacecraft, n_gpi, sparse_gpi_chunk_size):
    """Scan the Zarr v3 store for all populated (swath_time, spacecraft, gpi_chunk) slots.

    Uses ``os.scandir`` to enumerate existing chunk/shard files instead of
    probing individual paths, reducing the number of syscalls from
    O(n_swath_time * n_spacecraft * n_gpi_chunks) to O(number of existing files).

    Supports both sharded and unsharded sparse stores:
    - Sharded: enumerates ``time/c/{t}/{s}/`` to find shard files, then maps
      each shard index back to the inner gpi_chunk indices it covers.
    - Unsharded: enumerates ``time/c/{t}/{s}/`` to find chunk files directly.

    Returns
    -------
    dict[int, list[tuple[int, int]]]
        gpi_chunk_index -> list of (swath_time_idx, spacecraft_idx) with data.
    """
    import os

    n_gpi_chunks = -(-n_gpi // sparse_gpi_chunk_size)

    sparse_root = zarr.open(str(sparse_path), mode="r")
    time_meta = sparse_root["time"].metadata
    is_sharded = any(
        getattr(c, "name", None) == "sharding_indexed"
        or type(c).__name__ == "ShardingCodec"
        for c in (time_meta.codecs or [])
    )

    time_c = str(sparse_path / "time" / "c")

    if not os.path.isdir(time_c):
        return {}

    if is_sharded:
        shard_size_gpi = time_meta.chunk_grid.chunk_shape[-1]
        chunks_per_shard = shard_size_gpi // sparse_gpi_chunk_size

        # Single pass: scandir the tree to find all existing shard files.
        # shard_ts_map[shard_idx] = [(t_idx, s_idx), ...]
        shard_ts_map = {}

        for t_entry in os.scandir(time_c):
            if not t_entry.is_dir():
                continue
            t_idx = int(t_entry.name)
            for s_entry in os.scandir(t_entry.path):
                if not s_entry.is_dir():
                    continue
                s_idx = int(s_entry.name)
                for shard_entry in os.scandir(s_entry.path):
                    shard_idx = int(shard_entry.name)
                    shard_ts_map.setdefault(shard_idx, []).append(
                        (t_idx, s_idx)
                    )

        # Map inner gpi_chunk_idx -> populated (t, s) slots via shard lookup.
        populated_map = {}
        for gc in range(n_gpi_chunks):
            shard_idx = (gc * sparse_gpi_chunk_size) // shard_size_gpi
            slots = shard_ts_map.get(shard_idx)
            if slots:
                populated_map[gc] = sorted(slots)

    else:
        # Unsharded: each file in time/c/{t}/{s}/ is a gpi chunk directly.
        # chunk_ts_map[gc] = [(t_idx, s_idx), ...]
        chunk_ts_map = {}

        for t_entry in os.scandir(time_c):
            if not t_entry.is_dir():
                continue
            t_idx = int(t_entry.name)
            for s_entry in os.scandir(t_entry.path):
                if not s_entry.is_dir():
                    continue
                s_idx = int(s_entry.name)
                for chunk_entry in os.scandir(s_entry.path):
                    gc = int(chunk_entry.name)
                    chunk_ts_map.setdefault(gc, []).append(
                        (t_idx, s_idx)
                    )

        populated_map = {
            gc: sorted(slots)
            for gc, slots in chunk_ts_map.items()
        }

    return populated_map


# ---------------------------------------------------------------------------
# Output store creation and resizing
# ---------------------------------------------------------------------------

def _round_up_to_chunk(n, chunk_size):
    """Round n up to the nearest multiple of chunk_size."""
    return -(-n // chunk_size) * chunk_size


def _make_array_kwargs(inner_chunk_shape, sharding: Optional[_ShardingConfig]):
    """Return kwargs for ``create_array`` covering chunks, shards, and compression.

    Uses the zarr 3.1.x API: ``chunks=`` for the inner chunk shape,
    ``shards=`` for the shard shape (when sharding is active), and
    ``compressors=`` for inner compression in both cases.
    The beam axis (axis 2, if present) is never sharded.
    """
    kwargs = {
        "chunks": inner_chunk_shape,
        "compressors": BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
    }

    if sharding is not None:
        shard_shape = list(inner_chunk_shape)
        shard_shape[0] = sharding.shard_size_gpi
        if len(shard_shape) > 1 and sharding.shard_size_obs is not None:
            shard_shape[1] = sharding.shard_size_obs
        kwargs["shards"] = tuple(shard_shape)

    return kwargs


def _create_dense_structure(
    out_path,
    sparse_root,
    beam_vars,
    scalar_vars,
    has_beams,
    obs_dim_size,
    chunk_size_gpi,
    chunk_size_obs,
    sharding: Optional[_ShardingConfig] = None,
):
    """Create the empty dense Zarr store."""
    n_gpi = sparse_root["gpi"].shape[0]

    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in sorted(beam_vars):
            src = sparse_root[var]
            arr_kwargs = _make_array_kwargs((chunk_size_gpi, chunk_size_obs, 1), sharding)
            root.create_array(
                name=var,
                dtype=src.dtype,
                shape=(n_gpi, obs_dim_size, n_beams),
                dimension_names=("gpi", "obs", "beam"),
                fill_value=src.metadata.fill_value,
                attributes=dict(src.attrs),
                **arr_kwargs,
            )

    for var in sorted(scalar_vars):
        src = sparse_root[var]
        arr_kwargs = _make_array_kwargs((chunk_size_gpi, chunk_size_obs), sharding)
        root.create_array(
            name=var,
            dtype=src.dtype,
            shape=(n_gpi, obs_dim_size),
            dimension_names=("gpi", "obs"),
            fill_value=src.metadata.fill_value,
            attributes=dict(src.attrs),
            **arr_kwargs,
        )

    root.create_array(
        name="n_obs",
        dtype="uint32",
        shape=(n_gpi,),
        chunks=(sharding.shard_size_gpi if sharding else chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=0,
        compressors=[
            BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)
        ],
    )

    root.create_array(
        "obs",
        data=np.arange(obs_dim_size, dtype="int32"),
        chunks=(chunk_size_obs,),
        dimension_names=("obs",),
        fill_value=dtype_to_nan[np.dtype("int32")],
        compressors=None,
    )

    root.create_array(
        "gpi",
        data=sparse_root["gpi"][:],
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("int32")],
        compressors=None,
    )

    root.create_array(
        "longitude",
        data=sparse_root["longitude"][:],
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("float32")],
        compressors=None,
    )

    root.create_array(
        "latitude",
        data=sparse_root["latitude"][:],
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("float32")],
        compressors=None,
    )

    if has_beams:
        root.create_array(
            "beam",
            data=sparse_root["beam"][:],
            chunks=(1,),
            dimension_names=("beam",),
            fill_value=b"",
            compressors=None,
        )


def _expand_obs_dimension(out_root, needed_size, obs_alignment, beam_vars, scalar_vars, has_beams):
    """Expand the obs dimension in alignment-sized increments.

    ``obs_alignment`` is the shard obs size when sharding is active, or the
    plain chunk obs size otherwise, ensuring expansions always land on a
    shard/chunk boundary.
    """
    current_size = out_root["obs"].shape[0]
    new_size = _round_up_to_chunk(needed_size, obs_alignment)

    if new_size <= current_size:
        return

    print(f"Expanding obs dimension from {current_size} to {new_size}")

    for var in sorted(scalar_vars):
        arr = out_root[var]
        new_shape = list(arr.shape)
        new_shape[1] = new_size
        arr.resize(tuple(new_shape))

    if has_beams:
        for var in sorted(beam_vars):
            arr = out_root[var]
            new_shape = list(arr.shape)
            new_shape[1] = new_size
            arr.resize(tuple(new_shape))

    out_root["obs"].resize((new_size,))
    out_root["obs"][current_size:new_size] = np.arange(
        current_size, new_size, dtype="int32"
    )


# ---------------------------------------------------------------------------
# GPI chunk processing
# ---------------------------------------------------------------------------

def _build_gpi_chunk_ranges(n_gpi, chunk_size_gpi):
    """Build list of (start, end) index pairs for GPI chunks."""
    return [
        (i, min(i + chunk_size_gpi, n_gpi))
        for i in range(0, n_gpi, chunk_size_gpi)
    ]


def _process_gpi_chunk(
    gpi_range,
    sparse_path,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
    populated_map,
    sparse_gpi_chunk_size,
    inner_chunk_gpi,
    chunk_size_obs,
    obs_alignment,
    gpi_mask=None,
):
    """Process a range of GPIs: extract, sort, and write/append to the dense store.

    When sharding is active ``gpi_range`` spans a full shard (e.g. 1024 GPIs)
    so this worker has exclusive ownership of those shard files.  Internally
    we iterate in ``inner_chunk_gpi``-sized sub-chunks (e.g. 32 GPIs) to keep
    memory usage proportional to the inner chunk rather than the full shard.
    Without sharding ``inner_chunk_gpi`` equals the worker range size so the
    inner loop runs exactly once, preserving the original behaviour.

    Parameters
    ----------
    gpi_range : tuple of (int, int)
        (start, end) GPI indices owned by this worker.
    inner_chunk_gpi : int
        Sub-chunk size for the inner read-modify-write loop.
    obs_alignment : int
        Shard obs size (sharding) or plain chunk obs size (no sharding).
    """
    gpi_start, gpi_end = gpi_range

    if gpi_mask is not None:
        local_mask = gpi_mask[gpi_start:gpi_end]
        if local_mask.all():
            return
    else:
        local_mask = None

    # Collect all populated slots that overlap this worker's GPI range.
    gc_start = gpi_start // sparse_gpi_chunk_size
    gc_end = (gpi_end - 1) // sparse_gpi_chunk_size
    populated_set = set()
    for gc in range(gc_start, gc_end + 1):
        populated_set.update(populated_map.get(gc, []))

    if not populated_set:
        return

    populated = sorted(populated_set)

    # Pre-read sparse data for the entire worker range to eliminate chunk amplification.
    # This function computes which sparse chunks overlap [gpi_start, gpi_end),
    # reads all data for those chunks once per variable, and returns a cache dict.
    preread_cache = _preread_sparse_chunks_for_range(
        sparse_path=sparse_path,
        gpi_start=gpi_start,
        gpi_end=gpi_end,
        populated=populated,
        sparse_gpi_chunk_size=sparse_gpi_chunk_size,
        beam_vars=beam_vars,
        scalar_vars=scalar_vars,
        has_beams=has_beams,
    )

    for sub_start in range(gpi_start, gpi_end, inner_chunk_gpi):
        sub_end = min(sub_start + inner_chunk_gpi, gpi_end)

        if local_mask is not None:
            sub_offset = sub_start - gpi_start
            sub_mask = local_mask[sub_offset: sub_offset + (sub_end - sub_start)]
            if sub_mask.all():
                continue
        else:
            sub_mask = None

        _process_inner_gpi_chunk(
            gpi_start=sub_start,
            gpi_end=sub_end,
            sparse_path=sparse_path,
            out_path=out_path,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            populated=populated,
            sparse_gpi_chunk_size=sparse_gpi_chunk_size,
            chunk_size_obs=chunk_size_obs,
            obs_alignment=obs_alignment,
            local_mask=sub_mask,
            preread_cache=preread_cache,
        )


def _preread_sparse_chunks_for_range(
    sparse_path,
    gpi_start,
    gpi_end,
    populated,
    sparse_gpi_chunk_size,
    beam_vars,
    scalar_vars,
    has_beams,
):
    """Pre-read all sparse chunks that overlap [gpi_start, gpi_end).

    Eliminates read amplification by reading each sparse chunk once regardless
    of how many inner sub-chunks reference it.

    Generic design: works for any relationship between worker size and
    sparse chunk size. Examples:
    - sparse=4096, shard=1024: reads 2 sparse chunks (overlap by 1024)
    - sparse=4096, shard=4096: reads 1 sparse chunk (no overlap)
    - sparse=4096, shard=8192: reads 2 sparse chunks (split across shard boundary)

    Parameters
    ----------
    sparse_path : Path
        Path to sparse zarr store.
    gpi_start, gpi_end : int
        GPI range for this worker (exclusive end).
    populated : list of tuple
        List of (swath_time_idx, spacecraft_idx) with data in this range.
    sparse_gpi_chunk_size : int
        The GPI chunk size in the sparse store (typically 4096).
    beam_vars, scalar_vars : set
        Variable names for beam and scalar data.
    has_beams : bool
        Whether beam variables exist.

    Returns
    -------
    dict
        Nested cache: {(chunk_idx, var_name): np.ndarray}
        chunk_idx is the sparse chunk index, var_name is the variable name.
        Each array has shape (n_spacecraft, sparse_gpi_chunk_size) for scalar vars
        or (n_spacecraft, n_beams, sparse_gpi_chunk_size) for beam vars.
    """
    sparse_root = zarr.open(sparse_path, mode="r")

    gc_start = gpi_start // sparse_gpi_chunk_size
    gc_end = (gpi_end - 1) // sparse_gpi_chunk_size

    cache = {}

    for gc in range(gc_start, gc_end + 1):
        snap_start = gc * sparse_gpi_chunk_size
        snap_end = snap_start + sparse_gpi_chunk_size

        for t_idx, s_idx in populated:
            for var in sorted(scalar_vars):
                key = (gc, var, t_idx, s_idx)
                cache[key] = sparse_root[var][t_idx, s_idx, snap_start:snap_end].copy()

            if has_beams:
                for var in sorted(beam_vars):
                    key = (gc, var, t_idx, s_idx)
                    cache[key] = sparse_root[var][t_idx, s_idx, :, snap_start:snap_end].copy()

    return cache


def _slice_from_preread(preread_cache, gpi_start, gpi_end, sparse_gpi_chunk_size, var, t_idx, s_idx):
    """Extract slice from pre-read cache for a specific GPI range.

    Since the pre-read cache stores full sparse chunks, we need to compute
    which chunk(s) the requested range falls in and slice appropriately.
    Handles ranges spanning any number of sparse chunks.

    Works for both 2D arrays (gpi,) and 3D arrays (n_beams, gpi).
    """
    gc_start = gpi_start // sparse_gpi_chunk_size
    gc_end = (gpi_end - 1) // sparse_gpi_chunk_size

    if gc_start == gc_end:
        chunk_start = gc_start * sparse_gpi_chunk_size
        local_start = gpi_start - chunk_start
        local_end = gpi_end - chunk_start
        return preread_cache[(gc_start, var, t_idx, s_idx)][..., local_start:local_end]

    slices = []
    for gc in range(gc_start, gc_end + 1):
        chunk_start = gc * sparse_gpi_chunk_size

        local_start = max(gpi_start - chunk_start, 0)
        local_end = min(gpi_end - chunk_start, sparse_gpi_chunk_size)

        slices.append(
            preread_cache[(gc, var, t_idx, s_idx)][..., local_start:local_end]
        )

    return np.concatenate(slices, axis=-1)


def _process_inner_gpi_chunk(
    gpi_start,
    gpi_end,
    sparse_path,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
    populated,
    sparse_gpi_chunk_size,
    chunk_size_obs,
    obs_alignment,
    local_mask=None,
    preread_cache=None,
):
    """Core read-extract-sort-write logic for a single inner GPI sub-chunk.

    Uses a chunk-batched read-modify-write strategy: for each output
    (gpi_chunk, obs_chunk) that needs updating, read the full chunk from
    the output store, apply all GPI updates in-memory, write it back once.

    The obs-dimension loop iterates in ``obs_alignment``-sized steps so that
    each read-modify-write cycle aligns to shard (or plain chunk) boundaries.

    Parameters
    ----------
    preread_cache : dict or None
        Pre-read sparse data cache from _preread_sparse_chunks_for_range.
        If None, falls back to direct zarr reads (original behavior).
    """
    n_gpi_chunk = gpi_end - gpi_start

    sparse_root = zarr.open(sparse_path, mode="r")
    out_root = zarr.open(out_path, mode="a")

    time_var = "time"
    time_fill = sparse_root[time_var].metadata.fill_value

    # --- Read time values for all populated slots ---
    time_slices = np.empty(
        (len(populated), n_gpi_chunk), dtype=sparse_root[time_var].dtype
    )
    for i, (t_idx, s_idx) in enumerate(populated):
        if preread_cache is not None:
            time_slices[i, :] = _slice_from_preread(
                preread_cache, gpi_start, gpi_end, sparse_gpi_chunk_size, time_var, t_idx, s_idx
            )
        else:
            time_slices[i, :] = sparse_root[time_var][t_idx, s_idx, gpi_start:gpi_end]

    # --- Per-GPI: find valid observations, sort by time ---
    valid_mask = time_slices != time_fill

    gpi_sorted_slots = []

    from .numba_kernels import _sort_slots_kernel, _scatter_scalar_kernel, _scatter_beam_kernel

    mask_arr = local_mask.astype(np.bool_) if local_mask is not None else np.zeros(0, dtype=np.bool_)
    offsets, flat_indices, new_counts = _sort_slots_kernel(time_slices, time_fill, mask_arr)

    if new_counts.sum() == 0:
        return

    for g in range(n_gpi_chunk):
        if local_mask is not None and local_mask[g]:
            gpi_sorted_slots.append(np.array([], dtype=int))
            continue

        valid_slots = np.where(valid_mask[:, g])[0]
        if len(valid_slots) == 0:
            gpi_sorted_slots.append(np.array([], dtype=int))
            continue
        times_g = time_slices[valid_slots, g]
        sorted_slots = valid_slots[np.argsort(times_g)]
        gpi_sorted_slots.append(sorted_slots)
        new_counts[g] = len(sorted_slots)

    if new_counts.sum() == 0:
        return

    # --- Read all variable data for populated slots ---
    data_cache = {}

    for var in sorted(scalar_vars):
        if var == time_var:
            continue
        buf = np.empty((len(populated), n_gpi_chunk), dtype=sparse_root[var].dtype)
        for i, (t_idx, s_idx) in enumerate(populated):
            if preread_cache is not None:
                buf[i, :] = _slice_from_preread(
                    preread_cache, gpi_start, gpi_end, sparse_gpi_chunk_size, var, t_idx, s_idx
                )
            else:
                buf[i, :] = sparse_root[var][t_idx, s_idx, gpi_start:gpi_end]
        data_cache[var] = buf

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in sorted(beam_vars):
            buf = np.empty(
                (len(populated), n_beams, n_gpi_chunk), dtype=sparse_root[var].dtype
            )
            for i, (t_idx, s_idx) in enumerate(populated):
                if preread_cache is not None:
                    buf[i, :, :] = _slice_from_preread(
                        preread_cache, gpi_start, gpi_end, sparse_gpi_chunk_size, var, t_idx, s_idx
                    )
                else:
                    buf[i, :, :] = sparse_root[var][t_idx, s_idx, :, gpi_start:gpi_end]
            data_cache[var] = buf

    # --- Determine write positions ---
    existing_n_obs = out_root["n_obs"][gpi_start:gpi_end].astype(np.int32)
    write_starts = existing_n_obs.copy()
    write_ends = write_starts + new_counts

    if write_ends.max() == 0:
        return

    min_obs = int(write_starts[write_starts < write_ends].min())
    max_obs = int(write_ends.max())

    # Loop over obs_alignment-sized windows (shard or plain chunk boundary).
    first_obs_chunk = min_obs // obs_alignment
    last_obs_chunk = (max_obs - 1) // obs_alignment

    gpi_slice = slice(gpi_start, gpi_end)

    for obs_chunk_idx in range(first_obs_chunk, last_obs_chunk + 1):
        obs_chunk_start = obs_chunk_idx * obs_alignment
        obs_chunk_end = min(obs_chunk_start + obs_alignment,
                            out_root["obs"].shape[0])
        obs_slice = slice(obs_chunk_start, obs_chunk_end)

        has_write = (write_starts < obs_chunk_end) & (write_ends > obs_chunk_start)
        if not np.any(has_write):
            continue

        # --- Read-modify-write for each variable ---

        # Time (scalar)
        chunk_data_time = out_root[time_var][gpi_slice, obs_slice]
        _scatter_scalar_kernel(chunk_data_time, time_slices, offsets, flat_indices,
                            write_starts, obs_chunk_start, obs_chunk_end)
        out_root[time_var][gpi_slice, obs_slice] = chunk_data_time

        # Scalar variables
        for var in sorted(scalar_vars):
            if var == time_var:
                continue

            chunk_data = out_root[var][gpi_slice, obs_slice]
            _scatter_scalar_kernel(chunk_data, data_cache[var], offsets, flat_indices,
                                write_starts, obs_chunk_start, obs_chunk_end)
            out_root[var][gpi_slice, obs_slice] = chunk_data

        # Beam variables
        if has_beams:
            for var in sorted(beam_vars):
                chunk_data = out_root[var][gpi_slice, obs_slice, :]
                _scatter_beam_kernel(chunk_data, data_cache[var], offsets, flat_indices,
                                    write_starts, obs_chunk_start, obs_chunk_end)
                out_root[var][gpi_slice, obs_slice, :] = chunk_data

    # --- Update n_obs ---
    out_root["n_obs"][gpi_start:gpi_end] = write_ends.astype("uint32")
