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
Two-pass sparse-to-dense conversion via intermediate rechunked store.

Pass 1 (rechunk): Read the sparse swath-time Zarr store time-slot by
time-slot (in batches), writing to an intermediate Zarr store with a
compacted observation axis and small spatial chunks.  This transforms
the I/O pattern from random-access into sequential bulk reads and
chunk-aligned writes.

Pass 2 (densify): Read the intermediate store spatial-chunk by
spatial-chunk.  Each read delivers the full observation history for a
small group of GPIs.  Drop fill values, sort by time, write to the
final dense time-series store.

Pass 2 supports both *creation* (fresh dense store) and *append*
(add new observations to an existing dense store).  In append mode,
each GPI's new observations are placed at position ``n_obs[gpi]`` and
``n_obs`` is updated accordingly.  The new data is expected to be
temporally after the existing data — a per-chunk continuity check
runs inside each append worker and raises if any GPI would have new
data earlier than its existing last observation.

The two-pass approach reduces total Zarr read operations from hundreds
of millions (original method) to a few million, at the cost of
temporary disk space for the intermediate store.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from time import time as timer
from typing import Optional
import struct

import numpy as np
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec, BloscShuffle

from ascat.utils import dtype_to_nan


# ---------------------------------------------------------------------------
# Sharding configuration
# ---------------------------------------------------------------------------


@dataclass
class _ShardingConfig:
    """Resolved sharding parameters for the dense output store."""

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
        return (
            self.shard_size_obs
            if self.shard_size_obs is not None
            else self.inner_chunk_obs
        )

    @property
    def gpi_alignment(self) -> int:
        return self.shard_size_gpi


def _make_array_kwargs(inner_chunk_shape, sharding: Optional[_ShardingConfig]):
    """Return kwargs for ``create_array`` covering chunks, shards, and compression."""
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


# ---------------------------------------------------------------------------
# Shard index reading
# ---------------------------------------------------------------------------

SHARD_INDEX_EMPTY = 0xFFFFFFFFFFFFFFFF


def _read_shard_index(shard_path, n_inner_chunks):
    """Read the sharding_indexed codec footer and return populated inner chunk indices."""
    index_entry_bytes = n_inner_chunks * 16
    total_index_bytes = index_entry_bytes + 4  # + crc32c

    with open(shard_path, "rb") as f:
        f.seek(-total_index_bytes, 2)
        index_bytes = f.read(index_entry_bytes)

    populated = []
    for i in range(n_inner_chunks):
        offset, nbytes = struct.unpack_from("<QQ", index_bytes, i * 16)
        if offset != SHARD_INDEX_EMPTY or nbytes != SHARD_INDEX_EMPTY:
            populated.append(i)
    return populated


# ---------------------------------------------------------------------------
# Variable classification
# ---------------------------------------------------------------------------


def _classify_variables(root, has_beams):
    """Classify data variables into beam and scalar categories."""
    coord_names = {
        "swath_time",
        "spacecraft",
        "beam",
        "gpi",
        "longitude",
        "latitude",
        "obs",
        "n_obs",
        "_slot_index",
    }
    beam_vars = set()
    scalar_vars = set()

    for name in root:
        if name in coord_names:
            continue
        arr = root[name]
        if not hasattr(arr, "ndim"):
            continue
        if has_beams and arr.ndim == 4:
            beam_vars.add(name)
        elif arr.ndim == 3:
            scalar_vars.add(name)

    return beam_vars, scalar_vars


def _classify_intermediate_variables(root, has_beams):
    """Classify variables in the intermediate (obs, gpi) / (obs, beam, gpi) store."""
    coord_names = {"gpi", "longitude", "latitude", "beam", "obs", "n_obs"}
    beam_vars = set()
    scalar_vars = set()

    for name in root:
        if name in coord_names:
            continue
        arr = root[name]
        if not hasattr(arr, "ndim"):
            continue
        if has_beams and arr.ndim == 3:
            beam_vars.add(name)
        elif arr.ndim == 2:
            scalar_vars.add(name)

    return beam_vars, scalar_vars


# ---------------------------------------------------------------------------
# Populated-chunk scanning
# ---------------------------------------------------------------------------


def _scan_all_populated_slots(
    sparse_path, n_swath_time, n_spacecraft, n_gpi, sparse_gpi_chunk_size
):
    """Scan the sparse store and return all populated (swath_time, spacecraft) slots."""
    sparse_root = zarr.open(str(sparse_path), mode="r")
    time_meta = sparse_root["time"].metadata
    is_sharded = any(
        getattr(c, "name", None) == "sharding_indexed"
        or type(c).__name__ == "ShardingCodec"
        for c in (time_meta.codecs or [])
    )

    all_slots = set()

    if is_sharded:
        shard_size_gpi = time_meta.chunk_grid.chunk_shape[-1]
        n_gpi_shards = -(-n_gpi // shard_size_gpi)

        def _scan_one(t_idx, s_idx, shard_idx):
            shard_path = (
                sparse_path / "time" / "c" / str(t_idx) / str(s_idx) / str(shard_idx)
            )
            if not shard_path.exists():
                return None
            return (t_idx, s_idx)

        tasks = [
            (t_idx, s_idx, shard_idx)
            for t_idx in range(n_swath_time)
            for s_idx in range(n_spacecraft)
            for shard_idx in range(n_gpi_shards)
        ]

        with ThreadPoolExecutor(max_workers=32) as pool:
            for result in tqdm(
                pool.map(lambda t: _scan_one(*t), tasks),
                total=len(tasks),
                desc="Scanning populated slots",
            ):
                if result is not None:
                    all_slots.add(result)
    else:
        n_gpi_chunks = -(-n_gpi // sparse_gpi_chunk_size)
        for t_idx in range(n_swath_time):
            for s_idx in range(n_spacecraft):
                for gc in range(n_gpi_chunks):
                    chunk_path = (
                        sparse_path / "time" / "c" / str(t_idx) / str(s_idx) / str(gc)
                    )
                    if chunk_path.exists():
                        all_slots.add((t_idx, s_idx))
                        break

    return sorted(all_slots)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_up(n, multiple):
    """Round n up to the nearest multiple."""
    return -(-n // multiple) * multiple


def _valid_time_mask(time_data, time_fill):
    """Return a boolean mask of "valid" time entries.

    Handles the NaT quirk: under numpy.datetime64, ``NaT != NaT`` returns
    True, so a naive ``time_data != time_fill`` mask would treat NaT
    entries as valid.  This helper checks ``np.isnat`` for datetime
    arrays (and similar quirks for floats) in addition to fill-value
    inequality.
    """
    if np.issubdtype(time_data.dtype, np.datetime64) or np.issubdtype(
        time_data.dtype, np.timedelta64
    ):
        return ~np.isnat(time_data)
    if np.issubdtype(time_data.dtype, np.floating):
        return ~np.isnan(time_data) & (time_data != time_fill)
    return time_data != time_fill


# ---------------------------------------------------------------------------
# Shared-memory read worker for parallel decompression
# ---------------------------------------------------------------------------


def _reset_codec_pipeline_worker():
    """Reset codec pipeline to default zarr-python in worker processes."""
    try:
        import zarr

        zarr.config.set(
            {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
        )
    except Exception:
        pass


def _shm_read_worker(args):
    """Read one (t_idx, s_idx) slab for ALL variables into shared memory."""
    (
        sparse_path,
        t_idx,
        s_idx,
        row_idx,
        batch_actual,
        n_gpi,
        scalar_shm_info,
        beam_shm_info,
    ) = args

    root = zarr.open(sparse_path, mode="r")

    for var, shm_name, dtype_str in scalar_shm_info:
        shm = SharedMemory(name=shm_name)
        dtype = np.dtype(dtype_str)
        try:
            buf = np.ndarray((batch_actual, n_gpi), dtype=dtype, buffer=shm.buf)
            buf[row_idx, :] = root[var][t_idx, s_idx, :]
        finally:
            shm.close()

    for var, shm_name, dtype_str, n_beams in beam_shm_info:
        shm = SharedMemory(name=shm_name)
        dtype = np.dtype(dtype_str)
        try:
            buf = np.ndarray(
                (batch_actual, n_beams, n_gpi), dtype=dtype, buffer=shm.buf
            )
            buf[row_idx, :, :] = root[var][t_idx, s_idx, :, :]
        finally:
            shm.close()


# ---------------------------------------------------------------------------
# Pass 1: Rechunk
# ---------------------------------------------------------------------------


def rechunk_sparse(
    sparse_path,
    intermediate_path,
    target_gpi_chunk=64,
    batch_size=500,
    n_read_threads=16,
    date_range=None,
):
    """Rechunk a sparse swath-time Zarr store into an intermediate obs-major store.

    Parameters
    ----------
    sparse_path : str or Path
        Path to the sparse Zarr store.
    intermediate_path : str or Path
        Path for the intermediate rechunked Zarr store.
    target_gpi_chunk : int
        Spatial chunk size in the intermediate store.  Default 64.
    batch_size : int
        Number of time slots to read into memory at once.  Default 500.
    n_read_threads : int
        Number of worker processes for reads + threads for parallel
        writes.  Default 16.
    date_range : tuple of (datetime, datetime), optional
        If given, only ``(swath_time, spacecraft)`` slots whose
        ``swath_time`` falls within ``[start, end)`` are rechunked.
        Useful for incremental appends where the sparse store still
        contains older data.  Default None (all populated slots).
    """
    sparse_path = Path(sparse_path)
    intermediate_path = Path(intermediate_path)
    sparse_root = zarr.open(sparse_path, mode="r")

    has_beams = "beam" in sparse_root
    n_gpi = sparse_root["gpi"].shape[0]
    n_spacecraft = sparse_root["spacecraft"].shape[0]
    n_swath_time = sparse_root["swath_time"].shape[0]
    sparse_gpi_chunk_size = sparse_root["time"].chunks[-1]

    beam_vars, scalar_vars = _classify_variables(sparse_root, has_beams)
    all_vars_scalar = sorted(scalar_vars)
    all_vars_beam = sorted(beam_vars) if has_beams else []
    print(f"Scalar variables: {all_vars_scalar}")
    print(f"Beam variables: {all_vars_beam}")

    print("Scanning for populated slots...")
    scan_start = timer()
    all_slots = _scan_all_populated_slots(
        sparse_path,
        n_swath_time,
        n_spacecraft,
        n_gpi,
        sparse_gpi_chunk_size,
    )
    n_obs_total_scanned = len(all_slots)
    print(
        f"Scan complete in {timer() - scan_start:.1f}s, "
        f"found {n_obs_total_scanned} populated slots"
    )

    # --- Optional date filtering ---
    if date_range is not None:
        start_dt, end_dt = date_range
        # Read swath_time coordinate to map t_idx -> timestamp
        swath_time_arr = sparse_root["swath_time"][:]
        # Convert to numpy datetime64 if it isn't already, for safe comparison.
        if not np.issubdtype(swath_time_arr.dtype, np.datetime64):
            swath_time_arr = np.asarray(swath_time_arr, dtype="datetime64[ns]")
        start64 = np.datetime64(start_dt, "ns")
        end64 = np.datetime64(end_dt, "ns")

        in_range_mask = (swath_time_arr >= start64) & (swath_time_arr < end64)
        all_slots = [
            (t_idx, s_idx) for (t_idx, s_idx) in all_slots if in_range_mask[t_idx]
        ]
        print(
            f"After date filter [{start_dt}, {end_dt}): "
            f"{len(all_slots)}/{n_obs_total_scanned} slots remain"
        )

    n_obs = len(all_slots)

    if n_obs == 0:
        raise ValueError(
            "No populated slots remain after filtering — nothing to rechunk."
        )

    padded_n_obs = _round_up(n_obs, batch_size)

    print(f"Creating intermediate store at {intermediate_path}")
    print(
        f"  obs={padded_n_obs}, gpi={n_gpi}, chunks=({batch_size}, {target_gpi_chunk})"
    )

    store = zarr.storage.LocalStore(str(intermediate_path))
    int_root = zarr.create_group(store=store, overwrite=True, zarr_format=3)
    int_root.attrs.update(dict(sparse_root.attrs))

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)

    for var in all_vars_scalar:
        src = sparse_root[var]
        int_root.create_array(
            name=var,
            dtype=src.dtype,
            shape=(padded_n_obs, n_gpi),
            chunks=(batch_size, target_gpi_chunk),
            fill_value=src.metadata.fill_value,
            attributes=dict(src.attrs),
            dimension_names=["obs", "gpi"],
            compressors=compressor,
        )

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in all_vars_beam:
            src = sparse_root[var]
            int_root.create_array(
                name=var,
                dtype=src.dtype,
                shape=(padded_n_obs, n_beams, n_gpi),
                chunks=(batch_size, 1, target_gpi_chunk),
                fill_value=src.metadata.fill_value,
                attributes=dict(src.attrs),
                dimension_names=["obs", "beam", "gpi"],
                compressors=compressor,
            )

    sat_id = np.full((padded_n_obs, n_gpi), -128, dtype="int8")
    for i in range(n_obs):
        sat_id[i, :] = all_slots[i][1]
    int_root.create_array(
        "sat_id",
        data=sat_id,
        fill_value=-128,
        chunks=(batch_size, target_gpi_chunk),
        dimension_names=["obs", "gpi"],
        compressors=compressor,
    )

    for coord in ["gpi", "longitude", "latitude"]:
        if coord in sparse_root:
            int_root.create_array(
                coord,
                data=sparse_root[coord][:],
                attributes=dict(sparse_root[coord].attrs),
                dimension_names=["gpi"],
            )
    if has_beams and "beam" in sparse_root:
        int_root.create_array(
            "beam",
            data=sparse_root["beam"][:],
            attributes=dict(sparse_root["beam"].attrs),
            dimension_names=["beam"],
        )

    n_batches = -(-n_obs // batch_size)
    if n_batches == 1:
        batch_size = n_obs
    print(f"Processing {n_obs} slots in {n_batches} batches of {batch_size}")
    total_start = timer()

    var_info_scalar = [
        (var, str(sparse_root[var].dtype), sparse_root[var].metadata.fill_value)
        for var in all_vars_scalar
    ]
    var_info_beam = (
        [
            (
                var,
                str(sparse_root[var].dtype),
                sparse_root[var].metadata.fill_value,
                sparse_root["beam"].shape[0],
            )
            for var in all_vars_beam
        ]
        if has_beams
        else []
    )

    print(f"Allocating shared memory buffers for batch_size={batch_size}")
    scalar_shms = []
    for var, dtype_str, fill_val in var_info_scalar:
        dtype = np.dtype(dtype_str)
        buf_size = batch_size * n_gpi * dtype.itemsize
        shm = SharedMemory(create=True, size=buf_size)
        buf = np.ndarray((batch_size, n_gpi), dtype=dtype, buffer=shm.buf)
        scalar_shms.append((var, shm, buf, dtype_str, fill_val))

    beam_shms = []
    for var, dtype_str, fill_val, n_beams in var_info_beam:
        dtype = np.dtype(dtype_str)
        buf_size = batch_size * n_beams * n_gpi * dtype.itemsize
        shm = SharedMemory(create=True, size=buf_size)
        buf = np.ndarray((batch_size, n_beams, n_gpi), dtype=dtype, buffer=shm.buf)
        beam_shms.append((var, shm, buf, dtype_str, n_beams, fill_val))

    scalar_shm_info = [
        (var, shm.name, dtype_str) for var, shm, buf, dtype_str, fill_val in scalar_shms
    ]
    beam_shm_info = [
        (var, shm.name, dtype_str, n_beams)
        for var, shm, buf, dtype_str, n_beams, fill_val in beam_shms
    ]

    try:
        with ProcessPoolExecutor(
            max_workers=n_read_threads,
            initializer=_reset_codec_pipeline_worker,
        ) as pool:
            for batch_idx in range(n_batches):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min(batch_start_idx + batch_size, n_obs)
                batch = all_slots[batch_start_idx:batch_end_idx]
                actual = len(batch)

                batch_timer = timer()

                for var, shm, buf, dtype_str, fill_val in scalar_shms:
                    buf[:actual] = fill_val
                for var, shm, buf, dtype_str, n_beams, fill_val in beam_shms:
                    buf[:actual] = fill_val

                fill_elapsed = timer() - batch_timer

                tasks = [
                    (
                        str(sparse_path),
                        t_idx,
                        s_idx,
                        i,
                        batch_size,
                        n_gpi,
                        scalar_shm_info,
                        beam_shm_info,
                    )
                    for i, (t_idx, s_idx) in enumerate(batch)
                ]

                read_timer = timer()
                list(pool.map(_shm_read_worker, tasks))
                read_elapsed = timer() - read_timer

                write_timer = timer()
                obs_start = batch_start_idx
                obs_end = batch_start_idx + actual

                def _write_scalar(item):
                    var, shm, buf, dtype_str, fill_val = item
                    int_root[var][obs_start:obs_end, :] = buf[:actual]

                def _write_beam(item):
                    var, shm, buf, dtype_str, n_beams, fill_val = item
                    int_root[var][obs_start:obs_end, :, :] = buf[:actual]

                with ThreadPoolExecutor(max_workers=n_read_threads) as write_pool:
                    list(write_pool.map(_write_scalar, scalar_shms))
                    if beam_shms:
                        list(write_pool.map(_write_beam, beam_shms))

                write_elapsed = timer() - write_timer

                print(
                    f"  Batch {batch_idx + 1}/{n_batches}: {actual} slots, "
                    f"fill {fill_elapsed:.1f}s, "
                    f"read {read_elapsed:.1f}s, write {write_elapsed:.1f}s"
                )

        total_elapsed = timer() - total_start
        print(f"Rechunking complete in {total_elapsed:.1f}s")
    finally:
        for var, shm, buf, dtype_str, fill_val in scalar_shms:
            shm.close()
            shm.unlink()
        for var, shm, buf, dtype_str, n_beams, fill_val in beam_shms:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------
# Pass 2: Dense conversion from intermediate (create + append modes)
# ---------------------------------------------------------------------------


def _create_dense_structure_from_intermediate(
    out_path,
    int_root,
    beam_vars,
    scalar_vars,
    has_beams,
    n_gpi,
    obs_dim_size,
    chunk_size_gpi,
    chunk_size_obs,
    sharding: Optional[_ShardingConfig] = None,
):
    """Create the empty dense Zarr store for the final time-series output."""
    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    root.attrs.update(dict(int_root.attrs))

    if has_beams:
        n_beams = int_root["beam"].shape[0]
        for var in sorted(beam_vars):
            src = int_root[var]
            arr_kwargs = _make_array_kwargs(
                (chunk_size_gpi, chunk_size_obs, 1), sharding
            )
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
        src = int_root[var]
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
        dimension_names=("gpi",),
        fill_value=0,
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)],
    )

    root.create_array(
        "obs",
        data=np.arange(obs_dim_size, dtype="int32"),
        dimension_names=("obs",),
        fill_value=dtype_to_nan[np.dtype("int32")],
        compressors=None,
    )

    root.create_array(
        "gpi",
        data=int_root["gpi"][:],
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("int32")],
        attributes=dict(int_root["gpi"].attrs) if "gpi" in int_root else {},
        compressors=None,
    )

    root.create_array(
        "longitude",
        data=int_root["longitude"][:],
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("float32")],
        attributes=dict(int_root["longitude"].attrs) if "longitude" in int_root else {},
        compressors=None,
    )

    root.create_array(
        "latitude",
        data=int_root["latitude"][:],
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("float32")],
        attributes=dict(int_root["latitude"].attrs) if "latitude" in int_root else {},
        compressors=None,
    )

    if has_beams and "beam" in int_root:
        root.create_array(
            "beam",
            data=int_root["beam"][:],
            dimension_names=("beam",),
            fill_value=b"",
            attributes=dict(int_root["beam"].attrs),
            compressors=None,
        )


def _expand_obs_dimension(
    out_root, needed_size, obs_alignment, beam_vars, scalar_vars, has_beams
):
    """Expand the obs dimension in alignment-sized increments.

    ``obs_alignment`` is the shard obs size when sharding is active, or
    the plain chunk obs size otherwise, ensuring expansions always land
    on a shard/chunk boundary.
    """
    current_size = out_root["obs"].shape[0]
    new_size = _round_up(needed_size, obs_alignment)

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


def _sort_and_compact_slab(
    time_data,
    time_fill,
    scalar_data,
    beam_data,
    scalar_vars,
    beam_vars,
    has_beams,
    chunk_width,
):
    """Sort per-GPI valid observations by time and pack into compact arrays.

    Returns
    -------
    n_valid_per_gpi : (chunk_width,) int32
    new_time : (chunk_width, max_new) sorted time values
    new_scalars : dict var -> (chunk_width, max_new) sorted scalar values
    new_beams : dict var -> (chunk_width, max_new, n_beams) sorted beam values
    """
    # NaT comparison quirk: NaT != NaT is True under datetime64, so a
    # plain `!=` mask treats NaT as valid.  Use _valid_time_mask which
    # handles both datetime/NaT and integer fill semantics correctly.
    valid_mask = _valid_time_mask(time_data, time_fill)
    n_valid_per_gpi = valid_mask.sum(axis=0).astype(np.int32)
    max_new = int(n_valid_per_gpi.max()) if n_valid_per_gpi.size else 0

    if max_new == 0:
        return n_valid_per_gpi, None, {}, {}

    new_time = np.full((chunk_width, max_new), time_fill, dtype=time_data.dtype)
    new_scalars = {}
    for var in sorted(scalar_vars):
        if var == "time":
            continue
        src = scalar_data[var]
        new_scalars[var] = np.full(
            (chunk_width, max_new), scalar_data[var + "__fill"], dtype=src.dtype
        )

    new_beams = {}
    if has_beams:
        n_beams = None
        for var in sorted(beam_vars):
            src = beam_data[var]
            if n_beams is None:
                n_beams = src.shape[1]
            new_beams[var] = np.full(
                (chunk_width, max_new, n_beams),
                beam_data[var + "__fill"],
                dtype=src.dtype,
            )

    for g in range(chunk_width):
        n_v = int(n_valid_per_gpi[g])
        if n_v == 0:
            continue
        valid = valid_mask[:, g]
        times_g = time_data[valid, g]
        order = np.argsort(times_g)

        new_time[g, :n_v] = times_g[order]
        for var in sorted(scalar_vars):
            if var == "time":
                continue
            new_scalars[var][g, :n_v] = scalar_data[var][valid, g][order]
        if has_beams:
            for var in sorted(beam_vars):
                new_beams[var][g, :n_v, :] = beam_data[var][valid, :, g][order, :]

    return n_valid_per_gpi, new_time, new_scalars, new_beams


def _read_intermediate_slab(
    int_root, gpi_start, gpi_end, scalar_vars, beam_vars, has_beams
):
    """Read the full (obs, gpi-range) slab for this spatial chunk.

    Returns dictionaries including fill values, so the sort/compact
    helper can construct output arrays without re-reading metadata.
    """
    time_data = int_root["time"][:, gpi_start:gpi_end]
    time_fill = int_root["time"].metadata.fill_value

    scalar_data = {"time": time_data, "time__fill": time_fill}
    for var in sorted(scalar_vars):
        if var == "time":
            continue
        scalar_data[var] = int_root[var][:, gpi_start:gpi_end]
        scalar_data[var + "__fill"] = int_root[var].metadata.fill_value

    beam_data = {}
    if has_beams:
        for var in sorted(beam_vars):
            beam_data[var] = int_root[var][:, :, gpi_start:gpi_end]
            beam_data[var + "__fill"] = int_root[var].metadata.fill_value

    return time_data, time_fill, scalar_data, beam_data


def _process_spatial_chunk_create_mode(
    gpi_range,
    intermediate_path,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
):
    """Fresh-store variant: write starting at obs=0 for every GPI."""
    gpi_start, gpi_end = gpi_range
    chunk_width = gpi_end - gpi_start

    int_root = zarr.open(intermediate_path, mode="r")
    out_root = zarr.open(out_path, mode="a")

    time_data, time_fill, scalar_data, beam_data = _read_intermediate_slab(
        int_root,
        gpi_start,
        gpi_end,
        scalar_vars,
        beam_vars,
        has_beams,
    )

    n_valid_per_gpi, new_time, new_scalars, new_beams = _sort_and_compact_slab(
        time_data,
        time_fill,
        scalar_data,
        beam_data,
        scalar_vars,
        beam_vars,
        has_beams,
        chunk_width,
    )

    max_valid = int(n_valid_per_gpi.max()) if n_valid_per_gpi.size else 0
    if max_valid == 0:
        return

    obs_end = max_valid
    out_root["time"][gpi_start:gpi_end, :obs_end] = new_time
    for var in sorted(scalar_vars):
        if var == "time":
            continue
        out_root[var][gpi_start:gpi_end, :obs_end] = new_scalars[var]
    if has_beams:
        for var in sorted(beam_vars):
            out_root[var][gpi_start:gpi_end, :obs_end, :] = new_beams[var]

    out_root["n_obs"][gpi_start:gpi_end] = n_valid_per_gpi.astype("uint32")


def _process_spatial_chunk_append_mode(
    gpi_range,
    intermediate_path,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
    obs_alignment,
    do_continuity_check,
):
    """Append-mode variant: each GPI starts writing at existing n_obs[gpi].

    Uses obs-chunk-aligned read-modify-write loops so that each cycle
    affects exactly one (gpi_range × obs_chunk) block of storage.

    When ``do_continuity_check`` is True, verifies that no GPI's first
    new timestamp is earlier than its last existing timestamp (raises
    ValueError on violation, before any writes are issued).

    Parameters
    ----------
    obs_alignment : int
        Shard obs size (if sharding) or plain chunk obs size.  The RMW
        loop walks the obs axis in steps of this size.
    do_continuity_check : bool
        Whether to perform the per-chunk continuity check.

    Returns
    -------
    dict with keys:
        max_fwd_gap : int — largest forward gap (new vs existing time) in
                            this GPI range; 0 if no overlap to check.
    """
    gpi_start, gpi_end = gpi_range
    chunk_width = gpi_end - gpi_start

    int_root = zarr.open(intermediate_path, mode="r")
    out_root = zarr.open(out_path, mode="a")

    time_data, time_fill, scalar_data, beam_data = _read_intermediate_slab(
        int_root,
        gpi_start,
        gpi_end,
        scalar_vars,
        beam_vars,
        has_beams,
    )

    n_valid_per_gpi, new_time, new_scalars, new_beams = _sort_and_compact_slab(
        time_data,
        time_fill,
        scalar_data,
        beam_data,
        scalar_vars,
        beam_vars,
        has_beams,
        chunk_width,
    )

    if n_valid_per_gpi.sum() == 0:
        return {"max_fwd_gap": 0}

    # --- Determine write positions from existing n_obs ---
    existing_n_obs = out_root["n_obs"][gpi_start:gpi_end].astype(np.int32)
    write_starts = existing_n_obs.copy()
    new_counts = n_valid_per_gpi
    write_ends = write_starts + new_counts

    active = new_counts > 0
    if not active.any():
        return {"max_fwd_gap": 0}

    # --- Continuity check for this GPI range ---
    # Done before any writes so a violation aborts cleanly.
    max_fwd_gap = 0
    if do_continuity_check:
        need_check = (existing_n_obs > 0) & active
        if need_check.any():
            # One slab read covers all the last-existing positions.
            last_positions = existing_n_obs - 1
            check_gpis = np.where(need_check)[0]
            last_pos_active = last_positions[need_check]
            lo = int(last_pos_active.min())
            hi = int(last_pos_active.max()) + 1
            last_slab = out_root["time"][gpi_start:gpi_end, lo:hi]

            violations = 0
            max_back_gap = 0
            for g_local in check_gpis:
                last_existing = last_slab[g_local, last_positions[g_local] - lo]
                # new_time is sorted ascending, so [g, 0] is the minimum
                min_new = new_time[g_local, 0]

                if min_new < last_existing:
                    violations += 1
                    gap = last_existing - min_new
                    if gap > max_back_gap:
                        max_back_gap = gap
                else:
                    gap = min_new - last_existing
                    if gap > max_fwd_gap:
                        max_fwd_gap = gap

            if violations > 0:
                raise ValueError(
                    f"Continuity check failed in GPI range "
                    f"[{gpi_start}, {gpi_end}): {violations} GPIs have new "
                    f"data earlier than their last existing observation "
                    f"(max backwards gap: {max_back_gap})."
                )

    # --- Compute window over obs axis that needs touching ---
    min_obs = int(write_starts[active].min())
    max_obs = int(write_ends.max())

    first_obs_chunk = min_obs // obs_alignment
    last_obs_chunk = (max_obs - 1) // obs_alignment

    gpi_slice = slice(gpi_start, gpi_end)
    out_obs_size = out_root["obs"].shape[0]

    # --- RMW loop over obs chunks ---
    for obs_chunk_idx in range(first_obs_chunk, last_obs_chunk + 1):
        obs_chunk_start = obs_chunk_idx * obs_alignment
        obs_chunk_end = min(obs_chunk_start + obs_alignment, out_obs_size)
        obs_slice = slice(obs_chunk_start, obs_chunk_end)

        gpi_needs_write = (
            (write_starts < obs_chunk_end) & (write_ends > obs_chunk_start) & active
        )
        if not gpi_needs_write.any():
            continue

        # Time variable
        chunk_time = out_root["time"][gpi_slice, obs_slice]
        _scatter_into_chunk(
            chunk_time,
            new_time,
            write_starts,
            new_counts,
            gpi_needs_write,
            obs_chunk_start,
            obs_chunk_end,
        )
        out_root["time"][gpi_slice, obs_slice] = chunk_time

        # Other scalar variables
        for var in sorted(scalar_vars):
            if var == "time":
                continue
            chunk_data = out_root[var][gpi_slice, obs_slice]
            _scatter_into_chunk(
                chunk_data,
                new_scalars[var],
                write_starts,
                new_counts,
                gpi_needs_write,
                obs_chunk_start,
                obs_chunk_end,
            )
            out_root[var][gpi_slice, obs_slice] = chunk_data

        # Beam variables
        if has_beams:
            for var in sorted(beam_vars):
                chunk_data = out_root[var][gpi_slice, obs_slice, :]
                _scatter_beam_into_chunk(
                    chunk_data,
                    new_beams[var],
                    write_starts,
                    new_counts,
                    gpi_needs_write,
                    obs_chunk_start,
                    obs_chunk_end,
                )
                out_root[var][gpi_slice, obs_slice, :] = chunk_data

    # --- Update n_obs ---
    out_root["n_obs"][gpi_start:gpi_end] = write_ends.astype("uint32")

    return {"max_fwd_gap": max_fwd_gap}


def _scatter_into_chunk(
    chunk_data,
    new_data,
    write_starts,
    new_counts,
    gpi_needs_write,
    obs_chunk_start,
    obs_chunk_end,
):
    """Scatter per-GPI new data into a single (gpi, obs) output chunk slice.

    chunk_data : (chunk_width, obs_chunk_size) — modified in place
    new_data : (chunk_width, max_new) — compacted source
    write_starts : (chunk_width,) — global obs offset where each GPI writes
    new_counts : (chunk_width,) — number of new obs per GPI
    gpi_needs_write : (chunk_width,) bool — GPIs with overlap in this chunk
    """
    chunk_width = chunk_data.shape[0]
    for g in range(chunk_width):
        if not gpi_needs_write[g]:
            continue
        ws = int(write_starts[g])
        nc = int(new_counts[g])

        # Intersection with this obs chunk window
        src_lo = max(0, obs_chunk_start - ws)
        src_hi = min(nc, obs_chunk_end - ws)
        if src_hi <= src_lo:
            continue
        dst_lo = ws + src_lo - obs_chunk_start
        dst_hi = dst_lo + (src_hi - src_lo)

        chunk_data[g, dst_lo:dst_hi] = new_data[g, src_lo:src_hi]


def _scatter_beam_into_chunk(
    chunk_data,
    new_data,
    write_starts,
    new_counts,
    gpi_needs_write,
    obs_chunk_start,
    obs_chunk_end,
):
    """Beam variant of _scatter_into_chunk.

    chunk_data : (chunk_width, obs_chunk_size, n_beams)
    new_data : (chunk_width, max_new, n_beams)
    """
    chunk_width = chunk_data.shape[0]
    for g in range(chunk_width):
        if not gpi_needs_write[g]:
            continue
        ws = int(write_starts[g])
        nc = int(new_counts[g])

        src_lo = max(0, obs_chunk_start - ws)
        src_hi = min(nc, obs_chunk_end - ws)
        if src_hi <= src_lo:
            continue
        dst_lo = ws + src_lo - obs_chunk_start
        dst_hi = dst_lo + (src_hi - src_lo)

        chunk_data[g, dst_lo:dst_hi, :] = new_data[g, src_lo:src_hi, :]


def densify_from_intermediate(
    intermediate_path,
    out_path,
    chunk_size_gpi=64,
    chunk_size_obs=300,
    n_workers=16,
    shard_size_gpi=None,
    shard_size_obs=None,
    skip_continuity_check=False,
    gap_warn_seconds=86400 * 7,
):
    """Convert the intermediate rechunked store to a dense time-series store.

    If the output store already exists, this function runs in *append
    mode*: new observations from the intermediate are appended to the
    existing per-GPI time series at position ``n_obs[gpi]``, and
    ``n_obs`` is updated.  The existing chunking/sharding is detected
    and reused automatically.

    If the output store does not exist, it is created fresh.

    Parameters
    ----------
    intermediate_path : str or Path
        Path to the intermediate Zarr store from ``rechunk_sparse``.
    out_path : str or Path
        Path for the output dense Zarr store (gpi, obs, [beam]).
    chunk_size_gpi : int
        Chunk size along the gpi dimension.  Ignored in append mode
        (the existing store's chunking is used).  Default 64.
    chunk_size_obs : int
        Chunk size along the obs dimension.  Ignored in append mode.
        Default 300.
    n_workers : int
        Number of parallel workers.  Default 16.
    shard_size_gpi : int, optional
        GPIs per shard.  Ignored in append mode.  Default None.
    shard_size_obs : int, optional
        Obs per shard.  Ignored in append mode.  Default None.
    skip_continuity_check : bool
        Skip the per-chunk continuity check in append mode.
        Default False.
    gap_warn_seconds : int
        Warn after the append if the largest forward gap between
        existing and new data exceeds this many seconds.  Default
        604800 (1 week).
    """
    intermediate_path = Path(intermediate_path)
    out_path = Path(out_path)

    int_root = zarr.open(intermediate_path, mode="r")
    has_beams = "beam" in int_root
    n_gpi = int_root["gpi"].shape[0]

    beam_vars, scalar_vars = _classify_intermediate_variables(int_root, has_beams)
    print(f"Scalar variables: {sorted(scalar_vars)}")
    print(f"Beam variables: {sorted(beam_vars)}")

    append_mode = (out_path / "zarr.json").exists()

    if append_mode:
        print(f"Output store exists at {out_path} — running in APPEND mode")
        _densify_append(
            intermediate_path=intermediate_path,
            out_path=out_path,
            int_root=int_root,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            n_gpi=n_gpi,
            n_workers=n_workers,
            skip_continuity_check=skip_continuity_check,
            gap_warn_seconds=gap_warn_seconds,
        )
    else:
        print(f"Creating new dense store at {out_path}")
        _densify_create(
            intermediate_path=intermediate_path,
            out_path=out_path,
            int_root=int_root,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            n_gpi=n_gpi,
            chunk_size_gpi=chunk_size_gpi,
            chunk_size_obs=chunk_size_obs,
            n_workers=n_workers,
            shard_size_gpi=shard_size_gpi,
            shard_size_obs=shard_size_obs,
        )


def _densify_create(
    intermediate_path,
    out_path,
    int_root,
    beam_vars,
    scalar_vars,
    has_beams,
    n_gpi,
    chunk_size_gpi,
    chunk_size_obs,
    n_workers,
    shard_size_gpi,
    shard_size_obs,
):
    """Create a new dense store from the intermediate."""
    if shard_size_gpi is not None:
        sharding = _ShardingConfig(
            shard_size_gpi=shard_size_gpi,
            inner_chunk_gpi=chunk_size_gpi,
            shard_size_obs=shard_size_obs,
            inner_chunk_obs=chunk_size_obs,
        )
    else:
        sharding = None

    obs_alignment = sharding.obs_alignment if sharding else chunk_size_obs
    worker_gpi_size = sharding.gpi_alignment if sharding else chunk_size_gpi

    n_obs_total = int_root["time"].shape[0]
    obs_dim_size = _round_up(n_obs_total, obs_alignment)

    print(f"  n_gpi={n_gpi}, obs_dim_size={obs_dim_size}")
    print(f"  chunks=({chunk_size_gpi}, {chunk_size_obs})")
    if sharding:
        print(f"  shards=({shard_size_gpi}, {shard_size_obs})")

    _create_dense_structure_from_intermediate(
        out_path=out_path,
        int_root=int_root,
        beam_vars=beam_vars,
        scalar_vars=scalar_vars,
        has_beams=has_beams,
        n_gpi=n_gpi,
        obs_dim_size=obs_dim_size,
        chunk_size_gpi=chunk_size_gpi,
        chunk_size_obs=chunk_size_obs,
        sharding=sharding,
    )

    gpi_chunks = [
        (i, min(i + worker_gpi_size, n_gpi)) for i in range(0, n_gpi, worker_gpi_size)
    ]
    print(
        f"Processing {len(gpi_chunks)} GPI chunks with {n_workers} workers"
        f" (worker size: {worker_gpi_size})"
    )

    process_func = partial(
        _process_spatial_chunk_create_mode,
        intermediate_path=str(intermediate_path),
        out_path=str(out_path),
        beam_vars=beam_vars,
        scalar_vars=scalar_vars,
        has_beams=has_beams,
    )

    start = timer()
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in gpi_chunks]
            for f in tqdm(futures, total=len(gpi_chunks), desc="Spatial chunks"):
                f.result()
    else:
        for chunk in tqdm(gpi_chunks, desc="Spatial chunks"):
            process_func(chunk)

    print(f"Dense conversion complete in {timer() - start:.1f}s")


def _densify_append(
    intermediate_path,
    out_path,
    int_root,
    beam_vars,
    scalar_vars,
    has_beams,
    n_gpi,
    n_workers,
    skip_continuity_check,
    gap_warn_seconds=86400 * 7,
):
    """Append the intermediate into an existing dense store.

    Performs the time-continuity check inside each append worker (in
    parallel with the actual append work), aggregating forward-gap
    statistics in the parent process so a global warning can be emitted
    if the largest gap exceeds ``gap_warn_seconds``.
    """
    out_root = zarr.open(out_path, mode="a")

    if out_root["gpi"].shape[0] != n_gpi:
        raise ValueError(
            f"GPI count mismatch: intermediate has {n_gpi}, "
            f"dense store has {out_root['gpi'].shape[0]}"
        )

    # Detect existing chunking and sharding from the "time" array.
    time_arr = out_root["time"]

    shard_shape = None
    for c in time_arr.metadata.codecs or []:
        if (
            getattr(c, "name", None) == "sharding_indexed"
            or type(c).__name__ == "ShardingCodec"
        ):
            # When the sharding codec is active, the array's chunk_grid
            # chunk_shape is the shard shape, and the codec's chunk_shape
            # attribute is the inner chunk shape.
            shard_shape = time_arr.metadata.chunk_grid.chunk_shape
            break

    if shard_shape is not None:
        shard_size_gpi = shard_shape[0]
        shard_size_obs = shard_shape[1]
        obs_alignment = shard_size_obs
        worker_gpi_size = shard_size_gpi
        print(f"  Detected sharding: shards=({shard_size_gpi}, {shard_size_obs})")
    else:
        obs_alignment = time_arr.chunks[1]
        worker_gpi_size = time_arr.chunks[0]
        print(f"  No sharding, chunks=({worker_gpi_size}, {obs_alignment})")

    # --- Figure out how much we need to expand the obs dimension ---
    print("Computing required obs dimension size...")
    existing_n_obs_all = out_root["n_obs"][:]

    int_time = int_root["time"]
    int_time_fill = int_time.metadata.fill_value
    int_chunk_gpi = int_time.chunks[-1]

    max_new_per_gpi = np.zeros(n_gpi, dtype=np.int64)
    for gs in tqdm(range(0, n_gpi, int_chunk_gpi), desc="Counting new obs"):
        ge = min(gs + int_chunk_gpi, n_gpi)
        slab = int_time[:, gs:ge]
        max_new_per_gpi[gs:ge] = (slab != int_time_fill).sum(axis=0)

    worst_case_needed = int(
        (existing_n_obs_all.astype(np.int64) + max_new_per_gpi).max()
    )
    current_obs_size = out_root["obs"].shape[0]

    if worst_case_needed > current_obs_size:
        _expand_obs_dimension(
            out_root,
            needed_size=worst_case_needed,
            obs_alignment=obs_alignment,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
        )
    else:
        print(
            f"  obs dimension already sufficient "
            f"(current={current_obs_size}, needed={worst_case_needed})"
        )

    gpi_chunks = [
        (i, min(i + worker_gpi_size, n_gpi)) for i in range(0, n_gpi, worker_gpi_size)
    ]
    print(
        f"Appending {len(gpi_chunks)} GPI chunks with {n_workers} workers"
        f" (worker size: {worker_gpi_size})"
    )
    if skip_continuity_check:
        print("Per-chunk continuity check: DISABLED")
    else:
        print("Per-chunk continuity check: ENABLED")

    process_func = partial(
        _process_spatial_chunk_append_mode,
        intermediate_path=str(intermediate_path),
        out_path=str(out_path),
        beam_vars=beam_vars,
        scalar_vars=scalar_vars,
        has_beams=has_beams,
        obs_alignment=obs_alignment,
        do_continuity_check=not skip_continuity_check,
    )

    start = timer()
    max_fwd_gap_seen = 0

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in gpi_chunks]
            for f in tqdm(futures, total=len(gpi_chunks), desc="Append chunks"):
                result = f.result()
                if result and result.get("max_fwd_gap", 0) > max_fwd_gap_seen:
                    max_fwd_gap_seen = result["max_fwd_gap"]
    else:
        for chunk in tqdm(gpi_chunks, desc="Append chunks"):
            result = process_func(chunk)
            if result and result.get("max_fwd_gap", 0) > max_fwd_gap_seen:
                max_fwd_gap_seen = result["max_fwd_gap"]

    print(f"Append complete in {timer() - start:.1f}s")

    if not skip_continuity_check:
        if max_fwd_gap_seen > gap_warn_seconds:
            warnings.warn(
                f"Largest forward gap between existing and new data is "
                f"{max_fwd_gap_seen} seconds "
                f"({max_fwd_gap_seen / 86400:.1f} days). "
                f"This may indicate missing data between the two runs.",
                stacklevel=2,
            )
        else:
            print(
                f"Continuity check OK across all chunks "
                f"(max forward gap: {max_fwd_gap_seen} seconds)."
            )


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def sparse_to_dense_rechunked(
    sparse_path,
    out_path,
    intermediate_path=None,
    target_gpi_chunk=64,
    batch_size=500,
    n_read_threads=16,
    chunk_size_gpi=64,
    chunk_size_obs=300,
    n_workers=16,
    shard_size_gpi=None,
    shard_size_obs=None,
    keep_intermediate=False,
    skip_continuity_check=False,
    date_range=None,
):
    """Convert sparse swath-time Zarr to dense time-series via rechunking.

    Creates the dense store if it doesn't exist, or appends to it if
    it does.  The creation-time chunking and sharding parameters are
    ignored when appending (the existing layout is preserved).

    Parameters
    ----------
    date_range : tuple of (datetime, datetime), optional
        Restrict pass 1 to swath_time slots within ``[start, end)``.
        Useful for incremental appends where the sparse store still
        contains older data that has already been densified.  Default
        None (process all populated slots).
    """
    sparse_path = Path(sparse_path)
    out_path = Path(out_path)
    if intermediate_path is None:
        intermediate_path = out_path.parent / (out_path.stem + "_intermediate")
    intermediate_path = Path(intermediate_path)

    print("=" * 60)
    print("Pass 1: Rechunking sparse -> intermediate")
    print("=" * 60)
    rechunk_sparse(
        sparse_path=sparse_path,
        intermediate_path=intermediate_path,
        target_gpi_chunk=target_gpi_chunk,
        batch_size=batch_size,
        n_read_threads=n_read_threads,
        date_range=date_range,
    )

    print()
    print("=" * 60)
    print("Pass 2: Intermediate -> dense time-series")
    print("=" * 60)
    densify_from_intermediate(
        intermediate_path=intermediate_path,
        out_path=out_path,
        chunk_size_gpi=chunk_size_gpi,
        chunk_size_obs=chunk_size_obs,
        n_workers=n_workers,
        shard_size_gpi=shard_size_gpi,
        shard_size_obs=shard_size_obs,
        skip_continuity_check=skip_continuity_check,
    )

    if not keep_intermediate:
        import shutil

        print(f"Cleaning up intermediate store: {intermediate_path}")
        shutil.rmtree(intermediate_path, ignore_errors=True)

    print("Done!")
