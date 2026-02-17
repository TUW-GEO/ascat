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
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import time as timer

import numpy as np
import zarr

from ascat.utils import dtype_to_nan


def sparse_to_dense(
    sparse_path,
    out_path,
    chunk_size_gpi=None,
    chunk_size_obs=300,
    n_workers=1,
    gpi_mask=None,
):
    """Convert a sparse swath-time Zarr cube to a dense time-series store.

    Parameters
    ----------
    sparse_path : str or Path
        Path to the sparse Zarr store with dims (swath_time, spacecraft, [beam,] gpi).
    out_path : str or Path
        Path for the output dense Zarr store with dims (gpi, obs, [beam]).
    chunk_size_gpi : int, optional
        Chunk size along the gpi dimension. Defaults to sparse store's GPI chunk size.
    chunk_size_obs : int, optional
        Chunk size along the obs dimension. Default 300.
    n_workers : int, optional
        Number of parallel workers for processing GPI chunks. Default 1.
    gpi_mask : np.ndarray of bool, optional
        Boolean array of shape (n_gpi,). GPIs where gpi_mask is True will be
        skipped entirely — no data is read or written for them. Default None
        (process all GPIs).
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
    print(f"Scan complete in {timer() - scan_start:.1f}s, "
          f"found {sum(len(v) for v in populated_map.values())} populated chunk slots")

    # Estimate max new obs per GPI: each populated slot can contribute at most
    # one observation per GPI, and we merge all spacecraft.
    max_new_obs = max((len(slots) for slots in populated_map.values()), default=0)

    # --- Create or open output store, pre-expanding if needed ---
    if not out_path.exists():
        # Round up to chunk-aligned size
        needed = max_new_obs
        obs_dim_size = _round_up_to_chunk(needed, chunk_size_obs)
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
        )
    else:
        out_root_check = zarr.open(out_path, mode="a")
        current_obs_size = out_root_check["obs"].shape[0]
        existing_max_nobs = int(out_root_check["n_obs"][:].max())
        worst_case = existing_max_nobs + max_new_obs
        if worst_case > current_obs_size:
            _expand_obs_dimension(
                out_root_check, worst_case, chunk_size_obs,
                beam_vars, scalar_vars, has_beams,
            )

    # --- Process GPI chunks in parallel ---
    gpi_chunks = _build_gpi_chunk_ranges(n_gpi, chunk_size_gpi)
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
        chunk_size_obs=chunk_size_obs,
        gpi_mask=gpi_mask,
    )

    start = timer()
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_func, chunk) for chunk in gpi_chunks]
            for i, future in enumerate(futures):
                future.result()
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(gpi_chunks)} GPI chunks")
    else:
        for i, chunk in enumerate(gpi_chunks):
            process_func(chunk)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(gpi_chunks)} GPI chunks")

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

    Probes chunk file existence for the 'time' variable without reading data.

    Returns
    -------
    dict[int, list[tuple[int, int]]]
        gpi_chunk_index -> list of (swath_time_idx, spacecraft_idx) with data.
    """
    n_gpi_chunks = -(-n_gpi // sparse_gpi_chunk_size)  # ceiling division

    populated_map = {}

    for gc in range(n_gpi_chunks):
        slots = []
        for t_idx in range(n_swath_time):
            for s_idx in range(n_spacecraft):
                chunk_key = f"time/c/{t_idx}/{s_idx}/{gc}"
                chunk_path = sparse_path / chunk_key
                if chunk_path.exists():
                    slots.append((t_idx, s_idx))
        if slots:
            populated_map[gc] = slots

    return populated_map


# ---------------------------------------------------------------------------
# Output store creation and resizing
# ---------------------------------------------------------------------------

def _round_up_to_chunk(n, chunk_size):
    """Round n up to the nearest multiple of chunk_size."""
    return -(-n // chunk_size) * chunk_size


def _create_dense_structure(
    out_path,
    sparse_root,
    beam_vars,
    scalar_vars,
    has_beams,
    obs_dim_size,
    chunk_size_gpi,
    chunk_size_obs,
):
    """Create the empty dense Zarr store."""
    n_gpi = sparse_root["gpi"].shape[0]

    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    compressors = [
        zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        )
    ]

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in sorted(beam_vars):
            src = sparse_root[var]
            root.create_array(
                name=var,
                dtype=src.dtype,
                shape=(n_gpi, obs_dim_size, n_beams),
                chunks=(chunk_size_gpi, chunk_size_obs, 1),
                dimension_names=("gpi", "obs", "beam"),
                fill_value=src.metadata.fill_value,
                compressors=compressors,
                attributes=dict(src.attrs),
            )

    for var in sorted(scalar_vars):
        src = sparse_root[var]
        root.create_array(
            name=var,
            dtype=src.dtype,
            shape=(n_gpi, obs_dim_size),
            chunks=(chunk_size_gpi, chunk_size_obs),
            dimension_names=("gpi", "obs"),
            fill_value=src.metadata.fill_value,
            compressors=compressors,
            attributes=dict(src.attrs),
        )

    root.create_array(
        name="n_obs",
        dtype="uint32",
        shape=(n_gpi,),
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=0,
        compressors=compressors,
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


def _expand_obs_dimension(out_root, needed_size, chunk_size_obs, beam_vars, scalar_vars, has_beams):
    """Expand the obs dimension in chunk-aligned increments."""
    current_size = out_root["obs"].shape[0]
    new_size = _round_up_to_chunk(needed_size, chunk_size_obs)

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
    chunk_size_obs,
    gpi_mask=None,
):
    """Process a single chunk of GPIs: extract, sort, and write/append.

    Uses a chunk-batched read-modify-write strategy: for each output
    (gpi_chunk, obs_chunk) that needs updating, read the full chunk from
    the output store, apply all GPI updates in-memory, write it back once.

    Parameters
    ----------
    gpi_range : tuple of (int, int)
        (start, end) GPI indices for this chunk.
    sparse_path : str
        Path to sparse Zarr store.
    out_path : str
        Path to output dense Zarr store.
    beam_vars : set of str
        Beam-dimensioned variable names.
    scalar_vars : set of str
        Scalar variable names.
    has_beams : bool
        Whether beam dimension exists.
    populated_map : dict[int, list[tuple[int, int]]]
        Pre-scanned map of gpi_chunk_index -> populated (t_idx, s_idx) slots.
    sparse_gpi_chunk_size : int
        GPI chunk size in the sparse store.
    chunk_size_obs : int
        Chunk size along the obs dimension in the output store.
    gpi_mask : np.ndarray of bool or None
        If provided, GPIs where mask is True are skipped.
    """
    gpi_start, gpi_end = gpi_range
    n_gpi_chunk = gpi_end - gpi_start

    # Extract local mask for this chunk
    if gpi_mask is not None:
        local_mask = gpi_mask[gpi_start:gpi_end]
        if local_mask.all():
            return
    else:
        local_mask = None

    # Find populated slots for this GPI range
    gc_start = gpi_start // sparse_gpi_chunk_size
    gc_end = (gpi_end - 1) // sparse_gpi_chunk_size

    populated_set = set()
    for gc in range(gc_start, gc_end + 1):
        populated_set.update(populated_map.get(gc, []))

    if not populated_set:
        return

    populated = sorted(populated_set)

    sparse_root = zarr.open(sparse_path, mode="r")
    out_root = zarr.open(out_path, mode="a")

    time_var = "time"
    time_fill = sparse_root[time_var].metadata.fill_value

    # --- Read time values for all populated slots ---
    time_slices = np.empty(
        (len(populated), n_gpi_chunk), dtype=sparse_root[time_var].dtype
    )
    for i, (t_idx, s_idx) in enumerate(populated):
        time_slices[i, :] = sparse_root[time_var][t_idx, s_idx, gpi_start:gpi_end]

    # --- Per-GPI: find valid observations, sort by time ---
    valid_mask = time_slices != time_fill

    gpi_sorted_slots = []
    new_counts = np.zeros(n_gpi_chunk, dtype=np.int32)

    for g in range(n_gpi_chunk):
        # Skip masked GPIs
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
            buf[i, :] = sparse_root[var][t_idx, s_idx, gpi_start:gpi_end]
        data_cache[var] = buf

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in sorted(beam_vars):
            buf = np.empty(
                (len(populated), n_beams, n_gpi_chunk), dtype=sparse_root[var].dtype
            )
            for i, (t_idx, s_idx) in enumerate(populated):
                buf[i, :, :] = sparse_root[var][t_idx, s_idx, :, gpi_start:gpi_end]
            data_cache[var] = buf

    # --- Determine write positions ---
    existing_n_obs = out_root["n_obs"][gpi_start:gpi_end].astype(np.int32)
    write_starts = existing_n_obs.copy()
    write_ends = write_starts + new_counts

    # --- Group writes by output obs chunk, then do read-modify-write ---
    if write_ends.max() == 0:
        return

    min_obs = int(write_starts[write_starts < write_ends].min())
    max_obs = int(write_ends.max())
    first_obs_chunk = min_obs // chunk_size_obs
    last_obs_chunk = (max_obs - 1) // chunk_size_obs

    gpi_slice = slice(gpi_start, gpi_end)

    for obs_chunk_idx in range(first_obs_chunk, last_obs_chunk + 1):
        obs_chunk_start = obs_chunk_idx * chunk_size_obs
        obs_chunk_end = min(obs_chunk_start + chunk_size_obs,
                            out_root["obs"].shape[0])
        obs_slice = slice(obs_chunk_start, obs_chunk_end)

        # Check which GPIs have writes landing in this obs chunk
        has_write = (write_starts < obs_chunk_end) & (write_ends > obs_chunk_start)
        if not np.any(has_write):
            continue

        # --- Read-modify-write for each variable ---

        # Time (scalar)
        chunk_data_time = out_root[time_var][gpi_slice, obs_slice]

        for g in np.where(has_write)[0]:
            sorted_slots = gpi_sorted_slots[g]
            ws = int(write_starts[g])
            we = int(write_ends[g])

            ov_start = max(ws, obs_chunk_start)
            ov_end = min(we, obs_chunk_end)

            data_ov_start = ov_start - ws
            data_ov_end = ov_end - ws
            relevant_slots = sorted_slots[data_ov_start:data_ov_end]

            rel_obs_start = ov_start - obs_chunk_start
            rel_obs_end = ov_end - obs_chunk_start

            chunk_data_time[g, rel_obs_start:rel_obs_end] = (
                time_slices[relevant_slots, g]
            )

        out_root[time_var][gpi_slice, obs_slice] = chunk_data_time

        # Scalar variables
        for var in sorted(scalar_vars):
            if var == time_var:
                continue

            chunk_data = out_root[var][gpi_slice, obs_slice]

            for g in np.where(has_write)[0]:
                sorted_slots = gpi_sorted_slots[g]
                ws = int(write_starts[g])
                we = int(write_ends[g])

                ov_start = max(ws, obs_chunk_start)
                ov_end = min(we, obs_chunk_end)
                data_ov_start = ov_start - ws
                data_ov_end = ov_end - ws
                relevant_slots = sorted_slots[data_ov_start:data_ov_end]

                rel_obs_start = ov_start - obs_chunk_start
                rel_obs_end = ov_end - obs_chunk_start

                chunk_data[g, rel_obs_start:rel_obs_end] = (
                    data_cache[var][relevant_slots, g]
                )

            out_root[var][gpi_slice, obs_slice] = chunk_data

        # Beam variables
        if has_beams:
            for var in sorted(beam_vars):
                chunk_data = out_root[var][gpi_slice, obs_slice, :]

                for g in np.where(has_write)[0]:
                    sorted_slots = gpi_sorted_slots[g]
                    ws = int(write_starts[g])
                    we = int(write_ends[g])

                    ov_start = max(ws, obs_chunk_start)
                    ov_end = min(we, obs_chunk_end)
                    data_ov_start = ov_start - ws
                    data_ov_end = ov_end - ws
                    relevant_slots = sorted_slots[data_ov_start:data_ov_end]

                    rel_obs_start = ov_start - obs_chunk_start
                    rel_obs_end = ov_end - obs_chunk_start

                    chunk_data[g, rel_obs_start:rel_obs_end, :] = (
                        data_cache[var][relevant_slots, :, g]
                    )

                out_root[var][gpi_slice, obs_slice, :] = chunk_data

    # --- Update n_obs ---
    out_root["n_obs"][gpi_start:gpi_end] = write_ends.astype("uint32")
