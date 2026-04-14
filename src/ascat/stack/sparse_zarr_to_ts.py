
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

The two-pass approach reduces total Zarr read operations from hundreds
of millions (original method) to a few million, at the cost of
temporary disk space for the intermediate store.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

from dataclasses import dataclass

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
        return self.shard_size_obs if self.shard_size_obs is not None else self.inner_chunk_obs

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
    """Read the sharding_indexed codec footer and return populated inner chunk indices.

    The shard index is stored at the end of the file as *n_inner_chunks*
    pairs of (offset, nbytes) encoded as little-endian uint64, followed
    by a 4-byte crc32c checksum (the default index_codecs configuration).

    An inner chunk is empty when both offset and nbytes equal 2^64 - 1.
    """
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
# Variable classification (shared with original module)
# ---------------------------------------------------------------------------

def _classify_variables(root, has_beams):
    """Classify data variables into beam and scalar categories."""
    coord_names = {"swath_time", "spacecraft", "beam", "gpi",
                   "longitude", "latitude", "obs", "n_obs",
                   "_slot_index"}
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
    coord_names = {"gpi", "longitude", "latitude", "beam",
                   "_slot_index", "obs", "n_obs"}
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
# Populated-chunk scanning (with shard index support)
# ---------------------------------------------------------------------------

def _scan_all_populated_slots(sparse_path, n_swath_time, n_spacecraft,
                              n_gpi, sparse_gpi_chunk_size):
    """Scan the sparse store and return all populated (swath_time, spacecraft) slots.

    Uses shard index reading when the store is sharded, falling back to
    chunk file existence probing otherwise.

    Returns
    -------
    list of tuple[int, int]
        Sorted list of (swath_time_idx, spacecraft_idx) pairs that have data.
    """
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
                sparse_path / "time" / "c"
                / str(t_idx) / str(s_idx) / str(shard_idx)
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
                        sparse_path / "time" / "c"
                        / str(t_idx) / str(s_idx) / str(gc)
                    )
                    if chunk_path.exists():
                        all_slots.add((t_idx, s_idx))
                        break  # one hit is enough to know this slot exists

    return sorted(all_slots)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_up(n, multiple):
    """Round n up to the nearest multiple."""
    return -(-n // multiple) * multiple


# ---------------------------------------------------------------------------
# Shared-memory read worker for parallel decompression
# ---------------------------------------------------------------------------

def _reset_codec_pipeline_worker():
    """Initializer for shm read worker processes.

    When the parent process has configured zarrs-python as the codec
    pipeline (for fast writes), forked worker processes inherit that
    config.  But inside workers we want plain zarr-python — each worker
    is already a full process with its own GIL providing parallelism,
    and having each one also try to spin up Rust thread pools would
    cause oversubscription.
    """
    try:
        import zarr
        zarr.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"})
    except Exception:
        pass


def _shm_read_worker(args):
    """Read one (t_idx, s_idx) slab for ALL variables into shared memory.

    Runs in a worker process — opens its own zarr store handle, reads
    every variable for one time slot, writes results directly into
    pre-allocated shared memory buffers.  No data crosses the process
    boundary via pickling.
    """
    (sparse_path, t_idx, s_idx, row_idx, batch_actual, n_gpi,
     scalar_shm_info, beam_shm_info) = args

    root = zarr.open(sparse_path, mode="r")

    # scalar_shm_info: list of (var, shm_name, dtype_str)
    for var, shm_name, dtype_str in scalar_shm_info:
        shm = SharedMemory(name=shm_name)
        dtype = np.dtype(dtype_str)
        try:
            buf = np.ndarray((batch_actual, n_gpi), dtype=dtype, buffer=shm.buf)
            buf[row_idx, :] = root[var][t_idx, s_idx, :]
        finally:
            shm.close()

    # beam_shm_info: list of (var, shm_name, dtype_str, n_beams)
    for var, shm_name, dtype_str, n_beams in beam_shm_info:
        shm = SharedMemory(name=shm_name)
        dtype = np.dtype(dtype_str)
        try:
            buf = np.ndarray(
                (batch_actual, n_beams, n_gpi), dtype=dtype, buffer=shm.buf)
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
):
    """Rechunk a sparse swath-time Zarr store into an intermediate obs-major store.

    Reads the sparse store time-slot by time-slot in batches, writing to
    an intermediate store with dimensions (obs, gpi) for scalar variables
    and (obs, beam, gpi) for beam variables.  The obs axis is a compacted
    enumeration of all populated (swath_time, spacecraft) slots.

    This function uses a hybrid strategy:

    - **Reads**: ProcessPoolExecutor with shared memory buffers.  Each
      worker process truly parallelizes decompression (no GIL contention),
      reading one slab per variable directly into the shared buffer.
    - **Writes**: ThreadPoolExecutor in the main process.  For best
      performance, configure zarrs-python as the codec pipeline before
      calling this function, so writes go through Rust:

          import zarr
          zarr.config.set({
              "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
              "threading.max_workers": 64,
          })

      The worker processes automatically reset to the default zarr-python
      codec pipeline via an initializer, so the subprocess reads don't
      try to spawn Rust thread pools (which would oversubscribe the CPU).

    Parameters
    ----------
    sparse_path : str or Path
        Path to the sparse Zarr store (swath_time, spacecraft, [beam,] gpi).
    intermediate_path : str or Path
        Path for the intermediate rechunked Zarr store.
    target_gpi_chunk : int
        Spatial chunk size in the intermediate store.  Default 64.
    batch_size : int
        Number of time slots to read into memory at once.  Controls memory
        usage and chunk alignment along the obs axis.  Default 500.
    n_read_threads : int
        Number of worker processes for reading and also the number of
        threads used for the parallel write phase.  Default 16.
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

    # --- Scan for populated slots ---
    print("Scanning for populated slots...")
    scan_start = timer()
    all_slots = _scan_all_populated_slots(
        sparse_path, n_swath_time, n_spacecraft, n_gpi, sparse_gpi_chunk_size,
    )
    n_obs = len(all_slots)
    print(f"Scan complete in {timer() - scan_start:.1f}s, "
          f"found {n_obs} populated slots")

    padded_n_obs = _round_up(n_obs, batch_size)

    # --- Create intermediate store ---
    print(f"Creating intermediate store at {intermediate_path}")
    print(f"  obs={padded_n_obs}, gpi={n_gpi}, "
          f"chunks=({batch_size}, {target_gpi_chunk})")

    store = zarr.storage.LocalStore(str(intermediate_path))
    int_root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    # Copy group-level attributes
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
                compressors=compressor,
            )

    # Slot index: maps obs position -> (swath_time_idx, spacecraft_idx)
    slot_arr = np.array(all_slots, dtype="int32")
    int_root.create_array("_slot_index", data=slot_arr)

    # Copy spatial coordinates with attributes
    for coord in ["gpi", "longitude", "latitude"]:
        if coord in sparse_root:
            arr = int_root.create_array(
                coord, data=sparse_root[coord][:],
                attributes=dict(sparse_root[coord].attrs),
            )
    if has_beams and "beam" in sparse_root:
        int_root.create_array(
            "beam", data=sparse_root["beam"][:],
            attributes=dict(sparse_root["beam"].attrs),
        )

    # --- Process in batches ---
    n_batches = -(-n_obs // batch_size)
    print(f"Processing {n_obs} slots in {n_batches} batches of {batch_size}")
    total_start = timer()

    # Collect dtype and fill_value info (avoid pickling zarr objects)
    var_info_scalar = [
        (var, str(sparse_root[var].dtype), sparse_root[var].metadata.fill_value)
        for var in all_vars_scalar
    ]
    var_info_beam = [
        (var, str(sparse_root[var].dtype), sparse_root[var].metadata.fill_value,
         sparse_root["beam"].shape[0])
        for var in all_vars_beam
    ] if has_beams else []

    # --- Allocate shared memory buffers ONCE, reuse across all batches ---
    # Buffers are sized for the full batch_size.  The last batch may be
    # smaller (n_obs not divisible by batch_size) — in that case we just
    # write a partial slice.
    print(f"Allocating shared memory buffers for batch_size={batch_size}")
    scalar_shms = []  # (var, shm, buf, dtype_str)
    for var, dtype_str, fill_val in var_info_scalar:
        dtype = np.dtype(dtype_str)
        buf_size = batch_size * n_gpi * dtype.itemsize
        shm = SharedMemory(create=True, size=buf_size)
        buf = np.ndarray((batch_size, n_gpi), dtype=dtype, buffer=shm.buf)
        scalar_shms.append((var, shm, buf, dtype_str, fill_val))

    beam_shms = []  # (var, shm, buf, dtype_str, n_beams, fill_val)
    for var, dtype_str, fill_val, n_beams in var_info_beam:
        dtype = np.dtype(dtype_str)
        buf_size = batch_size * n_beams * n_gpi * dtype.itemsize
        shm = SharedMemory(create=True, size=buf_size)
        buf = np.ndarray(
            (batch_size, n_beams, n_gpi), dtype=dtype, buffer=shm.buf)
        beam_shms.append((var, shm, buf, dtype_str, n_beams, fill_val))

    # Pre-build the shm_info lists (constant across batches)
    scalar_shm_info = [
        (var, shm.name, dtype_str)
        for var, shm, buf, dtype_str, fill_val in scalar_shms
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

                # --- Reset buffers to fill values ---
                # Only needs to touch the rows we'll actually write to.
                # The last batch may be smaller than batch_size — any rows
                # beyond `actual` are left from the previous batch but we
                # only write [:actual] to the intermediate.
                for var, shm, buf, dtype_str, fill_val in scalar_shms:
                    buf[:actual] = fill_val
                for var, shm, buf, dtype_str, n_beams, fill_val in beam_shms:
                    buf[:actual] = fill_val

                fill_elapsed = timer() - batch_timer

                # --- Build task list: one task per slot, reads ALL variables ---
                tasks = [
                    (str(sparse_path), t_idx, s_idx, i, batch_size, n_gpi,
                     scalar_shm_info, beam_shm_info)
                    for i, (t_idx, s_idx) in enumerate(batch)
                ]

                read_timer = timer()
                list(pool.map(_shm_read_worker, tasks))
                read_elapsed = timer() - read_timer

                # --- Write all buffers to intermediate store in parallel ---
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

                print(f"  Batch {batch_idx + 1}/{n_batches}: {actual} slots, "
                      f"fill {fill_elapsed:.1f}s, "
                      f"read {read_elapsed:.1f}s, write {write_elapsed:.1f}s")

        total_elapsed = timer() - total_start
        print(f"Rechunking complete in {total_elapsed:.1f}s")
    finally:
        # Clean up shared memory
        for var, shm, buf, dtype_str, fill_val in scalar_shms:
            shm.close()
            shm.unlink()
        for var, shm, buf, dtype_str, n_beams, fill_val in beam_shms:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------
# Pass 1 (alternative): Rechunk using zarrs-python codec pipeline
# ---------------------------------------------------------------------------

def rechunk_sparse_zarrs(
    sparse_path,
    intermediate_path,
    target_gpi_chunk=64,
    batch_size=500,
    n_threads=64,
):
    """Rechunk using the zarrs-python (Rust) codec pipeline.

    Same logic as ``rechunk_sparse`` but relies on zarrs-python for
    parallel chunk decompression.  Because zarrs-python releases the
    GIL during reads/writes, plain ``ThreadPoolExecutor`` parallelism
    works — no need for shared memory or process pools.

    You must install zarrs-python and configure it before calling:

        import zarr
        zarr.config.set({
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": 64,
        })

    Parameters
    ----------
    sparse_path : str or Path
    intermediate_path : str or Path
    target_gpi_chunk : int
        Spatial chunk size in the intermediate store.  Default 64.
    batch_size : int
        Number of time slots per memory batch.  Default 500.
    n_threads : int
        Number of Python-side threads used to dispatch reads and writes.
        zarrs-python will manage its own internal thread pool below this.
        Default 64.
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

    # --- Scan for populated slots ---
    print("Scanning for populated slots...")
    scan_start = timer()
    all_slots = _scan_all_populated_slots(
        sparse_path, n_swath_time, n_spacecraft, n_gpi, sparse_gpi_chunk_size,
    )
    n_obs = len(all_slots)
    print(f"Scan complete in {timer() - scan_start:.1f}s, "
          f"found {n_obs} populated slots")

    padded_n_obs = _round_up(n_obs, batch_size)

    # --- Create intermediate store ---
    print(f"Creating intermediate store at {intermediate_path}")
    print(f"  obs={padded_n_obs}, gpi={n_gpi}, "
          f"chunks=({batch_size}, {target_gpi_chunk})")

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
                compressors=compressor,
            )

    slot_arr = np.array(all_slots, dtype="int32")
    int_root.create_array("_slot_index", data=slot_arr)

    for coord in ["gpi", "longitude", "latitude"]:
        if coord in sparse_root:
            int_root.create_array(
                coord, data=sparse_root[coord][:],
                attributes=dict(sparse_root[coord].attrs),
            )
    if has_beams and "beam" in sparse_root:
        int_root.create_array(
            "beam", data=sparse_root["beam"][:],
            attributes=dict(sparse_root["beam"].attrs),
        )

    # --- Allocate plain numpy buffers ONCE, reuse across all batches ---
    # No shared memory needed — zarrs-python releases the GIL.
    print(f"Allocating numpy buffers for batch_size={batch_size}")
    scalar_bufs = {}  # var -> (buf, fill_val)
    for var in all_vars_scalar:
        dtype = sparse_root[var].dtype
        fill_val = sparse_root[var].metadata.fill_value
        buf = np.empty((batch_size, n_gpi), dtype=dtype)
        scalar_bufs[var] = (buf, fill_val)

    beam_bufs = {}  # var -> (buf, fill_val, n_beams)
    if has_beams:
        for var in all_vars_beam:
            dtype = sparse_root[var].dtype
            fill_val = sparse_root[var].metadata.fill_value
            buf = np.empty((batch_size, n_beams, n_gpi), dtype=dtype)
            beam_bufs[var] = (buf, fill_val, n_beams)

    # Prefault the buffers so the first batch doesn't pay the cost of
    # OS page-faulting hundreds of GB of fresh anonymous memory.
    # np.empty() only reserves virtual memory; first write triggers faults.
    print("Prefaulting buffers...")
    prefault_start = timer()
    for var, (buf, fill_val) in scalar_bufs.items():
        buf[:] = fill_val
    for var, (buf, fill_val, n_beams) in beam_bufs.items():
        buf[:] = fill_val
    print(f"Prefault done in {timer() - prefault_start:.1f}s")

    # --- Process in batches ---
    n_batches = -(-n_obs // batch_size)
    print(f"Processing {n_obs} slots in {n_batches} batches of {batch_size}")
    total_start = timer()

    def _read_slot(args):
        """Read all variables for one (t_idx, s_idx) slot into row i."""
        i, t_idx, s_idx = args
        for var in all_vars_scalar:
            buf, _ = scalar_bufs[var]
            buf[i, :] = sparse_root[var][t_idx, s_idx, :]
        for var in all_vars_beam:
            buf, _, _ = beam_bufs[var]
            buf[i, :, :] = sparse_root[var][t_idx, s_idx, :, :]

    def _write_scalar(item):
        var, (buf, fill_val), obs_start, obs_end, actual = item
        int_root[var][obs_start:obs_end, :] = buf[:actual]

    def _write_beam(item):
        var, (buf, fill_val, n_beams), obs_start, obs_end, actual = item
        int_root[var][obs_start:obs_end, :, :] = buf[:actual]

    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, n_obs)
        batch = all_slots[batch_start_idx:batch_end_idx]
        actual = len(batch)

        batch_timer = timer()

        # Reset only the rows we'll actually use to fill values
        for var, (buf, fill_val) in scalar_bufs.items():
            buf[:actual] = fill_val
        for var, (buf, fill_val, n_beams) in beam_bufs.items():
            buf[:actual] = fill_val

        fill_elapsed = timer() - batch_timer

        # --- Reads: one task per slot, each reads all variables ---
        read_timer = timer()
        read_tasks = [(i, t_idx, s_idx) for i, (t_idx, s_idx) in enumerate(batch)]

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(_read_slot, read_tasks))

        read_elapsed = timer() - read_timer

        # --- Writes: one task per variable, in threads ---
        write_timer = timer()
        obs_start = batch_start_idx
        obs_end = batch_start_idx + actual

        scalar_write_items = [
            (var, scalar_bufs[var], obs_start, obs_end, actual)
            for var in all_vars_scalar
        ]
        beam_write_items = [
            (var, beam_bufs[var], obs_start, obs_end, actual)
            for var in all_vars_beam
        ]

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(_write_scalar, scalar_write_items))
            if beam_write_items:
                list(pool.map(_write_beam, beam_write_items))

        write_elapsed = timer() - write_timer

        print(f"  Batch {batch_idx + 1}/{n_batches}: {actual} slots, "
              f"fill {fill_elapsed:.1f}s, "
              f"read {read_elapsed:.1f}s, write {write_elapsed:.1f}s")

    total_elapsed = timer() - total_start
    print(f"Rechunking complete in {total_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Pass 2: Dense conversion from intermediate
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

    # Copy group-level attributes from intermediate
    root.attrs.update(dict(int_root.attrs))

    if has_beams:
        n_beams = int_root["beam"].shape[0]
        for var in sorted(beam_vars):
            src = int_root[var]
            arr_kwargs = _make_array_kwargs(
                (chunk_size_gpi, chunk_size_obs, 1), sharding)
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
        arr_kwargs = _make_array_kwargs(
            (chunk_size_gpi, chunk_size_obs), sharding)
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
        compressors=[
            BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)
        ],
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


def _process_spatial_chunk_from_intermediate(
    gpi_range,
    intermediate_path,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
):
    """Process one spatial chunk: read from intermediate, sort, write to dense.

    Each spatial chunk in the intermediate contains the full observation
    history for a small group of GPIs.  We read it all at once (one
    contiguous read per variable), drop fill values, sort by time per
    GPI, and write to the dense output.
    """
    gpi_start, gpi_end = gpi_range
    chunk_width = gpi_end - gpi_start

    int_root = zarr.open(intermediate_path, mode="r")
    out_root = zarr.open(out_path, mode="a")

    # --- Read all data for this spatial chunk ---
    # One read per variable: (n_obs_total, chunk_width)
    time_data = int_root["time"][:, gpi_start:gpi_end]
    time_fill = int_root["time"].metadata.fill_value

    scalar_data = {}
    for var in sorted(scalar_vars):
        if var == "time":
            continue
        scalar_data[var] = int_root[var][:, gpi_start:gpi_end]

    beam_data = {}
    if has_beams:
        n_beams = int_root["beam"].shape[0]
        for var in sorted(beam_vars):
            beam_data[var] = int_root[var][:, :, gpi_start:gpi_end]

    # --- Per-GPI: find valid observations, sort by time ---
    valid_mask = time_data != time_fill
    n_valid_per_gpi = valid_mask.sum(axis=0)
    max_valid = int(n_valid_per_gpi.max())

    if max_valid == 0:
        return

    # Build sorted output arrays
    out_time = np.full((chunk_width, max_valid), time_fill,
                       dtype=time_data.dtype)
    out_scalars = {}
    for var in sorted(scalar_vars):
        if var == "time":
            continue
        fill = int_root[var].metadata.fill_value
        out_scalars[var] = np.full((chunk_width, max_valid), fill,
                                   dtype=scalar_data[var].dtype)

    out_beams = {}
    if has_beams:
        for var in sorted(beam_vars):
            fill = int_root[var].metadata.fill_value
            out_beams[var] = np.full((chunk_width, max_valid, n_beams), fill,
                                     dtype=beam_data[var].dtype)

    n_obs_arr = np.zeros(chunk_width, dtype="uint32")

    for g in range(chunk_width):
        valid = valid_mask[:, g]
        n_v = int(n_valid_per_gpi[g])
        if n_v == 0:
            continue

        times_g = time_data[valid, g]
        order = np.argsort(times_g)
        n_obs_arr[g] = n_v

        out_time[g, :n_v] = times_g[order]

        for var in sorted(scalar_vars):
            if var == "time":
                continue
            out_scalars[var][g, :n_v] = scalar_data[var][valid, g][order]

        if has_beams:
            for var in sorted(beam_vars):
                # beam_data[var] is (n_obs_total, n_beams, chunk_width)
                out_beams[var][g, :n_v, :] = beam_data[var][valid, :, g][order, :]

    # --- Write to dense output ---
    obs_end = max_valid
    out_root["time"][gpi_start:gpi_end, :obs_end] = out_time
    for var in sorted(scalar_vars):
        if var == "time":
            continue
        out_root[var][gpi_start:gpi_end, :obs_end] = out_scalars[var]

    if has_beams:
        for var in sorted(beam_vars):
            out_root[var][gpi_start:gpi_end, :obs_end, :] = out_beams[var]

    out_root["n_obs"][gpi_start:gpi_end] = n_obs_arr


def densify_from_intermediate(
    intermediate_path,
    out_path,
    chunk_size_gpi=64,
    chunk_size_obs=300,
    n_workers=16,
    shard_size_gpi=None,
    shard_size_obs=None,
):
    """Convert the intermediate rechunked store to a dense time-series store.

    Reads the intermediate store spatial-chunk by spatial-chunk.  Each
    chunk delivers the full observation history for a group of GPIs,
    making the sort-and-compact step trivial.

    Parameters
    ----------
    intermediate_path : str or Path
        Path to the intermediate Zarr store from ``rechunk_sparse``.
    out_path : str or Path
        Path for the output dense Zarr store (gpi, obs, [beam]).
    chunk_size_gpi : int
        Chunk size along the gpi dimension in the output.  Default 64.
    chunk_size_obs : int
        Chunk size along the obs dimension in the output.  Default 300.
    n_workers : int
        Number of parallel workers.  Default 16.
    shard_size_gpi : int, optional
        Number of GPIs per shard in the output.  Must be a multiple of
        ``chunk_size_gpi``.  When provided, output arrays use the
        sharding_indexed codec along the gpi dimension and parallelism
        is aligned to shard boundaries.  Default None (no gpi sharding).
    shard_size_obs : int, optional
        Number of obs per shard in the output.  Must be a multiple of
        ``chunk_size_obs``.  Default None (no obs sharding).
    """
    intermediate_path = Path(intermediate_path)
    out_path = Path(out_path)

    int_root = zarr.open(intermediate_path, mode="r")
    has_beams = "beam" in int_root
    n_gpi = int_root["gpi"].shape[0]

    beam_vars, scalar_vars = _classify_intermediate_variables(int_root, has_beams)
    print(f"Scalar variables: {sorted(scalar_vars)}")
    print(f"Beam variables: {sorted(beam_vars)}")

    # Build sharding config if requested
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

    # Determine obs dimension size from the data.
    n_obs_total = int_root["time"].shape[0]
    obs_dim_size = _round_up(n_obs_total, obs_alignment)

    print(f"Creating dense store at {out_path}")
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

    # Build spatial chunk ranges.
    # With sharding, align workers to shard boundaries so no two workers
    # write to the same shard file.  The intermediate's gpi chunk size
    # should divide evenly into the worker size.
    int_gpi_chunk = int_root["time"].chunks[-1]
    gpi_chunks = [
        (i, min(i + worker_gpi_size, n_gpi))
        for i in range(0, n_gpi, worker_gpi_size)
    ]
    print(f"Processing {len(gpi_chunks)} GPI chunks with {n_workers} workers"
          f" (worker size: {worker_gpi_size})")

    process_func = partial(
        _process_spatial_chunk_from_intermediate,
        intermediate_path=str(intermediate_path),
        out_path=str(out_path),
        beam_vars=beam_vars,
        scalar_vars=scalar_vars,
        has_beams=has_beams,
    )

    start = timer()
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_func, chunk)
                       for chunk in gpi_chunks]
            for f in tqdm(futures, total=len(gpi_chunks),
                          desc="Spatial chunks"):
                f.result()
    else:
        for chunk in tqdm(gpi_chunks, desc="Spatial chunks"):
            process_func(chunk)

    elapsed = timer() - start
    print(f"Dense conversion complete in {elapsed:.1f}s")


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
):
    """Convert sparse swath-time Zarr to dense time-series via rechunking.

    This is the combined entry point that runs both passes.

    Parameters
    ----------
    sparse_path : str or Path
        Path to the sparse Zarr store.
    out_path : str or Path
        Path for the output dense Zarr store.
    intermediate_path : str or Path, optional
        Path for the intermediate rechunked store.  Defaults to
        ``out_path`` with ``_intermediate`` suffix.
    target_gpi_chunk : int
        Spatial chunk size in the intermediate store.  Default 64.
    batch_size : int
        Number of time slots per memory batch in pass 1.  Default 500.
    n_read_threads : int
        Threads for sparse reads in pass 1.  Default 16.
    chunk_size_gpi : int
        GPI chunk size in the dense output.  Default 64.
    chunk_size_obs : int
        Obs chunk size in the dense output.  Default 300.
    n_workers : int
        Parallel workers for pass 2.  Default 16.
    shard_size_gpi : int, optional
        Number of GPIs per shard in the dense output.  Must be a multiple
        of ``chunk_size_gpi``.  Default None (no gpi sharding).
    shard_size_obs : int, optional
        Number of obs per shard in the dense output.  Must be a multiple
        of ``chunk_size_obs``.  Default None (no obs sharding).
    keep_intermediate : bool
        If False, delete the intermediate store after conversion.
        Default False.
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
    )

    if not keep_intermediate:
        import shutil
        print(f"Cleaning up intermediate store: {intermediate_path}")
        shutil.rmtree(intermediate_path, ignore_errors=True)

    print("Done!")
