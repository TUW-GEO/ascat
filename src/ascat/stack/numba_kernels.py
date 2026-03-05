"""
Numba-accelerated kernels for sparse-to-dense conversion.

Replaces the per-GPI Python loops in _process_inner_gpi_chunk with
vectorized operations over a CSR (compressed sparse row) representation
of the sorted slot indices.

The CSR layout stores all per-GPI sorted slot arrays concatenated into
one flat array, with an offsets array indicating where each GPI's data
starts. This avoids Python-level ragged list iteration and lets numba
operate on contiguous memory.

CSR layout
----------
    offsets : int32[n_gpi + 1]
        offsets[g] .. offsets[g+1] index into flat_indices for GPI g
    flat_indices : int32[total_valid_obs]
        concatenated sorted slot indices for all GPIs
"""

import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
# Phase 1: Sort — replace the per-GPI valid-mask + argsort loop
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _sort_slots_kernel(time_slices, time_fill, mask):
    """Build CSR sorted-slot structure from sparse time data.

    Parameters
    ----------
    time_slices : float64[n_slots, n_gpi]
        Time values for each (populated_slot, gpi).
    time_fill : float64
        Fill value indicating missing data.
    mask : bool[n_gpi]
        True = skip this GPI (masked out). If empty (length 0), no masking.

    Returns
    -------
    offsets : int32[n_gpi + 1]
        CSR row pointers.
    flat_indices : int32[total_valid]
        Concatenated sorted slot indices.
    new_counts : int32[n_gpi]
        Number of valid observations per GPI.
    """
    n_slots, n_gpi = time_slices.shape
    has_mask = mask.shape[0] > 0

    # First pass: count valid slots per GPI to size the flat array
    new_counts = np.zeros(n_gpi, dtype=np.int32)
    for g in range(n_gpi):
        if has_mask and mask[g]:
            continue
        count = 0
        for s in range(n_slots):
            if time_slices[s, g] != time_fill:
                count += 1
        new_counts[g] = count

    # Build offsets
    offsets = np.empty(n_gpi + 1, dtype=np.int32)
    offsets[0] = 0
    for g in range(n_gpi):
        offsets[g + 1] = offsets[g] + new_counts[g]

    total = offsets[n_gpi]
    flat_indices = np.empty(total, dtype=np.int32)

    if total == 0:
        return offsets, flat_indices, new_counts

    # Second pass: collect valid slot indices per GPI
    # Use a temp array for the per-GPI times to sort
    max_count = 0
    for g in range(n_gpi):
        if new_counts[g] > max_count:
            max_count = new_counts[g]

    tmp_slots = np.empty(max_count, dtype=np.int32)
    tmp_times = np.empty(max_count, dtype=time_slices.dtype)

    for g in range(n_gpi):
        nc = new_counts[g]
        if nc == 0:
            continue

        # Gather valid slots
        k = 0
        for s in range(n_slots):
            if time_slices[s, g] != time_fill:
                tmp_slots[k] = s
                tmp_times[k] = time_slices[s, g]
                k += 1

        # Insertion sort (n_slots is typically small, <200)
        for i in range(1, nc):
            t_val = tmp_times[i]
            s_val = tmp_slots[i]
            j = i - 1
            while j >= 0 and tmp_times[j] > t_val:
                tmp_times[j + 1] = tmp_times[j]
                tmp_slots[j + 1] = tmp_slots[j]
                j -= 1
            tmp_times[j + 1] = t_val
            tmp_slots[j + 1] = s_val

        # Write to flat array
        base = offsets[g]
        for k in range(nc):
            flat_indices[base + k] = tmp_slots[k]

    return offsets, flat_indices, new_counts


# ---------------------------------------------------------------------------
# Phase 2: Scatter — replace the per-GPI write loops
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _scatter_scalar_kernel(
    chunk_data,
    source_data,
    offsets,
    flat_indices,
    write_starts,
    obs_chunk_start,
    obs_chunk_end,
):
    """Scatter sparse source data into a dense output chunk for scalar variables.

    Parameters
    ----------
    chunk_data : array[n_gpi, obs_window]
        Output buffer (read from dense store, modified in-place).
    source_data : array[n_slots, n_gpi]
        Source data indexed by (slot, gpi).
    offsets : int32[n_gpi + 1]
        CSR row pointers into flat_indices.
    flat_indices : int32[total]
        Sorted slot indices for all GPIs.
    write_starts : int32[n_gpi]
        Absolute obs index where writing begins for each GPI.
    obs_chunk_start : int
        Absolute obs index of the start of this chunk window.
    obs_chunk_end : int
        Absolute obs index of the end of this chunk window.
    """
    n_gpi = chunk_data.shape[0]

    for g in range(n_gpi):
        nc = offsets[g + 1] - offsets[g]
        if nc == 0:
            continue

        ws = write_starts[g]
        we = ws + nc

        # Overlap of [ws, we) with [obs_chunk_start, obs_chunk_end)
        ov_start = max(ws, obs_chunk_start)
        ov_end = min(we, obs_chunk_end)
        if ov_start >= ov_end:
            continue

        base = offsets[g]
        slot_offset = ov_start - ws
        obs_offset = ov_start - obs_chunk_start

        for k in range(ov_end - ov_start):
            slot_idx = flat_indices[base + slot_offset + k]
            chunk_data[g, obs_offset + k] = source_data[slot_idx, g]


@nb.njit(cache=True)
def _scatter_beam_kernel(
    chunk_data,
    source_data,
    offsets,
    flat_indices,
    write_starts,
    obs_chunk_start,
    obs_chunk_end,
):
    """Scatter sparse source data into a dense output chunk for beam variables.

    Parameters
    ----------
    chunk_data : array[n_gpi, obs_window, n_beams]
        Output buffer (modified in-place).
    source_data : array[n_slots, n_beams, n_gpi]
        Source data indexed by (slot, beam, gpi).
    offsets : int32[n_gpi + 1]
        CSR row pointers.
    flat_indices : int32[total]
        Sorted slot indices.
    write_starts : int32[n_gpi]
        Absolute obs write start per GPI.
    obs_chunk_start, obs_chunk_end : int
        Chunk window bounds.
    """
    n_gpi = chunk_data.shape[0]
    n_beams = chunk_data.shape[2]

    for g in range(n_gpi):
        nc = offsets[g + 1] - offsets[g]
        if nc == 0:
            continue

        ws = write_starts[g]
        we = ws + nc

        ov_start = max(ws, obs_chunk_start)
        ov_end = min(we, obs_chunk_end)
        if ov_start >= ov_end:
            continue

        base = offsets[g]
        slot_offset = ov_start - ws
        obs_offset = ov_start - obs_chunk_start

        for k in range(ov_end - ov_start):
            slot_idx = flat_indices[base + slot_offset + k]
            for b in range(n_beams):
                chunk_data[g, obs_offset + k, b] = source_data[slot_idx, b, g]
