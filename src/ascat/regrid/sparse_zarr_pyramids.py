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
Regrid sparse swath-time Zarr cubes from Fibonacci grid (GPI) to regular
lat/lon grids, with optional multiscale pyramid generation.

Each lat/lon cell is assigned its nearest GPI via KDTree lookup, with a
distance bound to avoid assigning distant GPIs to cells far from any data.

The output follows the Zarr multiscales conventions, with each pyramid level
stored as a numbered group within a single Zarr store.

Supports incremental append: if the output store already exists, only
swath_time slots that have not yet been regridded are processed.  Delete
the output store entirely to force a full rerun.
"""

import itertools
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import time as timer

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def _zarr_path_opener(sparse_path):
    """Module-level opener used as the default when callers pass a path.

    Module-level (not a lambda or closure) so ``functools.partial`` over it
    pickles cleanly across ``ProcessPoolExecutor`` worker boundaries.
    """
    return zarr.open(str(sparse_path), mode="r")


def regrid_to_latlon(
    sparse_path=None,
    out_path=None,
    grid=None,
    *,
    sparse_opener=None,
    resolution_deg=0.25,
    max_dist_m=None,
    n_pyramid_levels=4,
    lat_chunk=256,
    lon_chunk=256,
    n_workers=1,
):
    """Regrid a sparse swath-time Zarr cube to a regular lat/lon grid with pyramids.

    If the output store already exists, only swath_time slots that have not yet
    been regridded are processed (incremental append).  To force a full rerun,
    delete the output store first.

    Parameters
    ----------
    sparse_path : str or Path, optional
        Path to the sparse Zarr store with dims (swath_time, spacecraft, [beam,] gpi).
        Mutually exclusive with ``sparse_opener``; exactly one must be given.
    out_path : str or Path
        Path for the output Zarr store with multiscale pyramid groups. Required
        (the leading optional default is a backwards-compat artifact for callers
        that pass ``sparse_path`` positionally and ``out_path`` next).
    grid : FibGrid or similar
        Grid object with a KDTree (``grid.kdTree``) and ``get_grid_points()``
        returning (gpis, lons, lats, cells). Required.
    sparse_opener : callable, optional
        Zero-arg callable returning an opened ``zarr.Group`` over the sparse
        store. Must be picklable (use a module-level function wrapped in
        ``functools.partial`` — not a lambda or closure) so worker processes
        can re-open the source. Mutually exclusive with ``sparse_path``.
    resolution_deg : float, optional
        Base resolution in degrees for the lat/lon grid. Default 0.25.
    max_dist_m : float or None, optional
        Maximum distance in meters for NN lookup. Cells beyond this distance
        from any GPI get fill values. Default is 2x the grid spacing.
    n_pyramid_levels : int, optional
        Number of pyramid levels (including base). Default 4.
    lat_chunk : int, optional
        Chunk size along latitude. Default 256.
    lon_chunk : int, optional
        Chunk size along longitude. Default 256.
    n_workers : int, optional
        Number of parallel workers. Default 1.
    """
    if out_path is None:
        raise TypeError("regrid_to_latlon: 'out_path' is required")
    if grid is None:
        raise TypeError("regrid_to_latlon: 'grid' is required")
    if (sparse_path is None) == (sparse_opener is None):
        raise ValueError(
            "regrid_to_latlon: pass exactly one of sparse_path / sparse_opener"
        )
    if sparse_opener is None:
        sparse_opener = partial(_zarr_path_opener, str(sparse_path))
    out_path = Path(out_path)

    sparse_root = sparse_opener()
    has_beams = "beam" in sparse_root
    n_swath_time = sparse_root["swath_time"].shape[0]
    n_spacecraft = sparse_root["spacecraft"].shape[0]

    beam_vars, scalar_vars = _classify_variables(sparse_root, has_beams)
    print(f"Beam variables: {sorted(beam_vars)}")
    print(f"Scalar variables: {sorted(scalar_vars)}")

    # --- Build base lat/lon grid and NN lookup ---
    lats_1d, lons_1d = _build_regular_grid(resolution_deg)
    n_lat = len(lats_1d)
    n_lon = len(lons_1d)
    print(f"Base grid: {n_lat} x {n_lon} ({resolution_deg}deg)")

    print("Computing lat/lon -> GPI nearest-neighbor lookup...")
    lookup_start = timer()
    nn_gpi_indices, nn_valid_mask = _compute_nn_lookup(
        grid, lats_1d, lons_1d, resolution_deg, max_dist_m
    )
    n_valid = nn_valid_mask.sum()
    print(f"Lookup computed in {timer() - lookup_start:.1f}s, "
          f"{n_valid}/{n_lat * n_lon} cells have a nearby GPI")

    # --- Create output store if needed, otherwise open existing ---
    store_exists = (out_path / "zarr.json").exists()
    if not store_exists:
        print("Creating output Zarr structure...")
        _create_pyramid_store(
            out_path=out_path,
            sparse_root=sparse_root,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            lats_1d=lats_1d,
            lons_1d=lons_1d,
            resolution_deg=resolution_deg,
            n_pyramid_levels=n_pyramid_levels,
            lat_chunk=lat_chunk,
            lon_chunk=lon_chunk,
        )
    else:
        # Expand swath_time in all pyramid levels if the sparse store has grown
        _maybe_expand_pyramid_swath_time(out_path, sparse_root)

    # --- Determine which (swath_time, spacecraft) slices still need regridding ---
    all_slices = [
        (t, s) for t in range(n_swath_time) for s in range(n_spacecraft)
    ]

    if store_exists:
        pending_slices = _find_pending_slices(
            out_path=out_path,
            sparse_source=sparse_opener,
            all_slices=all_slices,
        )
        n_skip = len(all_slices) - len(pending_slices)
        print(f"Skipping {n_skip} already-regridded slices, "
              f"processing {len(pending_slices)} pending")
    else:
        pending_slices = all_slices

    # --- Regrid level 0 ---
    if pending_slices:
        print(f"Regridding {len(pending_slices)} slices to level 0 "
              f"with {n_workers} workers")

        regrid_func = partial(
            _regrid_slice,
            sparse_opener=sparse_opener,
            out_path=str(out_path),
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            nn_gpi_indices=nn_gpi_indices,
            nn_valid_mask=nn_valid_mask,
            n_lat=n_lat,
            n_lon=n_lon,
        )

        start = timer()
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(regrid_func, s) for s in pending_slices]
                for future in tqdm(futures, total=len(futures), desc="Regridding level 0"):
                    future.result()
        else:
            for sl in tqdm(pending_slices, total=len(pending_slices), desc="Regridding level 0"):
                regrid_func(sl)

        print(f"Level 0 done in {timer() - start:.1f}s")
    else:
        print("All level-0 slices already regridded, skipping")

    # --- Build pyramid levels 1..N ---
    if n_pyramid_levels > 1:
        print(f"Building {n_pyramid_levels - 1} pyramid levels...")
        pyr_start = timer()
        _build_pyramid_levels(
            out_path=str(out_path),
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            n_pyramid_levels=n_pyramid_levels,
            n_swath_time=n_swath_time,
            n_spacecraft=n_spacecraft,
            n_workers=n_workers,
        )
        print(f"Pyramids done in {timer() - pyr_start:.1f}s")

    print("All done!")


# ---------------------------------------------------------------------------
# Pending-slice detection
# ---------------------------------------------------------------------------

def _find_pending_slices(out_path, sparse_source, all_slices):
    """Return slices that have data in the sparse store but are not yet marked
    done in the pyramid's ``processed`` array.

    Parameters
    ----------
    out_path : Path or str
        Path to the output pyramid Zarr store.
    sparse_source : Path, str, or callable
        Either a path to the sparse Zarr store, or a zero-arg callable returning
        an opened ``zarr.Group`` over it.
    all_slices : list of (int, int)
        All (swath_time_idx, spacecraft_idx) pairs to consider.

    Returns
    -------
    list of (int, int)
        Slices that still need to be processed.
    """
    if callable(sparse_source):
        sparse_root = sparse_source()
    else:
        sparse_root = zarr.open(str(sparse_source), mode="r")
    out_root = zarr.open(str(out_path), mode="r")

    sparse_processed = sparse_root["processed"][:]   # (n_swath_time, n_spacecraft)
    pyramid_processed = out_root["0"]["processed"][:]

    pending = []
    for t_idx, s_idx in all_slices:
        if not sparse_processed[t_idx, s_idx]:
            continue
        if pyramid_processed[t_idx, s_idx]:
            continue
        pending.append((t_idx, s_idx))

    return pending


# ---------------------------------------------------------------------------
# Grid construction and NN lookup
# ---------------------------------------------------------------------------

def _build_regular_grid(resolution_deg):
    """Build 1D lat/lon coordinate arrays for a global regular grid."""
    half = resolution_deg / 2.0
    lats_1d = np.arange(90.0 - half, -90.0, -resolution_deg)
    lons_1d = np.arange(-180.0 + half, 180.0, resolution_deg)
    return lats_1d.astype("float64"), lons_1d.astype("float64")


def _compute_nn_lookup(grid, lats_1d, lons_1d, resolution_deg, max_dist_m):
    """For each lat/lon cell, find its nearest GPI using the grid's KDTree."""
    n_lat = len(lats_1d)
    n_lon = len(lons_1d)

    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    query_lons = lon_grid.ravel()
    query_lats = lat_grid.ravel()

    gpis_found, distances = grid.find_nearest_gpi(query_lons, query_lats)

    if max_dist_m is None:
        max_dist_m = 2.0 * resolution_deg * 111_000.0

    valid = distances <= max_dist_m

    all_gpis = grid.get_grid_points()[0]
    gpi_value_to_idx = np.full(int(all_gpis.max()) + 1, -1, dtype=np.int32)
    gpi_value_to_idx[all_gpis] = np.arange(len(all_gpis), dtype=np.int32)

    gpi_indices = gpi_value_to_idx[gpis_found]
    valid &= (gpi_indices >= 0)

    nn_gpi_indices = gpi_indices.reshape(n_lat, n_lon)
    nn_valid_mask = valid.reshape(n_lat, n_lon)

    return nn_gpi_indices, nn_valid_mask


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
# Output store creation
# ---------------------------------------------------------------------------

def _create_pyramid_store(
    out_path,
    sparse_root,
    beam_vars,
    scalar_vars,
    has_beams,
    lats_1d,
    lons_1d,
    resolution_deg,
    n_pyramid_levels,
    lat_chunk,
    lon_chunk,
):
    """Create the output Zarr store with multiscale pyramid groups."""
    out_path = Path(out_path)
    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    n_swath_time = sparse_root["swath_time"].shape[0]
    n_spacecraft = sparse_root["spacecraft"].shape[0]

    datasets = []
    for level in range(n_pyramid_levels):
        scale = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [1.0, 1.0, resolution_deg * scale, resolution_deg * scale],
                },
                {
                    "type": "translation",
                    "translation": [
                        0.0, 0.0,
                        90.0 - (resolution_deg * scale) / 2.0,
                        -180.0 + (resolution_deg * scale) / 2.0,
                    ],
                },
            ],
        })

    axes = [
        {"name": "swath_time", "type": "time"},
        {"name": "spacecraft", "type": ""},
        {"name": "latitude", "type": "space", "unit": "degree"},
        {"name": "longitude", "type": "space", "unit": "degree"},
    ]

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "regridded_swath_data",
            "axes": axes,
            "datasets": datasets,
            "type": "gaussian",
            "metadata": {
                "description": (
                    "Nearest-neighbor regridded ASCAT swath data with "
                    "Gaussian-smoothed pyramid levels"
                ),
            },
        }
    ]

    for level in range(n_pyramid_levels):
        scale = 2 ** level
        level_lats = lats_1d if level == 0 else _downsample_coords(lats_1d, scale)
        level_lons = lons_1d if level == 0 else _downsample_coords(lons_1d, scale)
        n_lat = len(level_lats)
        n_lon = len(level_lons)
        level_lat_chunk = max(1, lat_chunk // scale)
        level_lon_chunk = max(1, lon_chunk // scale)

        level_group = root.create_group(str(level))

        _create_level_arrays(
            group=level_group,
            sparse_root=sparse_root,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
            n_swath_time=n_swath_time,
            n_spacecraft=n_spacecraft,
            n_lat=n_lat,
            n_lon=n_lon,
            lat_chunk=level_lat_chunk,
            lon_chunk=level_lon_chunk,
            lats_1d=level_lats,
            lons_1d=level_lons,
            create_processed=True,
        )


def _downsample_coords(coords, scale):
    """Downsample coordinate array by taking block means."""
    n = len(coords)
    trimmed = n - (n % scale)
    return coords[:trimmed].reshape(-1, scale).mean(axis=1)


def _create_level_arrays(
    group,
    sparse_root,
    beam_vars,
    scalar_vars,
    has_beams,
    n_swath_time,
    n_spacecraft,
    n_lat,
    n_lon,
    lat_chunk,
    lon_chunk,
    lats_1d,
    lons_1d,
    create_processed=False,
):
    """Create arrays for a single pyramid level.

    Parameters
    ----------
    create_processed : bool
        If True, create the ``processed`` tracking array.  Should only be
        True for level 0, which is the authoritative record of completion.
    """
    compressors = [
        zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        )
    ]

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in sorted(beam_vars):
            src = sparse_root[var]
            group.create_array(
                name=var,
                dtype=src.dtype,
                shape=(n_swath_time, n_spacecraft, n_beams, n_lat, n_lon),
                chunks=(1, 1, 1, lat_chunk, lon_chunk),
                dimension_names=(
                    "swath_time", "spacecraft", "beam", "latitude", "longitude"
                ),
                fill_value=src.metadata.fill_value,
                compressors=compressors,
                attributes=dict(src.attrs),
            )

    for var in sorted(scalar_vars):
        src = sparse_root[var]
        group.create_array(
            name=var,
            dtype=src.dtype,
            shape=(n_swath_time, n_spacecraft, n_lat, n_lon),
            chunks=(1, 1, lat_chunk, lon_chunk),
            dimension_names=("swath_time", "spacecraft", "latitude", "longitude"),
            fill_value=src.metadata.fill_value,
            compressors=compressors,
            attributes=dict(src.attrs),
        )

    group.create_array(
        "swath_time",
        data=sparse_root["swath_time"][:],
        chunks=(1,),
        dimension_names=("swath_time",),
        compressors=None,
    )

    group.create_array(
        "spacecraft",
        data=sparse_root["spacecraft"][:],
        chunks=(1,),
        dimension_names=("spacecraft",),
        compressors=None,
    )

    if create_processed:
        group.create_array(
            "processed",
            shape=(n_swath_time, n_spacecraft),
            dtype="bool",
            chunks=(1, n_spacecraft),
            dimension_names=("swath_time", "spacecraft"),
            fill_value=False,
            compressors=None,
        )

    if has_beams:
        group.create_array(
            "beam",
            data=sparse_root["beam"][:],
            chunks=(1,),
            dimension_names=("beam",),
            fill_value=b"",
            compressors=None,
        )

    group.create_array(
        "latitude",
        data=lats_1d.astype("float64"),
        chunks=(lat_chunk,),
        dimension_names=("latitude",),
        compressors=None,
    )

    group.create_array(
        "longitude",
        data=lons_1d.astype("float64"),
        chunks=(lon_chunk,),
        dimension_names=("longitude",),
        compressors=None,
    )


# ---------------------------------------------------------------------------
# Slice-level regridding (level 0)
# ---------------------------------------------------------------------------

def _regrid_slice(
    ts_pair,
    sparse_opener,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
    nn_gpi_indices,
    nn_valid_mask,
    n_lat,
    n_lon,
):
    """Regrid a single (swath_time, spacecraft) slice from GPI to lat/lon."""
    t_idx, s_idx = ts_pair

    sparse_root = sparse_opener()
    out_root = zarr.open(out_path, mode="a")
    level0 = out_root["0"]

    # Quick check: does this slice have any data in the sparse store?
    # Use the processed array rather than probing chunk files directly,
    # which avoids needing to know whether the store is sharded or not.
    if not sparse_root["processed"][t_idx, s_idx]:
        return

    flat_gpi_idx = nn_gpi_indices[nn_valid_mask]

    for var in sorted(scalar_vars):
        src_arr = sparse_root[var]
        fill_val = src_arr.metadata.fill_value
        gpi_data = src_arr[t_idx, s_idx, :]
        grid_2d = np.full((n_lat, n_lon), fill_val, dtype=src_arr.dtype)
        grid_2d[nn_valid_mask] = gpi_data[flat_gpi_idx]
        level0[var][t_idx, s_idx, :, :] = grid_2d

    if has_beams:
        n_beams = sparse_root["beam"].shape[0]
        for var in sorted(beam_vars):
            src_arr = sparse_root[var]
            fill_val = src_arr.metadata.fill_value
            for b in range(n_beams):
                gpi_data = src_arr[t_idx, s_idx, b, :]
                grid_2d = np.full((n_lat, n_lon), fill_val, dtype=src_arr.dtype)
                grid_2d[nn_valid_mask] = gpi_data[flat_gpi_idx]
                level0[var][t_idx, s_idx, b, :, :] = grid_2d

    out_root["0"]["processed"][t_idx, s_idx] = True


# ---------------------------------------------------------------------------
# Pyramid building (levels 1..N)
# ---------------------------------------------------------------------------

def _build_pyramid_levels(
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
    n_pyramid_levels,
    n_swath_time,
    n_spacecraft,
    n_workers=1,
):
    """Build pyramid levels 1..N from level 0 using Gaussian smoothing.

    Skips slices that are already present in the destination level.
    """
    for level in range(1, n_pyramid_levels):
        print(f"  Building pyramid level {level}...")
        level_start = timer()

        all_slices = [
            (t, s) for t in range(n_swath_time) for s in range(n_spacecraft)
        ]

        # Use level-0 processed array as the source-of-truth for both
        # "has data" (src) and "already done" (dst) checks.
        # All pyramid levels are derived from level 0, so a slice is ready
        # to downsample iff level 0 is done, and a pyramid slice is done iff
        # it has already been downsampled (tracked via a per-level processed array).
        out_root = zarr.open(out_path, mode="r")
        level0_processed = out_root["0"]["processed"][:]

        dst_processed = out_root[str(level)]["processed"][:]

        pending_slices = []
        for t_idx, s_idx in all_slices:
            if not level0_processed[t_idx, s_idx]:
                continue
            if dst_processed[t_idx, s_idx]:
                continue
            pending_slices.append((t_idx, s_idx))

        if not pending_slices:
            print(f"  Level {level}: all slices already done, skipping")
            continue

        print(f"  Level {level}: {len(pending_slices)} slices to process")

        downsample_func = partial(
            _downsample_slice,
            out_path=out_path,
            src_level=level - 1,
            dst_level=level,
            beam_vars=beam_vars,
            scalar_vars=scalar_vars,
            has_beams=has_beams,
        )

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(downsample_func, s) for s in pending_slices]
                for future in futures:
                    future.result()
        else:
            for sl in pending_slices:
                downsample_func(sl)

        print(f"  Level {level} done in {timer() - level_start:.1f}s")


def _downsample_slice(
    ts_pair,
    out_path,
    src_level,
    dst_level,
    beam_vars,
    scalar_vars,
    has_beams,
):
    """Gaussian-smooth and 2x downsample a single slice between pyramid levels."""
    t_idx, s_idx = ts_pair

    root = zarr.open(out_path, mode="a")
    src_group = root[str(src_level)]
    dst_group = root[str(dst_level)]

    sigma = 1.0

    for var in sorted(scalar_vars):
        src_data = src_group[var][t_idx, s_idx, :, :]
        fill_val = src_group[var].metadata.fill_value
        downsampled = _gaussian_downsample_2d(src_data, fill_val, sigma)
        dst_group[var][t_idx, s_idx, :, :] = downsampled

    if has_beams:
        n_beams = src_group["beam"].shape[0]
        for var in sorted(beam_vars):
            for b in range(n_beams):
                src_data = src_group[var][t_idx, s_idx, b, :, :]
                fill_val = src_group[var].metadata.fill_value
                downsampled = _gaussian_downsample_2d(src_data, fill_val, sigma)
                dst_group[var][t_idx, s_idx, b, :, :] = downsampled

    dst_group["processed"][t_idx, s_idx] = True


def _gaussian_downsample_2d(data, fill_val, sigma):
    """Gaussian-smooth and 2x downsample a 2D array, respecting fill values."""
    out_dtype = data.dtype
    work = data.astype(np.float64)

    if np.issubdtype(data.dtype, np.floating):
        valid = np.isfinite(work) & (data != fill_val)
    else:
        valid = data != fill_val

    work[~valid] = 0.0
    weights = valid.astype(np.float64)

    smoothed_data = gaussian_filter(work, sigma=sigma, mode="constant", cval=0.0)
    smoothed_weights = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = smoothed_data / smoothed_weights

    h, w = normalized.shape
    h_trim = h - (h % 2)
    w_trim = w - (w % 2)
    downsampled = normalized[:h_trim:2, :w_trim:2]
    weight_down = smoothed_weights[:h_trim:2, :w_trim:2]

    no_data = weight_down < 1e-10

    if np.issubdtype(out_dtype, np.floating):
        result = downsampled.astype(out_dtype)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.round(downsampled).astype(out_dtype)
    result[no_data] = fill_val

    return result


def _maybe_expand_pyramid_swath_time(out_path, sparse_root):
    """Expand all pyramid level arrays along swath_time axis if the sparse store
    has grown beyond what the pyramid store was built for.

    Parameters
    ----------
    out_path : Path
        Path to the existing pyramid Zarr store.
    sparse_root : zarr.Group
        Opened sparse store (source of truth for swath_time).
    """
    out_root = zarr.open(str(out_path), mode="a")

    sparse_times = sparse_root["swath_time"][:]
    n_sparse_time = len(sparse_times)

    # All pyramid levels share the same swath_time axis size; check level 0
    level0 = out_root["0"]
    coord_names = {"swath_time", "spacecraft", "beam", "latitude", "longitude"}

    # Find a data array to check current size
    current_n_time = None
    for name in level0:
        arr = level0[name]
        if hasattr(arr, "ndim") and name not in coord_names and arr.ndim >= 2:
            current_n_time = arr.shape[0]
            break

    if current_n_time is None or n_sparse_time <= current_n_time:
        return

    n_new = n_sparse_time - current_n_time
    print(f"Expanding pyramid swath_time from {current_n_time} to {n_sparse_time} "
          f"(adding {n_new} new timesteps)")

    n_levels = len([k for k in out_root.keys() if k.isdigit()])

    for level in range(n_levels):
        level_group = out_root[str(level)]

        # Resize all data arrays along axis 0
        for name in level_group:
            arr = level_group[name]
            if not hasattr(arr, "ndim") or name in coord_names:
                continue
            new_shape = list(arr.shape)
            new_shape[0] = n_sparse_time
            arr.resize(tuple(new_shape))

        # processed is (swath_time, spacecraft) — resize axis 0
        if "processed" in level_group:
            level_group["processed"].resize(
                (n_sparse_time, level_group["processed"].shape[1])
            )

        # Update swath_time coordinate
        level_group["swath_time"].resize((n_sparse_time,))
        level_group["swath_time"][current_n_time:] = sparse_times[current_n_time:]


# =============================================================================
# Generic gpi-first regridder
# =============================================================================
#
# ``regrid_to_latlon`` above is shaped around the swath icechunk store, whose
# variables are dimensioned ``(swath_time, spacecraft, [beam,] gpi)`` — gpi is
# the *last* dim and the (time, slot) iteration is baked in.
#
# Many downstream products (the SWI ICDR is the first) have a different layout:
# gpi is the *first* dim, and the trailing dims are an arbitrary mix of a time
# axis plus zero-or-more "slot" / parameter dims. The function below handles
# that general gpi-first case. It is intentionally a separate top-level entry
# point rather than a refactor of ``regrid_to_latlon`` — gpi-first vs gpi-last
# stores need different array indexing and slice iteration, and trying to fuse
# them obscures both. The two functions share the genuinely-generic helpers
# (``_build_regular_grid``, ``_compute_nn_lookup``, ``_gaussian_downsample_2d``,
# ``_downsample_coords``).
#
# Output layout: variables are written with shape ``(*outer, lat, lon)`` where
# ``outer`` is the concatenation of (time_dim if present) + slot_dims, matching
# the dim order of the source. No incremental skip is supported — the caller
# is expected to delete the output store before each run (the SWI ICDR is
# rebuilt fresh per orchestration cycle).


def _nn_downsample_2d(data, fill_val):
    """2x downsample a 2D array by picking every other pixel (no smoothing).

    Used for variables where aggregation across cells is meaningless — quality
    flags (bitmasks), integer-encoded dates, categorical fields. Trim odd
    trailing rows/cols so the 2x relationship is preserved.
    """
    h, w = data.shape
    h_trim = h - (h % 2)
    w_trim = w - (w % 2)
    return data[:h_trim:2, :w_trim:2].copy()


def _classify_gpi_first_vars(sparse_root, include_vars, time_dim, slot_dims):
    """Categorize gpi-first variables by which outer dims they carry.

    Returns ``(time_varying, slot_static, fully_static)``: three lists of
    variable names. Categorization is by the variable's ``dimension_names``
    attribute, which zarr v3 always carries.

    - ``time_varying``: dims == ``(gpi, time_dim, *some_slot_dims)``
    - ``slot_static``: dims == ``(gpi, *some_slot_dims)`` — no time
    - ``fully_static``: dims == ``(gpi,)``

    Variables outside these three shapes (e.g. those carrying an ``obs`` dim)
    are rejected with a clear error so misconfiguration surfaces immediately
    instead of producing a half-built pyramid.
    """
    time_varying = []
    slot_static = []
    fully_static = []

    valid_outer = {time_dim} | set(slot_dims) if time_dim else set(slot_dims)

    for name in include_vars:
        arr = sparse_root[name]
        dims = tuple(arr.metadata.dimension_names or ())
        if not dims or dims[0] != "gpi":
            raise ValueError(
                f"Variable {name!r} has dims {dims}; expected gpi as first dim"
            )
        outer = dims[1:]
        if time_dim is not None and time_dim in outer:
            if not set(outer).issubset(valid_outer):
                raise ValueError(
                    f"Variable {name!r} has outer dims {outer} that include "
                    f"unsupported dims (not in time_dim/slot_dims). Drop it "
                    f"from include_vars."
                )
            time_varying.append(name)
        elif outer and set(outer).issubset(set(slot_dims)):
            slot_static.append(name)
        elif not outer:
            fully_static.append(name)
        else:
            raise ValueError(
                f"Variable {name!r} has unsupported outer dims {outer}; "
                f"expected subset of {{{time_dim}}} ∪ {set(slot_dims)}"
            )

    return time_varying, slot_static, fully_static


def regrid_gpi_first_zarr_to_latlon(
    *,
    sparse_opener,
    out_path,
    grid,
    time_dim,
    slot_dims=(),
    include_vars,
    nn_vars=(),
    resolution_deg=0.1,
    max_dist_m=None,
    n_pyramid_levels=4,
    lat_chunk=256,
    lon_chunk=256,
    n_workers=1,
):
    """Regrid a gpi-first sparse zarr store to a lat/lon multiscale pyramid.

    Source variables are expected to have ``gpi`` as their first dimension,
    optionally followed by a time dim and zero-or-more "slot" / parameter
    dims (e.g. ``ctime`` on the SWI ICDR). Each variable is regridded
    per-slice via nearest-neighbor lookup at level 0, then pyramid levels
    1..N-1 are built by 2x downsampling. Per-variable downsample policy:
    Gaussian smoothing by default, NN (decimation) for ``nn_vars``.

    The output store is built fresh — any existing ``out_path`` is removed.

    Parameters
    ----------
    sparse_opener : callable
        Zero-arg callable returning an opened ``zarr.Group`` over the
        source. Must be picklable across ``ProcessPoolExecutor`` workers
        (use a module-level function wrapped in ``functools.partial``).
    out_path : str or Path
        Output Zarr store path. Deleted and recreated.
    grid : FibGrid or similar
        Source grid with ``find_nearest_gpi`` and ``get_grid_points``.
    time_dim : str or None
        Name of the time dim (e.g. ``"daily_time_swi"``). If ``None``,
        all variables must be static (no time axis).
    slot_dims : tuple of str
        Names of "slot" / parameter dims preserved in the output
        (e.g. ``("ctime",)``).
    include_vars : list of str
        Variables to regrid. Each must have ``gpi`` as its first dim;
        trailing dims must be a subset of ``{time_dim} ∪ slot_dims``.
        Variables outside this shape are rejected.
    nn_vars : iterable of str
        Variables to downsample with nearest-neighbor decimation rather
        than Gaussian smoothing. Use for quality flags, integer-encoded
        dates, and other fields where smoothing is meaningless.
    resolution_deg, max_dist_m, n_pyramid_levels, lat_chunk, lon_chunk,
    n_workers
        See ``regrid_to_latlon``.
    """
    if time_dim is None and not slot_dims:
        # All-static is valid in principle but means the whole pyramid is
        # one (lat, lon) per variable — no slot/time iteration. Allowed.
        pass
    if not include_vars:
        raise ValueError("regrid_gpi_first_zarr_to_latlon: include_vars is required")

    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)

    sparse_root = sparse_opener()
    nn_vars = set(nn_vars)

    time_varying, slot_static, fully_static = _classify_gpi_first_vars(
        sparse_root, include_vars, time_dim, slot_dims
    )
    print(f"Time-varying vars ({time_dim}, *slots): {sorted(time_varying)}")
    print(f"Slot-static vars (*slots only): {sorted(slot_static)}")
    print(f"Fully-static vars (gpi only): {sorted(fully_static)}")
    print(f"NN-downsampled (no smoothing): {sorted(nn_vars)}")

    # --- Build base lat/lon grid + NN lookup ---
    lats_1d, lons_1d = _build_regular_grid(resolution_deg)
    n_lat = len(lats_1d)
    n_lon = len(lons_1d)
    print(f"Base grid: {n_lat} x {n_lon} ({resolution_deg}deg)")

    print("Computing lat/lon -> GPI nearest-neighbor lookup...")
    lookup_start = timer()
    nn_gpi_indices, nn_valid_mask = _compute_nn_lookup(
        grid, lats_1d, lons_1d, resolution_deg, max_dist_m
    )
    n_valid = nn_valid_mask.sum()
    print(f"Lookup computed in {timer() - lookup_start:.1f}s, "
          f"{n_valid}/{n_lat * n_lon} cells have a nearby GPI")

    # --- Slot dim sizes ---
    slot_sizes = tuple(int(sparse_root[d].shape[0]) for d in slot_dims)
    n_time = int(sparse_root[time_dim].shape[0]) if time_dim else None

    # --- Create output store ---
    print("Creating output Zarr structure...")
    _gf_create_pyramid_store(
        out_path=out_path,
        sparse_root=sparse_root,
        time_varying=time_varying,
        slot_static=slot_static,
        fully_static=fully_static,
        time_dim=time_dim,
        slot_dims=slot_dims,
        slot_sizes=slot_sizes,
        n_time=n_time,
        lats_1d=lats_1d,
        lons_1d=lons_1d,
        resolution_deg=resolution_deg,
        n_pyramid_levels=n_pyramid_levels,
        lat_chunk=lat_chunk,
        lon_chunk=lon_chunk,
    )

    # --- Regrid level 0 ---
    # Time-varying: iterate (t, *slot_idxs)
    # Slot-static: iterate (*slot_idxs,)  — gets written at every time? No:
    #              stored with shape (*slots, lat, lon), one write per slot tuple.
    # Fully-static: single write per var.
    time_varying_slices = list(
        itertools.product(range(n_time) if n_time else (None,),
                          *[range(sz) for sz in slot_sizes])
    ) if time_varying else []
    slot_static_slices = list(
        itertools.product(*[range(sz) for sz in slot_sizes])
    ) if slot_static else []

    print(f"Regridding {len(time_varying_slices)} time-varying slices, "
          f"{len(slot_static_slices)} slot-static slices, "
          f"{len(fully_static)} fully-static vars (level 0)")

    start = timer()
    if time_varying:
        regrid_tv = partial(
            _gf_regrid_time_varying_slice,
            sparse_opener=sparse_opener,
            out_path=str(out_path),
            time_varying=time_varying,
            nn_gpi_indices=nn_gpi_indices,
            nn_valid_mask=nn_valid_mask,
            n_lat=n_lat,
            n_lon=n_lon,
            has_time=time_dim is not None,
            n_slot_dims=len(slot_dims),
        )
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(regrid_tv, s) for s in time_varying_slices]
                for f in tqdm(futures, total=len(futures), desc="Regridding time-varying"):
                    f.result()
        else:
            for s in tqdm(time_varying_slices, desc="Regridding time-varying"):
                regrid_tv(s)

    if slot_static:
        regrid_ss = partial(
            _gf_regrid_slot_static_slice,
            sparse_opener=sparse_opener,
            out_path=str(out_path),
            slot_static=slot_static,
            nn_gpi_indices=nn_gpi_indices,
            nn_valid_mask=nn_valid_mask,
            n_lat=n_lat,
            n_lon=n_lon,
        )
        for s in tqdm(slot_static_slices, desc="Regridding slot-static"):
            regrid_ss(s)

    if fully_static:
        _gf_regrid_fully_static(
            sparse_opener=sparse_opener,
            out_path=str(out_path),
            fully_static=fully_static,
            nn_gpi_indices=nn_gpi_indices,
            nn_valid_mask=nn_valid_mask,
            n_lat=n_lat,
            n_lon=n_lon,
        )

    print(f"Level 0 done in {timer() - start:.1f}s")

    # --- Build pyramid levels ---
    if n_pyramid_levels > 1:
        print(f"Building {n_pyramid_levels - 1} pyramid levels...")
        pyr_start = timer()
        _gf_build_pyramid_levels(
            out_path=str(out_path),
            time_varying=time_varying,
            slot_static=slot_static,
            fully_static=fully_static,
            nn_vars=nn_vars,
            n_pyramid_levels=n_pyramid_levels,
            n_time=n_time,
            slot_sizes=slot_sizes,
            time_varying_slices=time_varying_slices,
            slot_static_slices=slot_static_slices,
            n_workers=n_workers,
        )
        print(f"Pyramids done in {timer() - pyr_start:.1f}s")

    print("All done!")


# ---------------------------------------------------------------------------
# gpi-first: output store creation
# ---------------------------------------------------------------------------

def _gf_create_pyramid_store(
    out_path,
    sparse_root,
    time_varying,
    slot_static,
    fully_static,
    time_dim,
    slot_dims,
    slot_sizes,
    n_time,
    lats_1d,
    lons_1d,
    resolution_deg,
    n_pyramid_levels,
    lat_chunk,
    lon_chunk,
):
    """Create the output store. One group per pyramid level, each carrying
    every variable at its level-specific lat/lon resolution. Slot/time
    coords are written once at level 0 and replicated into each level.
    """
    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    # OME-style multiscales metadata. Axes order matches array dim order:
    # [time_dim], *slot_dims, latitude, longitude. The scale applies only
    # to lat/lon — non-spatial dims get scale=1.0.
    axes = []
    if time_dim is not None:
        axes.append({"name": time_dim, "type": "time"})
    for d in slot_dims:
        axes.append({"name": d, "type": ""})
    axes.append({"name": "latitude", "type": "space", "unit": "degree"})
    axes.append({"name": "longitude", "type": "space", "unit": "degree"})

    n_outer = len(axes) - 2  # everything except lat/lon

    datasets = []
    for level in range(n_pyramid_levels):
        scale = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [1.0] * n_outer + [resolution_deg * scale,
                                                resolution_deg * scale],
                },
                {
                    "type": "translation",
                    "translation": [0.0] * n_outer + [
                        90.0 - (resolution_deg * scale) / 2.0,
                        -180.0 + (resolution_deg * scale) / 2.0,
                    ],
                },
            ],
        })

    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "regridded_gpi_first_data",
        "axes": axes,
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {
            "description": (
                "Nearest-neighbor regridded gpi-first data with mixed "
                "Gaussian/NN pyramid downsampling"
            ),
        },
    }]

    compressors = [
        zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        )
    ]

    for level in range(n_pyramid_levels):
        scale = 2 ** level
        level_lats = lats_1d if level == 0 else _downsample_coords(lats_1d, scale)
        level_lons = lons_1d if level == 0 else _downsample_coords(lons_1d, scale)
        n_lat_l = len(level_lats)
        n_lon_l = len(level_lons)
        level_lat_chunk = max(1, lat_chunk // scale)
        level_lon_chunk = max(1, lon_chunk // scale)

        level_group = root.create_group(str(level))

        # Time-varying: (time, *slots, lat, lon)
        for var in time_varying:
            src = sparse_root[var]
            shape = (n_time,) + slot_sizes + (n_lat_l, n_lon_l)
            chunks = (1,) + (1,) * len(slot_sizes) + (level_lat_chunk, level_lon_chunk)
            dim_names = (time_dim,) + tuple(slot_dims) + ("latitude", "longitude")
            level_group.create_array(
                name=var, dtype=src.dtype, shape=shape, chunks=chunks,
                dimension_names=dim_names,
                fill_value=src.metadata.fill_value,
                compressors=compressors,
                attributes=dict(src.attrs),
            )

        # Slot-static: (*slots, lat, lon)
        for var in slot_static:
            src = sparse_root[var]
            shape = slot_sizes + (n_lat_l, n_lon_l)
            chunks = (1,) * len(slot_sizes) + (level_lat_chunk, level_lon_chunk)
            dim_names = tuple(slot_dims) + ("latitude", "longitude")
            level_group.create_array(
                name=var, dtype=src.dtype, shape=shape, chunks=chunks,
                dimension_names=dim_names,
                fill_value=src.metadata.fill_value,
                compressors=compressors,
                attributes=dict(src.attrs),
            )

        # Fully-static: (lat, lon)
        for var in fully_static:
            src = sparse_root[var]
            level_group.create_array(
                name=var, dtype=src.dtype,
                shape=(n_lat_l, n_lon_l),
                chunks=(level_lat_chunk, level_lon_chunk),
                dimension_names=("latitude", "longitude"),
                fill_value=src.metadata.fill_value,
                compressors=compressors,
                attributes=dict(src.attrs),
            )

        # Coord arrays
        if time_dim is not None:
            level_group.create_array(
                time_dim, data=sparse_root[time_dim][:],
                chunks=(1,), dimension_names=(time_dim,), compressors=None,
            )
        for d in slot_dims:
            level_group.create_array(
                d, data=sparse_root[d][:],
                chunks=(1,), dimension_names=(d,), compressors=None,
            )
        level_group.create_array(
            "latitude", data=level_lats.astype("float64"),
            chunks=(level_lat_chunk,), dimension_names=("latitude",),
            compressors=None,
        )
        level_group.create_array(
            "longitude", data=level_lons.astype("float64"),
            chunks=(level_lon_chunk,), dimension_names=("longitude",),
            compressors=None,
        )


# ---------------------------------------------------------------------------
# gpi-first: level-0 regridding
# ---------------------------------------------------------------------------

def _gf_regrid_time_varying_slice(
    idx_tuple,
    sparse_opener,
    out_path,
    time_varying,
    nn_gpi_indices,
    nn_valid_mask,
    n_lat,
    n_lon,
    has_time,
    n_slot_dims,
):
    """Regrid one (time, *slots) slice for every time-varying variable.

    Source vars have dims ``(gpi, [time_dim,] *slot_dims)`` — we index the
    trailing dims to extract a 1D gpi vector, then scatter into a 2D
    (lat, lon) array via the precomputed NN lookup.
    """
    sparse_root = sparse_opener()
    out_root = zarr.open(out_path, mode="a")
    level0 = out_root["0"]

    flat_gpi_idx = nn_gpi_indices[nn_valid_mask]

    if has_time:
        src_idx = idx_tuple              # (t, *slots)
        out_idx = idx_tuple              # (t, *slots, :, :)
    else:
        src_idx = idx_tuple[1:]          # drop the None placeholder
        out_idx = idx_tuple[1:]

    for var in time_varying:
        src_arr = sparse_root[var]
        fill_val = src_arr.metadata.fill_value
        gpi_data = src_arr[(slice(None),) + src_idx]
        grid_2d = np.full((n_lat, n_lon), fill_val, dtype=src_arr.dtype)
        grid_2d[nn_valid_mask] = gpi_data[flat_gpi_idx]
        level0[var][out_idx + (slice(None), slice(None))] = grid_2d


def _gf_regrid_slot_static_slice(
    slot_idx_tuple,
    sparse_opener,
    out_path,
    slot_static,
    nn_gpi_indices,
    nn_valid_mask,
    n_lat,
    n_lon,
):
    """Regrid one (*slots,) slice for every slot-static variable.

    Source vars have dims ``(gpi, *slot_dims)`` — no time axis. Output
    shape per slice is (lat, lon), placed at the matching slot indices.
    """
    sparse_root = sparse_opener()
    out_root = zarr.open(out_path, mode="a")
    level0 = out_root["0"]
    flat_gpi_idx = nn_gpi_indices[nn_valid_mask]

    for var in slot_static:
        src_arr = sparse_root[var]
        fill_val = src_arr.metadata.fill_value
        gpi_data = src_arr[(slice(None),) + slot_idx_tuple]
        grid_2d = np.full((n_lat, n_lon), fill_val, dtype=src_arr.dtype)
        grid_2d[nn_valid_mask] = gpi_data[flat_gpi_idx]
        level0[var][slot_idx_tuple + (slice(None), slice(None))] = grid_2d


def _gf_regrid_fully_static(
    sparse_opener,
    out_path,
    fully_static,
    nn_gpi_indices,
    nn_valid_mask,
    n_lat,
    n_lon,
):
    """Regrid each fully-static (gpi,) variable into a single (lat, lon)."""
    sparse_root = sparse_opener()
    out_root = zarr.open(out_path, mode="a")
    level0 = out_root["0"]
    flat_gpi_idx = nn_gpi_indices[nn_valid_mask]

    for var in fully_static:
        src_arr = sparse_root[var]
        fill_val = src_arr.metadata.fill_value
        gpi_data = src_arr[:]
        grid_2d = np.full((n_lat, n_lon), fill_val, dtype=src_arr.dtype)
        grid_2d[nn_valid_mask] = gpi_data[flat_gpi_idx]
        level0[var][:, :] = grid_2d


# ---------------------------------------------------------------------------
# gpi-first: pyramid build (levels 1..N-1)
# ---------------------------------------------------------------------------

def _gf_build_pyramid_levels(
    out_path,
    time_varying,
    slot_static,
    fully_static,
    nn_vars,
    n_pyramid_levels,
    n_time,
    slot_sizes,
    time_varying_slices,
    slot_static_slices,
    n_workers,
):
    """Build pyramid levels by 2x downsampling each (lat, lon) slice."""
    for level in range(1, n_pyramid_levels):
        print(f"  Building pyramid level {level}...")
        level_start = timer()

        # Time-varying slices
        if time_varying:
            downsample_tv = partial(
                _gf_downsample_time_varying_slice,
                out_path=out_path,
                src_level=level - 1,
                dst_level=level,
                time_varying=time_varying,
                nn_vars=nn_vars,
                has_time=n_time is not None,
            )
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(downsample_tv, s)
                               for s in time_varying_slices]
                    for f in tqdm(futures, total=len(futures),
                                  desc=f"  Level {level} time-varying"):
                        f.result()
            else:
                for s in tqdm(time_varying_slices,
                              desc=f"  Level {level} time-varying"):
                    downsample_tv(s)

        # Slot-static + fully-static run serially — small, no need for workers
        if slot_static:
            for s in tqdm(slot_static_slices,
                          desc=f"  Level {level} slot-static"):
                _gf_downsample_slot_static_slice(
                    s, out_path=out_path, src_level=level - 1, dst_level=level,
                    slot_static=slot_static, nn_vars=nn_vars,
                )
        if fully_static:
            _gf_downsample_fully_static(
                out_path=out_path, src_level=level - 1, dst_level=level,
                fully_static=fully_static, nn_vars=nn_vars,
            )

        print(f"  Level {level} done in {timer() - level_start:.1f}s")


def _gf_apply_downsample(src_data, fill_val, var, nn_vars):
    """Pick the right per-variable downsampler."""
    if var in nn_vars:
        return _nn_downsample_2d(src_data, fill_val)
    return _gaussian_downsample_2d(src_data, fill_val, sigma=1.0)


def _gf_downsample_time_varying_slice(
    idx_tuple,
    out_path,
    src_level,
    dst_level,
    time_varying,
    nn_vars,
    has_time,
):
    root = zarr.open(out_path, mode="a")
    src_group = root[str(src_level)]
    dst_group = root[str(dst_level)]
    out_idx = idx_tuple if has_time else idx_tuple[1:]

    for var in time_varying:
        sel = out_idx + (slice(None), slice(None))
        src_data = src_group[var][sel]
        fill_val = src_group[var].metadata.fill_value
        dst_group[var][sel] = _gf_apply_downsample(src_data, fill_val, var, nn_vars)


def _gf_downsample_slot_static_slice(
    slot_idx_tuple,
    out_path,
    src_level,
    dst_level,
    slot_static,
    nn_vars,
):
    root = zarr.open(out_path, mode="a")
    src_group = root[str(src_level)]
    dst_group = root[str(dst_level)]

    for var in slot_static:
        sel = slot_idx_tuple + (slice(None), slice(None))
        src_data = src_group[var][sel]
        fill_val = src_group[var].metadata.fill_value
        dst_group[var][sel] = _gf_apply_downsample(src_data, fill_val, var, nn_vars)


def _gf_downsample_fully_static(
    out_path,
    src_level,
    dst_level,
    fully_static,
    nn_vars,
):
    root = zarr.open(out_path, mode="a")
    src_group = root[str(src_level)]
    dst_group = root[str(dst_level)]

    for var in fully_static:
        src_data = src_group[var][:, :]
        fill_val = src_group[var].metadata.fill_value
        dst_group[var][:, :] = _gf_apply_downsample(src_data, fill_val, var, nn_vars)
