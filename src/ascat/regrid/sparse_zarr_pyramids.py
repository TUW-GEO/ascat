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
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import time as timer

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ascat.utils import dtype_to_nan


def regrid_to_latlon(
    sparse_path,
    out_path,
    grid,
    resolution_deg=0.25,
    max_dist_m=None,
    n_pyramid_levels=4,
    lat_chunk=256,
    lon_chunk=256,
    n_workers=1,
):
    """Regrid a sparse swath-time Zarr cube to a regular lat/lon grid with pyramids.

    Parameters
    ----------
    sparse_path : str or Path
        Path to the sparse Zarr store with dims (swath_time, spacecraft, [beam,] gpi).
    out_path : str or Path
        Path for the output Zarr store with multiscale pyramid groups.
    grid : FibGrid or similar
        Grid object with a KDTree (``grid.kdTree``) and ``get_grid_points()``
        returning (gpis, lons, lats, cells).
    resolution_deg : float, optional
        Base resolution in degrees for the lat/lon grid. Default 0.25.
    max_dist_m : float or None, optional
        Maximum distance in meters for NN lookup. Cells beyond this distance
        from any GPI get fill values. Default is 2x the grid spacing
        (estimated as resolution_deg * 111_000 m).
    n_pyramid_levels : int, optional
        Number of pyramid levels (including base). Default 4.
        Level 0 = base resolution, level k = resolution * 2^k.
    lat_chunk : int, optional
        Chunk size along latitude. Default 256.
    lon_chunk : int, optional
        Chunk size along longitude. Default 256.
    n_workers : int, optional
        Number of parallel workers. Default 1.
    """
    sparse_path = Path(sparse_path)
    out_path = Path(out_path)

    sparse_root = zarr.open(sparse_path, mode="r")
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

    # --- Create output store with pyramid groups ---
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

    # --- Regrid level 0: parallel over (swath_time, spacecraft) ---
    slices = [
        (t, s) for t in range(n_swath_time) for s in range(n_spacecraft)
    ]
    print(f"Regridding {len(slices)} slices to level 0 with {n_workers} workers")

    regrid_func = partial(
        _regrid_slice,
        sparse_path=str(sparse_path),
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
            futures = [executor.submit(regrid_func, s) for s in slices]
            for future in tqdm(futures, total=len(futures), desc="Regridding level 0"):
                future.result()
    else:
        for sl in tqdm(slices, total=len(slices), desc="Regridding level 0"):
            regrid_func(sl)

    print(f"Level 0 done in {timer() - start:.1f}s")

    # --- Build pyramid levels 1..N via Gaussian downsampling ---
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
# Grid construction and NN lookup
# ---------------------------------------------------------------------------

def _build_regular_grid(resolution_deg):
    """Build 1D lat/lon coordinate arrays for a global regular grid.

    Latitude descends from north to south (image convention).
    Coordinates are cell centers.

    Returns
    -------
    lats_1d : np.ndarray
        Latitude centers, descending.
    lons_1d : np.ndarray
        Longitude centers, ascending.
    """
    half = resolution_deg / 2.0
    lats_1d = np.arange(90.0 - half, -90.0, -resolution_deg)
    lons_1d = np.arange(-180.0 + half, 180.0, resolution_deg)
    return lats_1d.astype("float64"), lons_1d.astype("float64")


def _compute_nn_lookup(grid, lats_1d, lons_1d, resolution_deg, max_dist_m):
    """For each lat/lon cell, find its nearest GPI using the grid's KDTree.

    Parameters
    ----------
    grid : FibGrid
        Source grid with KDTree.
    lats_1d, lons_1d : np.ndarray
        Target grid coordinate arrays.
    resolution_deg : float
        Grid resolution in degrees.
    max_dist_m : float or None
        Maximum lookup distance in meters. If None, defaults to
        2x the grid resolution (approx. resolution_deg * 111_000).

    Returns
    -------
    nn_gpi_indices : np.ndarray, shape (n_lat, n_lon)
        Index into the GPI array for the nearest neighbor.
        Invalid where nn_valid_mask is False.
    nn_valid_mask : np.ndarray of bool, shape (n_lat, n_lon)
        True where a GPI was found within the distance threshold.
    """
    n_lat = len(lats_1d)
    n_lon = len(lons_1d)

    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    query_lons = lon_grid.ravel()
    query_lats = lat_grid.ravel()

    gpis_found, distances = grid.find_nearest_gpi(query_lons, query_lats)

    if max_dist_m is None:
        max_dist_m = 2.0 * resolution_deg * 111_000.0

    valid = distances <= max_dist_m

    # We need indices into the GPI array (positional), not GPI values.
    # Build a value->index mapping.
    all_gpis = grid.get_grid_points()[0]
    gpi_value_to_idx = np.full(int(all_gpis.max()) + 1, -1, dtype=np.int32)
    gpi_value_to_idx[all_gpis] = np.arange(len(all_gpis), dtype=np.int32)

    gpi_indices = gpi_value_to_idx[gpis_found]
    # If find_nearest_gpi returned a GPI not in our grid, mark invalid
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

    # Build multiscales metadata
    datasets = []
    for level in range(n_pyramid_levels):
        scale = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [
                        1.0,  # swath_time
                        1.0,  # spacecraft
                        resolution_deg * scale,  # latitude
                        resolution_deg * scale,  # longitude
                    ],
                },
                {
                    "type": "translation",
                    "translation": [
                        0.0,
                        0.0,
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
):
    """Create arrays for a single pyramid level."""
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

    # Coordinates
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
    sparse_path,
    out_path,
    beam_vars,
    scalar_vars,
    has_beams,
    nn_gpi_indices,
    nn_valid_mask,
    n_lat,
    n_lon,
):
    """Regrid a single (swath_time, spacecraft) slice from GPI to lat/lon.

    For each lat/lon cell, reads the data at its nearest GPI. Cells without
    a nearby GPI (per nn_valid_mask) retain fill values.

    Parameters
    ----------
    ts_pair : tuple of (int, int)
        (swath_time_idx, spacecraft_idx).
    sparse_path : str
        Path to sparse Zarr store.
    out_path : str
        Path to output Zarr store.
    beam_vars, scalar_vars : set of str
        Variable name sets.
    has_beams : bool
        Whether beam dimension exists.
    nn_gpi_indices : np.ndarray, shape (n_lat, n_lon)
        Nearest GPI index for each lat/lon cell.
    nn_valid_mask : np.ndarray of bool, shape (n_lat, n_lon)
        True where a valid nearest GPI exists.
    n_lat, n_lon : int
        Output grid dimensions.
    """
    t_idx, s_idx = ts_pair

    sparse_root = zarr.open(sparse_path, mode="r")
    out_root = zarr.open(out_path, mode="a")
    level0 = out_root["0"]

    # Quick check: does this slice have any data?
    store_path = Path(sparse_root.store.root)
    n_gpi_chunks = -(-sparse_root["gpi"].shape[0] // sparse_root["time"].chunks[-1])
    has_any_data = False
    for gc in range(n_gpi_chunks):
        chunk_path = store_path / "time" / "c" / str(t_idx) / str(s_idx) / str(gc)
        if chunk_path.exists():
            has_any_data = True
            break

    if not has_any_data:
        return

    # Flat index array for valid cells — used for gathering from GPI data
    flat_gpi_idx = nn_gpi_indices[nn_valid_mask]

    for var in sorted(scalar_vars):
        src_arr = sparse_root[var]
        fill_val = src_arr.metadata.fill_value

        # Read full GPI vector for this slice
        gpi_data = src_arr[t_idx, s_idx, :]

        # Gather: for each valid lat/lon cell, take data from its nearest GPI
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
    """Build pyramid levels 1..N from level 0 using Gaussian smoothing."""
    for level in range(1, n_pyramid_levels):
        print(f"  Building pyramid level {level}...")
        level_start = timer()

        slices = [
            (t, s) for t in range(n_swath_time) for s in range(n_spacecraft)
        ]

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
                futures = [executor.submit(downsample_func, s) for s in slices]
                for future in futures:
                    future.result()
        else:
            for sl in slices:
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


def _gaussian_downsample_2d(data, fill_val, sigma):
    """Gaussian-smooth and 2x downsample a 2D array, respecting fill values.

    Uses normalized convolution: smooths both data (with fills zeroed) and a
    binary weight mask, then divides. Cells with negligible weight get fill.

    Parameters
    ----------
    data : np.ndarray, shape (H, W)
        Input array.
    fill_val : scalar
        Fill value to treat as missing.
    sigma : float
        Gaussian kernel sigma.

    Returns
    -------
    np.ndarray, shape (H//2, W//2)
        Downsampled array in the original dtype.
    """
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

    # 2x downsample by striding
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
