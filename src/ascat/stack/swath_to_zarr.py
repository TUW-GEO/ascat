# Copyright (c) 2025, TU Wien
# All rights reserved.

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

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Convert swath files to Zarr time-series format.

This module provides functionality to reorganize swath observation files into
a structured Zarr array indexed by time, spacecraft, and grid point index (GPI).
"""

from concurrent.futures import ProcessPoolExecutor
import re
import warnings
from datetime import timedelta
from functools import partial
from pathlib import Path
from time import time

import math
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from xarray.backends.zarr import FillValueCoder, encode_zarr_attr_value
from dask import delayed
from dask.base import compute
from tqdm import tqdm

from ascat.utils import dtype_to_nan

MISSION_SAT_IDS_MAP = {
    "ers": [1, 2],
    "metop": [3, 4, 5],
}

MISSION_SAT_ID_IDX_MAP = {
    "ers": {"1": 0, "2": 1},
    "metop": {"a": 0, "b": 1, "c": 2},
}

BEAM_SUFFIXES = {
    '_for': 0,
    '_fore': 0,
    '_mid': 1,
    '_aft': 2,
}


def _freq_to_timedelta64(time_resolution):
    # pd.date_range with 2 periods gives us the actual step size
    rng = pd.date_range("2000-01-01", periods=2, freq=time_resolution)
    return (rng[1] - rng[0]).to_timedelta64()


def stack_swaths_to_zarr(
    swath_files,
    out_path,
    date_range,
    time_resolution="h",
    n_workers=1,
    chunk_size_gpi=4096,
    gpi_shard_size=None,
    sorted_grid=None,
):
    """Convert swath files to Zarr time-series format.

    Creates a Zarr array with dimensions (swath_time, spacecraft, [beam,] gpi) and
    populates it with data from swath files. The Zarr structure is created on first
    call and subsequent calls append data, expanding the swath_time dimension if
    the requested date_range extends beyond the existing store.

    Parameters
    ----------
    swath_files : SwathGridFiles
        Swath file collection to convert.
    out_path : str or Path
        Output Zarr directory path.
    date_range : tuple of datetime
        (start, end) date range for data to include.
    time_resolution : str, optional
        Pandas frequency string for the time dimension, e.g. 'h' (hourly),
        '2h' (2-hourly), '3min' (3-minutely), 'D' (daily).
        Default is 'h'.
    n_workers : int, optional
        Number of worker processes for parallel processing. Default is 1.
    chunk_size_gpi : int, optional
        Chunk size (inner chunk) for the GPI dimension. Default is 4096.
    gpi_shard_size : int or None, optional
        Shard size along the GPI dimension.  Must be a multiple of
        ``chunk_size_gpi``.  If None (default), a balanced two-shard split is
        computed automatically as the smallest multiple of ``chunk_size_gpi``
        that is >= half the total number of GPIs.
    """
    out_path = Path(out_path)
    dt_start, dt_end = date_range

    if not (out_path / "zarr.json").exists():
        print(f"Creating Zarr structure at {out_path}")

        filenames = swath_files.search_period(
            dt_start=dt_start,
            dt_end=dt_start + timedelta(days=7),
            date_field_fmt=swath_files.date_field_fmt,
            end_inclusive=False
        )

        _create_zarr_structure(
            out_path=out_path,
            grid=sorted_grid or swath_files.grid,
            date_start=dt_start,
            date_end=dt_end,
            time_resolution=time_resolution,
            chunk_size_gpi=chunk_size_gpi,
            gpi_shard_size=gpi_shard_size,
            sat_series=swath_files.sat_series,
            sample_file=filenames[0]
        )
    else:
        # Store exists — expand swath_time if date_range extends past current end.
        _maybe_expand_swath_time(out_path, dt_end, time_resolution)

    print(f"Populating Zarr with data from {dt_start} to {dt_end}")

    zarr_root = zarr.open(out_path, mode="a")
    time_coords = zarr_root["swath_time"][:]

    _populate_zarr(
        swath_files=swath_files,
        zarr_root=zarr_root,
        time_coords=time_coords,
        time_resolution=time_resolution,
        date_range=date_range,
        n_workers=n_workers,
        sorted_grid=sorted_grid,
    )

    print("Done!")


def _maybe_expand_swath_time(out_path, dt_end, time_resolution):
    """Expand the swath_time dimension and all data arrays if dt_end is beyond
    the current store's last time coordinate.

    Parameters
    ----------
    out_path : Path
        Path to the existing Zarr store.
    dt_end : datetime
        Requested end date.
    time_resolution : str
        Pandas frequency string matching the store's time resolution.
    """
    root = zarr.open(out_path, mode="a")
    current_times = root["swath_time"][:]
    last_time = pd.Timestamp(current_times[-1])
    dt_end_ts = pd.Timestamp(dt_end)

    if dt_end_ts <= last_time:
        return

    # Generate only the new time steps beyond the current end
    new_times = pd.date_range(
        start=last_time,
        end=dt_end_ts,
        freq=time_resolution,
        inclusive="neither",  # exclude last_time itself
    ).values

    if len(new_times) == 0:
        return

    n_new = len(new_times)
    old_n_time = len(current_times)
    new_n_time = old_n_time + n_new
    print(f"Expanding swath_time from {old_n_time} to {new_n_time} steps "
          f"(adding {n_new} new timesteps up to {dt_end_ts})")

    # Resize all data arrays along axis 0 (swath_time)
    coord_names = {"swath_time", "spacecraft", "beam", "gpi", "longitude", "latitude"}
    for name in root:
        arr = root[name]
        if not hasattr(arr, "ndim"):
            continue
        if name in coord_names:
            continue
        new_shape = list(arr.shape)
        new_shape[0] = new_n_time
        arr.resize(tuple(new_shape))

    # Resize processed tracking array
    root["processed"].resize((new_n_time, root["processed"].shape[1]))

    # Append new time coordinates
    root["swath_time"].resize((new_n_time,))
    root["swath_time"][old_n_time:] = new_times


def _sanitize_attrs(attrs):
    sanitized = {}
    for key, value in attrs.items():
        if key == "scale_factor" and value == 1:
            continue
        # Fix common typo: calender -> calendar
        if key == "calender":
            sanitized["calendar"] = value
            continue
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def _balanced_shard_size(n_gpi, chunk_size_gpi, n_shards=2):
    """Compute the smallest multiple of chunk_size_gpi >= ceil(n_gpi / n_shards).

    Produces n_shards shards of roughly equal size rather than one large shard
    and one tiny remainder shard.
    """
    per_shard = math.ceil(n_gpi / n_shards)
    return math.ceil(per_shard / chunk_size_gpi) * chunk_size_gpi


def _create_zarr_structure(
    out_path,
    grid,
    date_start,
    date_end,
    time_resolution,
    chunk_size_gpi,
    gpi_shard_size,
    sat_series,
    sample_file,
):
    """Generate empty Zarr array with proper schema and dimensions.

    Parameters
    ----------
    out_path : Path
        Output Zarr directory path.
    grid : Grid
        Grid object defining spatial structure.
    date_start : datetime
        Start date for time dimension.
    date_end : datetime
        End date for time dimension.
    time_resolution : str
        Pandas frequency string (e.g., 'h', '2h', '3min', 'D').
    chunk_size_gpi : int
        Inner chunk size for GPI dimension.
    gpi_shard_size : int or None
        Shard size for GPI dimension.  None triggers automatic balanced split.
    sat_series : str
        Satellite series name (e.g., 'metop', 'ers') to determine number of spacecraft.
    sample_file : Path
        Sample file to determine schema.
    """
    time_coords = _generate_time_coords(date_start, date_end, time_resolution)
    n_time = len(time_coords)
    n_gpi = grid.n_gpi
    spacecraft_ids = MISSION_SAT_IDS_MAP.get(sat_series.lower())
    n_spacecraft = len(spacecraft_ids)

    if gpi_shard_size is None:
        gpi_shard_size = _balanced_shard_size(n_gpi, chunk_size_gpi)
    elif gpi_shard_size % chunk_size_gpi != 0:
        raise ValueError(
            f"gpi_shard_size ({gpi_shard_size}) must be a multiple of "
            f"chunk_size_gpi ({chunk_size_gpi})"
        )

    sample_ds = xr.open_dataset(
        sample_file,
        mask_and_scale=False,
        decode_cf=False,
        engine="h5netcdf"
    )

    has_beams, data_vars = _detect_beam_structure(sample_ds)
    sample_ds.close()

    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    if has_beams:
        dims = ("swath_time", "spacecraft", "beam", "gpi")
        n_beams = 3
        base_shape = (n_time, n_spacecraft, n_beams, n_gpi)
        inner_chunks = (1, 1, 1, chunk_size_gpi)
        # Shard shape: (t, s, beam) dims are size 1 so each shard file belongs
        # to exactly one (t, s, beam) slot — no concurrent write contention.
        base_shards = (1, 1, 1, gpi_shard_size)
    else:
        dims = ("swath_time", "spacecraft", "gpi")
        base_shape = (n_time, n_spacecraft, n_gpi)
        inner_chunks = (1, 1, chunk_size_gpi)
        base_shards = (1, 1, gpi_shard_size)

    for var in sorted(data_vars):
        var_dtype = sample_ds[var].dtype
        attrs = _sanitize_attrs(sample_ds[var].attrs)
        fill_val = attrs.get("_FillValue", dtype_to_nan[np.dtype(var_dtype)])
        attrs["_FillValue"] = FillValueCoder.encode(fill_val, var_dtype)

        root.create_array(
            name=var,
            dtype=var_dtype,
            shape=base_shape,
            chunks=inner_chunks,
            shards=base_shards,
            dimension_names=dims,
            fill_value=fill_val,
            compressors=zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=3,
                shuffle=zarr.codecs.BloscShuffle.shuffle
            ),
            attributes=attrs,
        )

    root.create_array(
        "swath_time",
        data=time_coords,
        chunks=(1,),
        dimension_names=("swath_time",),
        fill_value=dtype_to_nan[np.dtype("datetime64[ns]")],
        compressors=None,
    )

    root.create_array(
        "spacecraft",
        data=np.array(spacecraft_ids, dtype="int8"),
        chunks=(1,),
        dimension_names=("spacecraft",),
        fill_value=dtype_to_nan[np.dtype("int8")],
        compressors=None,
    )

    root.create_array(
        "processed",
        shape=(n_time, n_spacecraft),
        dtype="bool",
        chunks=(1, n_spacecraft),
        dimension_names=("swath_time", "spacecraft"),
        fill_value=False,
        compressors=None,
    )

    if has_beams:
        beam_names = np.array([b"fore", b"mid", b"aft"], dtype="S4")
        root.create_array(
            "beam",
            data=beam_names,
            chunks=(1,),
            dimension_names=("beam",),
            fill_value=b"",
            compressors=None,
        )

    # Handle both BasicGrid and CellGrid
    try:
        gpis, lons, lats, _ = grid.get_grid_points()
    except ValueError:
        gpis, lons, lats = grid.get_grid_points()

    root.create_array(
        "gpi",
        data=np.asarray(gpis, dtype="int32"),
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("int32")],
        compressors=None,
    )

    root.create_array(
        "longitude",
        data=np.asarray(lons, dtype="float32"),
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("float32")],
        compressors=None,
    )

    root.create_array(
        "latitude",
        data=np.asarray(lats, dtype="float32"),
        chunks=(chunk_size_gpi,),
        dimension_names=("gpi",),
        fill_value=dtype_to_nan[np.dtype("float32")],
        compressors=None,
    )


def _populate_zarr(
    swath_files,
    zarr_root,
    time_coords,
    time_resolution,
    date_range,
    sorted_grid=None,
    n_workers=1,
):
    """Fill Zarr array with data from swath files.

    Parameters
    ----------
    swath_files : SwathGridFiles
        Swath file collection.
    zarr_root : zarr.Group
        Opened Zarr group to write to.
    time_coords : np.ndarray
        Time coordinates array from Zarr.
    date_range : tuple of datetime
        (start, end) date range.
    time_resolution : str
        Pandas frequency string (e.g., 'h', '2h', '3min', 'D').
    sorted_grid : CellGrid, optional
        If provided, use this grid to map GPIs instead of the original grid.
    n_workers : int, optional
        Number of worker processes. Default is 1.
    """
    dt_start, dt_end = date_range

    filenames = swath_files.search_period(
        dt_start=dt_start,
        dt_end=dt_end,
        date_field_fmt=swath_files.date_field_fmt,
        end_inclusive=False
    )

    print(f"Found {len(filenames)} files to process")

    insert_func = partial(
        _insert_swath_file,
        swath_files=swath_files,
        zarr_root=zarr_root,
        time_coords=time_coords,
        time_resolution=time_resolution,
        sorted_grid=sorted_grid,
    )

    n_success = 0
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(insert_func, f)
                for f in filenames
            ]
            futures = tqdm(futures, total=len(filenames), desc="Processing files")
            n_success = sum(1 for future in futures if future.result())
    else:
        for i, filename in enumerate(filenames):
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(filenames)} files")
            if insert_func(filename):
                n_success += 1


def _insert_swath_file(filename, swath_files, zarr_root, time_coords, time_resolution, sorted_grid=None):
    """Insert data from one swath file into Zarr array.

    Parameters
    ----------
    filename : str or Path
        Path to swath file.
    swath_files : SwathGridFiles
        SwathGridFiles instance for parsing dates and extracting metadata.
    zarr_root : zarr.Group
        Opened Zarr group to write to.
    time_coords : np.ndarray
        Time coordinate array for indexing.
    time_resolution : str
        Pandas frequency string (e.g., 'h', '2h', '3min', 'D').
    sorted_grid : CellGrid, optional
        If provided, use this grid to map GPIs instead of the original grid.

    Returns
    -------
    bool
        True if insertion succeeded, False otherwise.
    """
    try:
        filename = Path(filename)

        dt = swath_files._parse_date(
            filename,
            date_field="date",
            date_field_fmt=swath_files.date_field_fmt
        )
        dt_np = np.datetime64(dt)

        sat_id = _extract_sat_id(
            filename.name,
            fn_pattern=swath_files.ft.fn_templ,
            sat_series=swath_files.sat_series,
        )

        time_delta = _freq_to_timedelta64(time_resolution)

        time_idx = np.searchsorted(time_coords, dt_np)
        time_idx_out_of_bounds = (time_idx >= len(time_coords))
        if time_idx_out_of_bounds or time_coords[time_idx] != dt_np:
            time_idx -= 1
        if time_coords[time_idx] + time_delta <= dt_np:
            time_idx += 1
        time_idx_out_of_bounds = (time_idx >= len(time_coords)) or (time_idx < 0)
        if time_idx_out_of_bounds:
            warnings.warn(
                f"Timestamp {dt_np} from {filename.name} not in time coordinates, skipping"
            )
            return False

        sat_idx = MISSION_SAT_ID_IDX_MAP[swath_files.sat_series][str(sat_id)]

        has_beams = "beam" in zarr_root

        with xr.open_dataset(
            filename,
            mask_and_scale=False,
            decode_cf=False,
            engine="h5netcdf",
        ) as ds:
            gpi = ds["location_id"].values.astype(int)

            if sorted_grid is not None:
                lookup = np.argsort(sorted_grid.get_grid_points()[0])
                gpi = lookup[gpi]

            start_time = time()
            for var in ds.data_vars:
                if var in ["location_id", "latitude", "longitude"]:
                    continue

                if var not in zarr_root:
                    warnings.warn(f"Variable {var} not in Zarr schema, skipping")
                    continue

                var_data = ds[var].values

                if has_beams:
                    beam_idx = _get_beam_index(var)
                    if beam_idx is not None:
                        for i, gpi_val in enumerate(gpi):
                            zarr_root[var][time_idx, sat_idx, beam_idx, gpi_val] = var_data[i]
                    elif "beams" in ds[var].dims and var_data.ndim == 2 and var_data.shape[1] == 3:
                        for b in range(3):
                            for i, gpi_val in enumerate(gpi):
                                zarr_root[var][time_idx, sat_idx, b, gpi_val] = var_data[i, b]
                    else:
                        for b in range(3):
                            for i, gpi_val in enumerate(gpi):
                                zarr_root[var][time_idx, sat_idx, b, gpi_val] = var_data[i]
                else:
                    for i, gpi_val in enumerate(gpi):
                        zarr_root[var][time_idx, sat_idx, gpi_val] = var_data[i]

        zarr_root["processed"][time_idx, sat_idx] = True
        return True

    except ValueError as e:
        warnings.warn(f"Skipping {filename}: {e}")
        return False
    except Exception as e:
        import traceback
        warnings.warn(f"Failed to insert {filename}: {e}\n{traceback.format_exc()}")
        return False


def _generate_time_coords(date_start, date_end, time_resolution):
    """Generate time coordinate array.

    Parameters
    ----------
    date_start : datetime
        Start date.
    date_end : datetime
        End date.
    time_resolution : str
        Pandas frequency string (e.g., 'h', '2h', '3min', 'D').

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] values.
    """
    return pd.date_range(
        start=date_start, end=date_end, freq=time_resolution, inclusive="left"
    ).values


def _detect_beam_structure(sample_ds):
    """Detect if dataset has beam variants and extract variable names.

    Parameters
    ----------
    sample_ds : xr.Dataset
        Sample dataset to analyze.

    Returns
    -------
    has_beams : bool
        True if dataset has beam-specific variables.
    data_vars : set of str
        Set of data variable names to include in Zarr.
    """
    data_vars = {
        var for var in sample_ds.data_vars
        if var not in ["location_id", "latitude", "longitude", "lon", "lat"]
    }

    has_beams = any(
        var.endswith(suffix)
        for var in data_vars
        for suffix in BEAM_SUFFIXES
    ) or any(
        "beams" in sample_ds[var].dims
        for var in data_vars
    )

    if not has_beams:
        return False, data_vars

    base_names = set()
    for var in data_vars:
        stripped = False
        for suffix in BEAM_SUFFIXES:
            if var.endswith(suffix):
                base_names.add(var[:-len(suffix)])
                stripped = True
                break
        if not stripped:
            base_names.add(var)

    result_vars = set()
    for base in base_names:
        beam_variants = [f"{base}{suffix}" for suffix in BEAM_SUFFIXES]
        if any(bv in sample_ds.data_vars for bv in beam_variants):
            result_vars.update(bv for bv in beam_variants if bv in sample_ds.data_vars)
        elif base in sample_ds.data_vars:
            result_vars.add(base)

    return True, result_vars


def _get_beam_index(var_name):
    """Extract beam index from variable name suffix.

    Parameters
    ----------
    var_name : str
        Variable name potentially containing beam suffix.

    Returns
    -------
    int or None
        Beam index (0, 1, 2) or None if no beam suffix.
    """
    for suffix, idx in BEAM_SUFFIXES.items():
        if var_name.endswith(suffix):
            return idx
    return None


def _extract_sat_id(filename, fn_pattern, sat_series):
    """Extract satellite ID from filename using the product's filename pattern.

    Parameters
    ----------
    filename : str
        Filename (not full path).
    fn_pattern : str
        Filename pattern with {sat} field (e.g., "...-METOP{sat}-...").
    sat_series : str
        Satellite series name (e.g., 'metop', 'ers', 'metop-sg').

    Returns
    -------
    int
        Satellite ID.

    Raises
    ------
    ValueError
        If filename doesn't match pattern or satellite can't be identified.
    """
    escaped_pattern = re.escape(fn_pattern)
    pattern_with_sat = re.sub(r'\\{sat\\}', r'(?P<sat>.*?)', escaped_pattern)
    pattern_with_wildcards = re.sub(r'\\{[^}]+\\}', r'.*?', pattern_with_sat)

    match = re.match(pattern_with_wildcards, filename)

    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern '{fn_pattern}'. "
            f"This file may be corrupted or from a different product."
        )

    try:
        sat_identifier = match.group('sat').lower()
    except (IndexError, AttributeError) as e:
        raise ValueError(
            f"Could not extract satellite identifier from '{filename}' "
            f"using pattern '{fn_pattern}'. Pattern may not contain {{sat}} field."
        ) from e

    sat_series_lower = sat_series.lower()
    mission_idxs = MISSION_SAT_ID_IDX_MAP.get(sat_series_lower)

    if mission_idxs is None or sat_identifier not in mission_idxs:
        known_sats = [
            f"{series.upper()} {id}"
            for series, ids in MISSION_SAT_ID_IDX_MAP.items()
            for id in ids
        ]
        raise ValueError(
            f"Unknown satellite: mission='{sat_series}', identifier='{sat_identifier}' "
            f"(from filename '{filename}'). "
            f"Known satellites: {', '.join(known_sats)}. "
            f"Add ({sat_series_lower!r}, {sat_identifier!r}): <sat_id> "
            f"to MISSION_SAT_ID_MAP in swath_to_zarr.py, or check that your mission "
            f"and fn_pattern are configured correctly."
        )

    return sat_identifier
