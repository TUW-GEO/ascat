"""
Shared pytest fixtures for swath_to_zarr, sparse_to_dense, and
sparse_zarr_to_latlon test suites.

All fixtures are designed to be fast (sub-second) and self-contained —
no real swath files required.
"""

import re
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import netCDF4
import numpy as np
import pytest
import zarr


# ---------------------------------------------------------------------------
# Constants matching the minimal synthetic dataset
# ---------------------------------------------------------------------------

N_GPI = 500          # small enough to be fast, large enough to be realistic
N_SWATH_TIME = 4     # hourly slots
N_SPACECRAFT = 3     # three Metop satellites (A, B, C → indices 0, 1, 2)
TIME_RESOLUTION = "60min"
SAT_SERIES = "metop"
CHUNK_SIZE_GPI = 50
SHARD_SIZE_GPI = 100   # 2 shards of 50 chunks each


# ---------------------------------------------------------------------------
# Synthetic grid fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_grid():
    """A minimal mock grid with N_GPI points arranged on a regular lat/lon
    grid.  Exposes the interface expected by stack_swaths_to_zarr and
    regrid_to_latlon (n_gpi, get_grid_points, find_nearest_gpi).
    """
    lats = np.linspace(-45.0, 45.0, N_GPI, dtype="float32")
    lons = np.linspace(-90.0, 90.0, N_GPI, dtype="float32")
    gpis = np.arange(N_GPI, dtype="int32")

    grid = MagicMock()
    grid.n_gpi = N_GPI
    grid.get_grid_points.return_value = (gpis, lons, lats, np.zeros(N_GPI, dtype="int32"))

    # find_nearest_gpi: returns (gpi_values, distances) for queried lon/lat arrays.
    # For testing, just map each query point to the nearest index by longitude.
    def _find_nearest(query_lons, query_lats):
        idx = np.clip(
            np.round((np.asarray(query_lons) + 90.0) / 180.0 * (N_GPI - 1)).astype(int),
            0, N_GPI - 1,
        )
        dist = np.abs(query_lons - lons[idx]) * 111_000.0
        return gpis[idx], dist.astype("float32")

    grid.find_nearest_gpi.side_effect = _find_nearest
    return grid


# ---------------------------------------------------------------------------
# Synthetic swath NetCDF files
# ---------------------------------------------------------------------------

def _write_synthetic_swath(path, gpi_indices, time_val, sat_letter, obs_values):
    """Write a minimal synthetic swath NetCDF file.

    Parameters
    ----------
    path : Path
        Output file path.
    gpi_indices : np.ndarray
        1D array of GPI (location_id) values to include in this swath.
    time_val : float
        Scalar time value (days since 1970-01-01) for all observations.
    sat_letter : str
        'a' or 'b' (used in filename only, not in file content).
    obs_values : dict[str, np.ndarray]
        Variable name -> 1D array of values, same length as gpi_indices.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("obs", len(gpi_indices))
        loc = ds.createVariable("location_id", "i4", ("obs",))
        loc[:] = gpi_indices
        lat = ds.createVariable("latitude", "f4", ("obs",))
        lat[:] = np.zeros(len(gpi_indices), dtype="f4")
        lon = ds.createVariable("longitude", "f4", ("obs",))
        lon[:] = np.zeros(len(gpi_indices), dtype="f4")
        for var_name, values in obs_values.items():
            v = ds.createVariable(var_name, values.dtype, ("obs",),
                                  fill_value=np.float32(-9999.0))
            v[:] = values
            v.units = "unitless"
            v.long_name = var_name


@pytest.fixture
def swath_file_factory(tmp_path):
    """Factory fixture: call with (slot_idx, sat_letter, gpi_indices, values_dict)
    to create a synthetic swath file with a predictable filename that
    SwathGridFiles can parse, or use directly with _insert_swath_file.
    """
    files_created = []

    def _make(slot_idx, sat_letter, gpi_indices, obs_values, time_offset_days=0.0):
        dt = datetime(2024, 12, 1) + __import__("datetime").timedelta(hours=slot_idx)
        date_str = dt.strftime("%Y%m%d%H%M%S")
        fname = (
            f"W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat_letter.upper()}"
            f"-6.25km-H139_C_LIIB_00000000000000_{date_str}____.nc"
        )
        fpath = tmp_path / "swaths" / fname
        time_val = float(
            (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0 + time_offset_days
        )
        obs_values_with_time = {"time": np.full(len(gpi_indices), time_val, dtype="f8")}
        obs_values_with_time.update(obs_values)
        _write_synthetic_swath(fpath, gpi_indices, time_val, sat_letter, obs_values_with_time)
        files_created.append(fpath)
        return fpath

    yield _make
    # cleanup handled by tmp_path


# ---------------------------------------------------------------------------
# Pre-built sparse swath zarr store
# ---------------------------------------------------------------------------

@pytest.fixture
def sparse_zarr(tmp_path, synthetic_grid):
    """A small but complete sparse swath Zarr store built by calling
    _create_zarr_structure and then directly calling _insert_swath_file for
    a handful of synthetic slots.  Does NOT depend on SwathGridFiles.

    Layout: N_SWATH_TIME=4 hourly slots, N_SPACECRAFT=2 (metop A+B),
    N_GPI=500.  Slots (0,0), (1,0), (2,1), (3,1) are populated.
    Variable: 'surface_soil_moisture' (float32).
    """
    from ascat.stack.swath_to_zarr import _create_zarr_structure, _insert_swath_file

    out_path = tmp_path / "sparse.zarr"

    sample_file = tmp_path / "sample.nc"
    import netCDF4 as nc
    with nc.Dataset(sample_file, "w") as ds:
        ds.createDimension("obs", 10)
        loc = ds.createVariable("location_id", "i4", ("obs",))
        loc[:] = np.arange(10, dtype="i4")
        lat = ds.createVariable("latitude", "f4", ("obs",))
        lat[:] = np.zeros(10, dtype="f4")
        lat.units = "degrees_north"
        lon = ds.createVariable("longitude", "f4", ("obs",))
        lon[:] = np.zeros(10, dtype="f4")
        lon.units = "degrees_east"
        ssm = ds.createVariable("surface_soil_moisture", "f4", ("obs",),
                               fill_value=np.float32(-9999.0))
        ssm[:] = np.ones(10, dtype="f4")
        ssm.units = "unitless"
        tm = ds.createVariable("time", "f8", ("obs",))
        tm.units = "days since 1970-01-01"
        tm.calendar = "standard"
        tm[:] = np.zeros(10, dtype="f8")

    from datetime import datetime
    _create_zarr_structure(
        out_path=out_path,
        grid=synthetic_grid,
        date_start=datetime(2024, 12, 1),
        date_end=datetime(2024, 12, 1, 4),
        time_resolution=TIME_RESOLUTION,
        chunk_size_gpi=CHUNK_SIZE_GPI,
        gpi_shard_size=SHARD_SIZE_GPI,
        sat_series=SAT_SERIES,
        sample_file=sample_file,
    )

    from unittest.mock import MagicMock
    swath_mock = MagicMock()
    swath_mock.sat_series = SAT_SERIES
    swath_mock.date_field_fmt = "%Y%m%d%H%M%S"
    swath_mock.ft.fn_templ = (
        "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H139_C_LIIB_"
        "{placeholder}_{date}____.nc"
    )
    swath_mock.grid = synthetic_grid

    import re as _re
    def _parse_date(path, date_field, date_field_fmt):
        m = _re.search(r"_(\d{14})____\.nc$", str(path))
        if not m:
            raise ValueError(f"Cannot parse date from {path}")
        return datetime.strptime(m.group(1), date_field_fmt)

    swath_mock._parse_date.side_effect = _parse_date

    root = zarr.open(out_path, mode="a")
    time_coords = root["swath_time"][:]

    slot_configs = [
        (0, "a", np.arange(50)),
        (1, "a", np.arange(50)),
        (2, "b", np.arange(50)),
        (3, "b", np.arange(50)),
    ]

    for slot_idx, sat_letter, gpi_indices in slot_configs:
        from unittest.mock import patch
        swath_file = tmp_path / f"swath_{slot_idx}_{sat_letter}.nc"
        with nc.Dataset(swath_file, "w") as ds:
            ds.createDimension("obs", len(gpi_indices))
            loc = ds.createVariable("location_id", "i4", ("obs",))
            loc[:] = gpi_indices.astype("i4")
            lat = ds.createVariable("latitude", "f4", ("obs",))
            lat[:] = np.zeros(len(gpi_indices), dtype="f4")
            lat.units = "degrees_north"
            lon = ds.createVariable("longitude", "f4", ("obs",))
            lon[:] = np.zeros(len(gpi_indices), dtype="f4")
            lon.units = "degrees_east"
            tm = ds.createVariable("time", "f8", ("obs",))
            tm.units = "days since 1970-01-01"
            tm.calendar = "standard"
            dt = datetime(2024, 12, 1) + timedelta(hours=slot_idx)
            tm[:] = (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0
            ssm = ds.createVariable("surface_soil_moisture", "f4", ("obs",),
                                   fill_value=np.float32(-9999.0))
            ssm[:] = gpi_indices.astype("f4") / 100.0
            ssm.units = "unitless"

        _insert_swath_file(swath_file, swath_mock, root, time_coords, TIME_RESOLUTION)

    root = zarr.open(out_path, mode="r")
    return out_path


# ---------------------------------------------------------------------------
# Pre-built dense timeseries zarr store
# ---------------------------------------------------------------------------

@pytest.fixture
def dense_zarr(tmp_path, sparse_zarr):
    """Dense timeseries store built from sparse_zarr by calling sparse_to_dense."""
    from ascat.stack.sparse_zarr_to_ts import sparse_to_dense

    out_path = tmp_path / "dense.zarr"
    sparse_to_dense(
        sparse_zarr,
        out_path,
        chunk_size_gpi=CHUNK_SIZE_GPI,
        chunk_size_obs=10,
        n_workers=1,
    )
    return out_path


# ---------------------------------------------------------------------------
# Pre-built pyramid zarr store
# ---------------------------------------------------------------------------

@pytest.fixture
def pyramid_zarr(tmp_path, sparse_zarr, synthetic_grid):
    """Pyramid store built from sparse_zarr by calling regrid_to_latlon."""
    from ascat.regrid.sparse_zarr_pyramids import regrid_to_latlon

    out_path = tmp_path / "pyramids.zarr"
    regrid_to_latlon(
        sparse_zarr,
        out_path,
        grid=synthetic_grid,
        resolution_deg=5.0,   # coarse grid for speed
        max_dist_m=500_000,
        n_pyramid_levels=2,
        lat_chunk=4,
        lon_chunk=4,
        n_workers=1,
    )
    return out_path
