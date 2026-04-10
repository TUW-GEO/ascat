#!/usr/bin/env python3

import xarray as xr
import numpy as np

contiguous_ragged_ds_2588 = xr.Dataset(
    {
        "lon": (
            "locations",
            np.array(
                [175.80013, 175.37308, 179.1304, 179.82138, 178.70335], dtype=np.float32
            ),
        ),
        "lat": (
            "locations",
            np.array(
                [70.00758, 70.04549, 70.65371, 70.67787, 70.692825], dtype=np.float32
            ),
        ),
        "alt": (
            "locations",
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        ),
        "location_id": (
            "locations",
            np.array([1549346, 1549723, 1555679, 1555912, 1556056], dtype=np.int64),
          {"cf_role": "timeseries_id"}
        ),
        "row_size": (
            "locations",
            np.array([2, 1, 2, 4, 1], dtype=np.int32),
            {"sample_dimension": "time"},
        ),
        "time": (
            "time",
            np.array(
                [
                    np.datetime64("2020-01-01T00:00:00"),
                    np.datetime64("2020-01-01T00:00:01"),
                    np.datetime64("2020-01-01T00:00:02"),
                    np.datetime64("2020-01-01T00:00:03"),
                    np.datetime64("2020-01-01T00:00:24"),
                    np.datetime64("2020-01-01T00:00:05"),
                    np.datetime64("2020-01-01T00:00:06"),
                    np.datetime64("2020-01-01T00:00:07"),
                    np.datetime64("2020-01-01T00:00:08"),
                    np.datetime64("2020-01-01T00:00:09"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "sm":
            (
                ["beam", "time"],
                np.array(
                    [
                        [0.1,
                         0.2,
                         0.3,
                         0.4,
                         0.5,
                         0.6,
                         0.7,
                         0.8,
                         0.9,
                         1.0,],
                        [0.2,
                         0.3,
                         0.4,
                         0.5,
                         0.6,
                         0.7,
                         0.8,
                         0.9,
                         1.0,
                         1.1,],
                        [0.3,
                         0.4,
                         0.5,
                         0.6,
                         0.7,
                         0.8,
                         0.9,
                         1.0,
                         1.1,
                         1.2,],
                    ],
                    dtype=np.float32,
                ),
            ),
        "beam":
            (
                "beam",
                ["for", "mid", "aft"],
            ),
    },
    attrs={"featureType": "timeseries"},
).set_coords(["lon", "lat", "alt"])

indexed_ragged_ds_2588 = xr.Dataset(
    {
        "lon": (
            "locations",
            np.array(
                [175.80013, 175.37308, 179.1304, 179.82138, 178.70335], dtype=np.float32
            ),
        ),
        "lat": (
            "locations",
            np.array(
                [70.00758, 70.04549, 70.65371, 70.67787, 70.692825], dtype=np.float32
            ),
        ),
        "alt": (
            "locations",
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        ),
        "location_id": (
            "locations",
            np.array([1549346, 1549723, 1555679, 1555912, 1556056], dtype=np.int64),
          {"cf_role": "timeseries_id"}
        ),
        "locationIndex": (
            "time",
            np.array([0, 0, 1, 2, 2, 3, 3, 3, 3, 4], dtype=np.int32),
            {"instance_dimension": "locations"},
        ),
        "time": (
            "time",
            np.array(
                [
                    np.datetime64("2020-01-01T00:00:00"),
                    np.datetime64("2020-01-01T00:00:01"),
                    np.datetime64("2020-01-01T00:00:02"),
                    np.datetime64("2020-01-01T00:00:03"),
                    np.datetime64("2020-01-01T00:00:24"),
                    np.datetime64("2020-01-01T00:00:05"),
                    np.datetime64("2020-01-01T00:00:06"),
                    np.datetime64("2020-01-01T00:00:07"),
                    np.datetime64("2020-01-01T00:00:08"),
                    np.datetime64("2020-01-01T00:00:09"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "sm":
            (
                ["beam", "time"],
                np.array(
                    [
                        [0.1,
                         0.2,
                         0.3,
                         0.4,
                         0.5,
                         0.6,
                         0.7,
                         0.8,
                         0.9,
                         1.0,],
                        [0.2,
                         0.3,
                         0.4,
                         0.5,
                         0.6,
                         0.7,
                         0.8,
                         0.9,
                         1.0,
                         1.1,],
                        [0.3,
                         0.4,
                         0.5,
                         0.6,
                         0.7,
                         0.8,
                         0.9,
                         1.0,
                         1.1,
                         1.2,],
                    ],
                    dtype=np.float32,
                ),
            ),
        "beam":
            (
                "beam",
                ["for", "mid", "aft"],
            ),
    },
    attrs={"featureType": "timeseries"},
).set_coords(["lon", "lat", "alt"])

contiguous_ragged_ds_2587 = xr.Dataset(
    {
        "lon": (
            "locations",
            np.array(
                [175.88971, 177.6987, 179.5077, 176.58069, 178.38968], dtype=np.float32
            ),
        ),
        "lat": (
            "locations",
            np.array(
                [65.00168, 65.00892, 65.01617, 65.020645, 65.02789], dtype=np.float32
            ),
        ),
        "alt": (
            "locations",
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        ),
        "location_id": (
            "locations",
            np.array([1493629, 1493718, 1493807, 1493862, 1493951], dtype=np.int64),
          {"cf_role": "timeseries_id"}
        ),
        "row_size": ("locations", np.array([3, 1, 1, 3, 2], dtype=np.int32), {"sample_dimension": "time"}),
        "time": (
            "time",
            np.array(
                [
                    np.datetime64("2020-01-01T00:00:01"),
                    np.datetime64("2020-01-01T00:00:03"),
                    np.datetime64("2020-01-01T00:00:24"),
                    np.datetime64("2020-01-01T00:00:05"),
                    np.datetime64("2020-01-01T00:00:07"),
                    np.datetime64("2020-01-01T00:00:09"),
                    np.datetime64("2020-01-01T00:00:10"),
                    np.datetime64("2020-01-01T00:00:11"),
                    np.datetime64("2020-01-01T00:00:14"),
                    np.datetime64("2020-01-01T00:00:18"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "sm":
            (
                ["beam", "time"],
                np.array(
                    [
                        [1.1,
                         1.2,
                         1.3,
                         1.4,
                         1.5,
                         1.6,
                         1.7,
                         1.8,
                         1.9,
                         2.0,],
                        [1.2,
                         1.3,
                         1.4,
                         1.5,
                         1.6,
                         1.7,
                         1.8,
                         1.9,
                         2.0,
                         2.1,],
                        [1.3,
                         1.4,
                         1.5,
                         1.6,
                         1.7,
                         1.8,
                         1.9,
                         1.0,
                         2.1,
                         2.2,],
                    ],
                    dtype=np.float32,
                ),
            ),
        "beam":
            (
                "beam",
                ["for", "mid", "aft"],
            ),
    },
    attrs={"featureType": "timeseries"},
).set_coords(["lon", "lat", "alt"])

indexed_ragged_ds_2587 = xr.Dataset(
    {
        "lon": (
            "locations",
            np.array(
                [175.88971, 177.6987, 179.5077, 176.58069, 178.38968], dtype=np.float32
            ),
        ),
        "lat": (
            "locations",
            np.array(
                [65.00168, 65.00892, 65.01617, 65.020645, 65.02789], dtype=np.float32
            ),
        ),
        "alt": (
            "locations",
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32),
        ),
        "location_id": (
            "locations",
            np.array([1493629, 1493718, 1493807, 1493862, 1493951], dtype=np.int64),
          {"cf_role": "timeseries_id"}
        ),
        "locationIndex": (
            "time",
            np.array([0, 0, 0, 1, 2, 3, 3, 3, 4, 4], dtype=np.int32),
            {"instance_dimension": "locations"},
        ),
        "time": (
            "time",
            np.array(
                [
                    np.datetime64("2020-01-01T00:00:01"),
                    np.datetime64("2020-01-01T00:00:03"),
                    np.datetime64("2020-01-01T00:00:24"),
                    np.datetime64("2020-01-01T00:00:05"),
                    np.datetime64("2020-01-01T00:00:07"),
                    np.datetime64("2020-01-01T00:00:09"),
                    np.datetime64("2020-01-01T00:00:10"),
                    np.datetime64("2020-01-01T00:00:11"),
                    np.datetime64("2020-01-01T00:00:14"),
                    np.datetime64("2020-01-01T00:00:18"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "sm":
            (
                ["beam", "time"],
                np.array(
                    [
                        [1.1,
                         1.2,
                         1.3,
                         1.4,
                         1.5,
                         1.6,
                         1.7,
                         1.8,
                         1.9,
                         2.0,],
                        [1.2,
                         1.3,
                         1.4,
                         1.5,
                         1.6,
                         1.7,
                         1.8,
                         1.9,
                         2.0,
                         2.1,],
                        [1.3,
                         1.4,
                         1.5,
                         1.6,
                         1.7,
                         1.8,
                         1.9,
                         1.0,
                         2.1,
                         2.2,],
                    ],
                    dtype=np.float32,
                ),
            ),
        "beam":
            (
                "beam",
                ["for", "mid", "aft"],
            ),
    },
    attrs={"featureType": "timeseries"},
).set_coords(["lon", "lat", "alt"])

swath_ds = xr.Dataset(
    {
        "longitude": ("obs", np.array([143.2, 143.3, 143.1, 143.2], dtype=np.float64)),
        "latitude": ("obs", np.array([42.01, 42.08, 42.13, 42.21], dtype=np.float64)),
        "location_id": (
            "obs",
            np.array([1100178, 1101775, 1102762, 1104359], dtype=np.int64),
          {"cf_role": "timeseries_id"}
        ),
        "time": (
            "obs",
            np.array(
                [
                    np.datetime64("2021-11-15T09:04:49.940999936"),
                    np.datetime64("2021-11-15T09:04:50.790000128"),
                    np.datetime64("2021-11-15T09:04:51.639000064"),
                    np.datetime64("2021-11-15T09:04:52.488000000"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "surface_soil_moisture": (
            "obs",
            np.array([58.18, 57.43, 55.469997, 47.489998], dtype=np.float32),
        ),
    },
    attrs={"featureType": "point"},
)

swath_ds_2 = xr.Dataset(
    {
        "longitude": (
            "obs",
            np.array(
                [142.937536, 143.302272, 143.038352, 142.774416],
                dtype=np.float64
            ),
        ),
        "latitude": (
            "obs",
            np.array([42.176548, 42.279804, 42.251248, 42.222704], 
                    dtype=np.float64),
        ),
        "location_id": (
            "obs",
            np.array([1103749.0, 1105956.0, 1105346.0, 1104736.0], dtype=np.int64),
          {"cf_role": "timeseries_id"}
        ),
        "time": (
            "obs",
            np.array(
                [
                    np.datetime64("2021-11-15T09:04:53.338000128"),
                    np.datetime64("2021-11-15T09:04:53.338000128"),
                    np.datetime64("2021-11-15T09:04:54.188000000"),
                    np.datetime64("2021-11-15T09:04:55.036000000"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "surface_soil_moisture": (
            "obs",
            np.array([46.289997, 39.629997, 40.36, 44.19], dtype=np.float32),
        ),
    },
    attrs={"featureType": "point"},
)


def generate_synthetic_swath_data(
    location_ids,
    lons,
    lats,
    timestamp,
    with_beams=False,
    with_existing_beam_dim=False,
    seed=None
):
    """
    Generate synthetic swath dataset for testing swath_to_zarr.
    
    Parameters
    ----------
    location_ids : np.ndarray
        Grid point indices (GPIs)
    lons : np.ndarray
        Longitude values for each GPI
    lats : np.ndarray
        Latitude values for each GPI
    timestamp : np.datetime64
        Timestamp for the swath
    with_beams : bool, optional
        If True, include beam-specific variables (with suffixes _for, _mid, _aft)
    with_existing_beam_dim : bool, optional
        If True, include variables with an existing beam dimension
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    xr.Dataset
        Synthetic swath dataset
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_obs = len(location_ids)
    
    data_vars = {
        "location_id": (
            "obs", location_ids.astype(np.int32), 
            {"cf_role": "timeseries_id"}
        ),
        "longitude": ("obs", lons.astype(np.float32)),
        "latitude": ("obs", lats.astype(np.float32)),
        "time": ("obs", np.full(n_obs, timestamp, dtype="datetime64[ns]")),
    }
    
    # Surface soil moisture (0-100 range typical for H129/H139)
    data_vars["surface_soil_moisture"] = (
        "obs", 
        np.random.uniform(0, 100, n_obs).astype(np.float32),
        {"_FillValue": -9999.0, "units": "%"}
    )
    
    # Backscatter at 40 degree incidence angle
    data_vars["backscatter40"] = (
        "obs",
        np.random.uniform(-10, 10, n_obs).astype(np.float32),
        {"_FillValue": -9999.0, "units": "dB"}
    )
    
    # Slope and curvature
    data_vars["slope40"] = (
        "obs",
        np.random.uniform(-5, 5, n_obs).astype(np.float32),
        {"_FillValue": -9999.0}
    )
    data_vars["curvature40"] = (
        "obs",
        np.random.uniform(-2, 2, n_obs).astype(np.float32),
        {"_FillValue": -9999.0}
    )
    
    # Soil moisture sensitivity
    data_vars["surface_soil_moisture_sensitivity"] = (
        "obs",
        np.random.uniform(0.5, 1.0, n_obs).astype(np.float32),
        {"_FillValue": -9999.0, "units": "1"}
    )
    
    # Flags (uint8)
    data_vars["snow_cover_probability"] = (
        "obs",
        np.random.randint(0, 101, n_obs, dtype=np.uint8),
        {"_FillValue": 255}
    )
    data_vars["frozen_soil_probability"] = (
        "obs",
        np.random.randint(0, 101, n_obs, dtype=np.uint8),
        {"_FillValue": 255}
    )
    data_vars["subsurface_scattering_probability"] = (
        "obs",
        np.random.randint(0, 101, n_obs, dtype=np.uint8),
        {"_FillValue": 255}
    )
    
    # Processing and correction flags (int8)
    data_vars["complexity_flag"] = (
        "obs",
        np.random.randint(0, 16, n_obs, dtype=np.int8),
        {"_FillValue": -128}
    )
    data_vars["correction_flag"] = (
        "obs",
        np.random.randint(0, 16, n_obs, dtype=np.int8),
        {"_FillValue": -128}
    )
    
    if with_existing_beam_dim:
        # Variables with existing beam dimension
        data_vars["sig"] = (
            ("obs", "beam"),
            np.random.uniform(-15, -5, (n_obs, 3)).astype(np.float32),
            {"_FillValue": -9999.0, "units": "dB"}
        )
        data_vars["azi"] = (
            ("obs", "beam"),
            np.random.uniform(0, 360, (n_obs, 3)).astype(np.float32),
            {"_FillValue": -9999.0, "units": "degrees"}
        )
        data_vars["inc"] = (
            ("obs", "beam"),
            np.random.uniform(20, 60, (n_obs, 3)).astype(np.float32),
            {"_FillValue": -9999.0, "units": "degrees"}
        )
        data_vars["kp"] = (
            ("obs", "beam"),
            np.random.uniform(0, 1, (n_obs, 3)).astype(np.float32),
            {"_FillValue": -9999.0}
        )
    elif with_beams:
        # Beam-specific backscatter (with suffixes)
        data_vars["backscatter_for"] = (
            "obs",
            np.random.uniform(-10, 10, n_obs).astype(np.float32),
            {"_FillValue": -9999.0, "units": "dB"}
        )
        data_vars["backscatter_mid"] = (
            "obs",
            np.random.uniform(-10, 10, n_obs).astype(np.float32),
            {"_FillValue": -9999.0, "units": "dB"}
        )
        data_vars["backscatter_aft"] = (
            "obs",
            np.random.uniform(-10, 10, n_obs).astype(np.float32),
            {"_FillValue": -9999.0, "units": "dB"}
        )
        
        # Beam-specific incidence angles (with suffixes)
        data_vars["incidence_angle_for"] = (
            "obs",
            np.random.uniform(20, 60, n_obs).astype(np.float32),
            {"_FillValue": -9999.0, "units": "degrees"}
        )
        data_vars["incidence_angle_mid"] = (
            "obs",
            np.random.uniform(20, 60, n_obs).astype(np.float32),
            {"_FillValue": -9999.0, "units": "degrees"}
        )
        data_vars["incidence_angle_aft"] = (
            "obs",
            np.random.uniform(20, 60, n_obs).astype(np.float32),
            {"_FillValue": -9999.0, "units": "degrees"}
        )
    
    attrs = {
        "featureType": "point",
        "title": "ASCAT surface soil moisture test product",
        "instrument": "ASCAT",
        "satellite": "Metop-A",
        "conventions": "CF-1.10"
    }
    
    # Add the misspelled calendar attribute for testing
    attrs["calender"] = "proleptic_gregorian"
    
    return xr.Dataset(data_vars, attrs=attrs)


def create_test_grid(n_points=500):
    """
    Create a proper standalone test Fibonacci grid with latband sorting.
    
    Uses compute_fib_grid(n) to generate a standalone N-point Fibonacci lattice,
    then determines latband sorting by clustering points by latitude bands.
    
    Parameters
    ----------
    n_points : int, optional
        Desired number of grid points. Default 500.
        Note: compute_fib_grid(n) generates 2n+1 points, so the actual
        number of points will be the nearest 2n+1 to n_points.
        
    Returns
    -------
    dict
        Dictionary with keys 'gpi', 'lon', 'lat', 'latband_sorting', 'cluster_id'
    """
    try:
        from fibgrid.construction import compute_fib_grid
    except ImportError as e:
        raise ImportError(
            f"fibgrid.construction is required to create test grid: {e}. "
            "Install with: pip install fibgrid"
        )
    
    # compute_fib_grid(n) generates 2n+1 points
    # We want approximately n_points total
    n = (n_points - 1) // 2
    
    # Compute standalone Fibonacci lattice
    points, gpi_indices, lon, lat = compute_fib_grid(n)
    
    # Convert to proper types
    gpi = gpi_indices.astype(np.int32)
    lon = lon.astype(np.float32)
    lat = lat.astype(np.float32)
    
    actual_n = len(gpi)
    
    # Create latband clustering
    # Divide latitude range into bands and assign each point to a band
    n_lat_bands = 10  # Number of latitude bands (adjustable)
    lat_bands = np.linspace(np.min(lat), np.max(lat), n_lat_bands + 1)
    
    # Assign each point to a lat band
    cluster_id = np.zeros(actual_n, dtype=np.int32)
    for i in range(actual_n):
        cluster_id[i] = np.searchsorted(lat_bands, lat[i]) - 1
    
    # Sort by cluster_id to get latband sorting order
    # Within each cluster, we also sort by longitude for spatial consistency
    sort_keys = cluster_id * 1000000 + ((lon + 180) * 100000).astype(np.int32)
    latband_sorting = np.argsort(sort_keys).astype(np.int32)
    
    return {
        "gpi": gpi,
        "lon": lon,
        "lat": lat,
        "latband_sorting": latband_sorting,
        "cluster_id": cluster_id
    }


_test_grid_data = None

def get_test_grid_data(n_points=500):
    """
    Get cached test grid data.
    
    Parameters
    ----------
    n_points : int, optional
        Approximate desired number of grid points
        
    Returns
    -------
    dict
        Test grid data with keys 'gpi', 'lon', 'lat', 'latband_sorting', 'cluster_id'
    """
    global _test_grid_data
    if _test_grid_data is None or len(_test_grid_data["gpi"]) < n_points:
        _test_grid_data = create_test_grid(n_points)
    return _test_grid_data
