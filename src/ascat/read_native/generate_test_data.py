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
                [142.937536, 143.302272, 143.038352, 142.774416], dtype=np.float64
            ),
        ),
        "latitude": (
            "obs",
            np.array([42.176548, 42.279804, 42.251248, 42.222704], dtype=np.float64),
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
