#!/usr/bin/env python3

import xarray as xr
import numpy as np

contiguous_ragged_ds_2588 = xr.Dataset(
    {
        "lon": ("locations",
                np.array([175.80013, 175.37308, 179.1304 , 179.82138, 178.70335],
                         dtype=np.float32)),
        "lat": ("locations",
                np.array([70.00758 , 70.04549 , 70.65371 , 70.67787 , 70.692825],
                         dtype=np.float32)),
        "alt": ("locations",
                np.array([np.nan, np.nan, np.nan, np.nan, np.nan],
                         dtype=np.float32)
                ),
        "location_id": ("locations",
                        np.array([1549346, 1549723, 1555679, 1555912, 1556056],
                                 dtype=np.int64)),
        "row_size": ("locations",
                        np.array([2, 1, 2, 4, 1], dtype=np.int32)),
        "time": ("time",
                 np.array([np.datetime64('2020-01-01T00:00:00'),
                           np.datetime64('2020-01-01T00:00:01'),
                           np.datetime64('2020-01-01T00:00:02'),
                           np.datetime64('2020-01-01T00:00:03'),
                           np.datetime64('2020-01-01T00:00:04'),
                           np.datetime64('2020-01-01T00:00:05'),
                           np.datetime64('2020-01-01T00:00:06'),
                           np.datetime64('2020-01-01T00:00:07'),
                           np.datetime64('2020-01-01T00:00:08'),
                           np.datetime64('2020-01-01T00:00:09')],
                          dtype='datetime64[ns]')),
    }
)

indexed_ragged_ds_2588 = xr.Dataset(
    {
        "lon": ("locations",
                np.array([175.80013, 175.37308, 179.1304 , 179.82138, 178.70335],
                         dtype=np.float32)),
        "lat": ("locations",
                np.array([70.00758 , 70.04549 , 70.65371 , 70.67787 , 70.692825],
                         dtype=np.float32)),
        "alt": ("locations",
                np.array([np.nan, np.nan, np.nan, np.nan, np.nan],
                         dtype=np.float32)
                ),
        "location_id": ("locations",
                        np.array([1549346, 1549723, 1555679, 1555912, 1556056],
                                 dtype=np.int64)),
        "locationIndex": ("time",
                          np.array([0, 0, 1, 2, 2, 3, 3, 3, 3, 4],
                                   dtype=np.int32)),
        "time": ("time",
                 np.array([np.datetime64('2020-01-01T00:00:00'),
                           np.datetime64('2020-01-01T00:00:01'),
                           np.datetime64('2020-01-01T00:00:02'),
                           np.datetime64('2020-01-01T00:00:03'),
                           np.datetime64('2020-01-01T00:00:04'),
                           np.datetime64('2020-01-01T00:00:05'),
                           np.datetime64('2020-01-01T00:00:06'),
                           np.datetime64('2020-01-01T00:00:07'),
                           np.datetime64('2020-01-01T00:00:08'),
                           np.datetime64('2020-01-01T00:00:09')],
                          dtype='datetime64[ns]')),
    }
)

contiguous_ragged_ds_2587 = xr.Dataset(
    {
        "lon": ("locations",
                np.array([175.88971, 177.6987 , 179.5077 , 176.58069, 178.38968],
                         dtype=np.float32)),
        "lat": ("locations",
                np.array([65.00168 , 65.00892 , 65.01617 , 65.020645, 65.02789 ],
                         dtype=np.float32)),
        "alt": ("locations",
                np.array([np.nan, np.nan, np.nan, np.nan, np.nan],
                         dtype=np.float32)
                ),
        "location_id": ("locations",
                        np.array([1493629, 1493718, 1493807, 1493862, 1493951],
                                 dtype=np.int64)),
        "row_size": ("locations",
                        np.array([3, 1, 1, 3, 2], dtype=np.int32)),
        "time": ("time",
                 np.array([np.datetime64('2020-01-01T00:00:01'),
                           np.datetime64('2020-01-01T00:00:03'),
                           np.datetime64('2020-01-01T00:00:04'),
                           np.datetime64('2020-01-01T00:00:05'),
                           np.datetime64('2020-01-01T00:00:07'),
                           np.datetime64('2020-01-01T00:00:09'),
                           np.datetime64('2020-01-01T00:00:10'),
                           np.datetime64('2020-01-01T00:00:11'),
                           np.datetime64('2020-01-01T00:00:14'),
                           np.datetime64('2020-01-01T00:00:18')],
                          dtype='datetime64[ns]')),
    }
)

indexed_ragged_ds_2587 = xr.Dataset(
    {
        "lon": ("locations",
                np.array([175.88971, 177.6987 , 179.5077 , 176.58069, 178.38968],
                         dtype=np.float32)),
        "lat": ("locations",
                np.array([65.00168 , 65.00892 , 65.01617 , 65.020645, 65.02789 ],
                         dtype=np.float32)),
        "alt": ("locations",
                np.array([np.nan, np.nan, np.nan, np.nan, np.nan],
                         dtype=np.float32)
                ),
        "location_id": ("locations",
                        np.array([1493629, 1493718, 1493807, 1493862, 1493951],
                                 dtype=np.int64)),
        "locationIndex": ("time",
                          np.array([0, 0, 0, 1, 2, 3, 3, 3, 4, 4],
                                   dtype=np.int32)),
        "time": ("time",
                 np.array([np.datetime64('2020-01-01T00:00:01'),
                           np.datetime64('2020-01-01T00:00:03'),
                           np.datetime64('2020-01-01T00:00:04'),
                           np.datetime64('2020-01-01T00:00:05'),
                           np.datetime64('2020-01-01T00:00:07'),
                           np.datetime64('2020-01-01T00:00:09'),
                           np.datetime64('2020-01-01T00:00:10'),
                           np.datetime64('2020-01-01T00:00:11'),
                           np.datetime64('2020-01-01T00:00:14'),
                           np.datetime64('2020-01-01T00:00:18')],
                          dtype='datetime64[ns]')),
    }
)
