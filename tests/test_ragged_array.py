# Copyright (c) 2023, TU Wien, Department of Geodesy and Geoinformation
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

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory
from copy import deepcopy

import numpy as np
from numpy.testing import assert_equal
import xarray as xr

from ascat.read_native.ragged_array_ts import var_order
from ascat.read_native.ragged_array_ts import indexed_to_contiguous
from ascat.read_native.ragged_array_ts import contiguous_to_indexed
from ascat.read_native.ragged_array_ts import dataset_ra_type
from ascat.read_native.ragged_array_ts import set_attributes
from ascat.read_native.ragged_array_ts import create_encoding
from ascat.read_native.ragged_array_ts import merge_netCDFs
from ascat.read_native.ragged_array_ts import RACollection



def data_setup(outdir):
    """
    Setup data for tests
    """
    location_info = {
        "location_id": [1001, 1002, 1003, 1004, 1005],
        "lat": [38.97, 37.75, 48.68, 32.82, 44.44],
        "lon": [-114.30, -119.59, -113.61, -106.27, -110.61],
        "alt": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "location_description": [
            "Great Basin",
            "Yosemite",
            "Glacier",
            "White Sands",
            "Yellowstone",
        ],
    }

    def random_date_range(begin, d_range, n_dates):
        np.random.seed(42)
        return sorted(
            [
                (np.datetime64(begin) + np.random.choice(np.arange(0, d_range))).astype(
                    "datetime64[ns]"
                )
                for i in range(0, n_dates)
            ]
        )

    # data that will be stored in contiguous ragged format
    old_data_contiguous = {
        #    "locationIndex": {1, 1, 1, 0, 0, 3, 3, 3, 3},
        "sat_id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "temp": [24, 23, 23, 31, 32, 35, 34, 34, 35],
        "humidity": [0.44, 0.32, 0.21, 0, 0, 0, 0, 0, 0],
        "location_id": [1002, 1001, 1004],
        "location_description": ["Yosemite", "Great Basin", "White Sands"],
        "time": random_date_range("2021-05-01 00:00:00", 31 * 24 * 3600, 9),
        "row_size": [3, 2, 4],
    }
    old_data_contiguous["lon"] = [
        location_info["lon"][location_info["location_id"].index(s)]
        for s in old_data_contiguous["location_id"]
    ]
    old_data_contiguous["lat"] = [
        location_info["lat"][location_info["location_id"].index(s)]
        for s in old_data_contiguous["location_id"]
    ]
    old_data_contiguous["alt"] = [
        location_info["alt"][location_info["location_id"].index(s)]
        for s in old_data_contiguous["location_id"]
    ]

    ctg_ds_old = xr.Dataset(
        data_vars=dict(
            row_size=(["locations"], old_data_contiguous["row_size"]),
            location_id=(["locations"], old_data_contiguous["location_id"]),
            location_description=(["locations"], old_data_contiguous["location_description"]),
            sat_id=(["obs"], old_data_contiguous["sat_id"]),
            temp=(["obs"], old_data_contiguous["temp"]),
            humidity=(["obs"], old_data_contiguous["humidity"]),
        ),
        coords=dict(
            lon=(["locations"], old_data_contiguous["lon"]),
            lat=(["locations"], old_data_contiguous["lat"]),
            alt=(["locations"], old_data_contiguous["alt"]),
            time=(["obs"], old_data_contiguous["time"]),
        ),
        attrs=dict(
            id="test_contiguous_old.nc",
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            featureType="timeSeries",
        ),
    )

    old_data_indexed = deepcopy(old_data_contiguous)
    del old_data_indexed["row_size"]
    old_data_indexed["locationIndex"] = [0, 0, 0, 1, 1, 2, 2, 2, 2]

    old_data_indexed["lon"] = [
        location_info["lon"][location_info["location_id"].index(s)]
        for s in old_data_indexed["location_id"]
    ]
    old_data_indexed["lat"] = [
        location_info["lat"][location_info["location_id"].index(s)]
        for s in old_data_indexed["location_id"]
    ]
    old_data_indexed["alt"] = [
        location_info["alt"][location_info["location_id"].index(s)]
        for s in old_data_indexed["location_id"]
    ]
    idx_ds_old = xr.Dataset(
        data_vars=dict(
            locationIndex=(["obs"], old_data_indexed["locationIndex"]),
            location_id=(["locations"], old_data_indexed["location_id"]),
            location_description=(
                ["locations"],
                old_data_indexed["location_description"],
            ),
            sat_id=(["obs"], old_data_indexed["sat_id"]),
            temp=(["obs"], old_data_indexed["temp"]),
            humidity=(["obs"], old_data_indexed["humidity"]),
        ),
        coords=dict(
            lon=(["locations"], old_data_indexed["lon"]),
            lat=(["locations"], old_data_indexed["lat"]),
            alt=(["locations"], old_data_indexed["alt"]),
            time=(["obs"], old_data_indexed["time"]),
        ),
        attrs=dict(
            id="test_indexed_old.nc",
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            featureType="timeSeries",
        ),
    )

    # new data we want to add on top, which is in indexed ragged format
    new_data_indexed = {
        "locationIndex": [1, 2, 1, 1, 2, 0, 0, 0, 0],
        "temp": [31, 19, 32, 32, 22, 18, 16, 17, 17],
        "humidity": [0, 0.33, 0, 0, 0.21, 0.97, 0.86, 0.22, 0.31],
        "location_id": [1003, 1004, 1005],
        "sat_id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "location_description": ["Glacier", "White Sands", "Yellowstone"],
        "time": random_date_range("2021-06-01 00:00:00", 30 * 24 * 3600, 9),
    }
    # new_data_indexed = {
    #     "locationIndex": [1, 0, 1, 1, 1, 0, 0, 0, 0],
    #     "temp": [31, 19, 32, 32, 22, 18, 16, 17, 17],
    #     "humidity": [0, 0.33, 0, 0, 0.21, 0.97, 0.86, 0.22, 0.31],
    #     "location_id": [1003, 1004],
    #     "sat_id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     "location_description": ["Glacier", "White Sands"],
    #     "time": random_date_range("2021-06-01 00:00:00", 30 * 24 * 3600, 9),
    # }
    new_data_indexed["lon"] = [
        location_info["lon"][location_info["location_id"].index(s)]
        for s in new_data_indexed["location_id"]
    ]
    new_data_indexed["lat"] = [
        location_info["lat"][location_info["location_id"].index(s)]
        for s in new_data_indexed["location_id"]
    ]
    new_data_indexed["alt"] = [
        location_info["alt"][location_info["location_id"].index(s)]
        for s in new_data_indexed["location_id"]
    ]

    idx_ds_new = xr.Dataset(
        data_vars=dict(
            locationIndex=(["obs"], new_data_indexed["locationIndex"]),
            location_id=(["locations"], new_data_indexed["location_id"]),
            location_description=(
                ["locations"],
                new_data_indexed["location_description"],
            ),
            sat_id=(["obs"], new_data_indexed["sat_id"]),
            temp=(["obs"], new_data_indexed["temp"]),
            humidity=(["obs"], new_data_indexed["humidity"]),
        ),
        coords=dict(
            lon=(["locations"], new_data_indexed["lon"]),
            lat=(["locations"], new_data_indexed["lat"]),
            alt=(["locations"], new_data_indexed["alt"]),
            time=(["obs"], new_data_indexed["time"]),
        ),
        attrs=dict(
            id="test_indexed_new.nc",
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            featureType="timeSeries",
        ),
    )

    # data that will be stored in contiguous ragged format but with some different satellites and timestamps
    original_time_range = random_date_range("2021-05-01 00:00:00", 31 * 24 * 3600, 9)
    new_time_range = [i + np.timedelta64(120, "s") for i in original_time_range]
    new_data_contiguous = {
        #    "locationIndex": {1, 1, 1, 0, 0, 3, 3, 3, 3},
        "sat_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        "temp": [24, 23, 23, 31, 32, 35, 34, 34, 35],
        "humidity": [0.44, 0.32, 0.21, 0, 0, 0, 0, 0, 0],
        "location_id": [1002, 1001, 1004],
        "location_description": ["Yosemite", "Great Basin", "White Sands"],
        "time": new_time_range,
        "row_size": [3, 2, 4],
    }
    new_data_contiguous["lon"] = [
        location_info["lon"][location_info["location_id"].index(s)]
        for s in new_data_contiguous["location_id"]
    ]
    new_data_contiguous["lat"] = [
        location_info["lat"][location_info["location_id"].index(s)]
        for s in new_data_contiguous["location_id"]
    ]
    new_data_contiguous["alt"] = [
        location_info["alt"][location_info["location_id"].index(s)]
        for s in new_data_contiguous["location_id"]
    ]

    ctg_ds_new = xr.Dataset(
        data_vars=dict(
            row_size=(["locations"], new_data_contiguous["row_size"]),
            location_id=(["locations"], new_data_contiguous["location_id"]),
            location_description=(["locations"], new_data_contiguous["location_description"]),
            sat_id=(["obs"], new_data_contiguous["sat_id"]),
            temp=(["obs"], new_data_contiguous["temp"]),
            humidity=(["obs"], new_data_contiguous["humidity"]),
        ),
        coords=dict(
            lon=(["locations"], new_data_contiguous["lon"]),
            lat=(["locations"], new_data_contiguous["lat"]),
            alt=(["locations"], new_data_contiguous["alt"]),
            time=(["obs"], new_data_contiguous["time"]),
        ),
        attrs=dict(
            id="test_contiguous_new.nc",
            date_created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            featureType="timeSeries",
        ),
    )

    ctg_ds_old.to_netcdf(outdir / "contiguous_RA_old.nc")
    idx_ds_old.to_netcdf(outdir / "indexed_RA_old.nc")
    idx_ds_new.to_netcdf(outdir / "indexed_RA_new.nc")
    ctg_ds_new.to_netcdf(outdir / "contiguous_RA_new.nc")
    return ctg_ds_old, idx_ds_old, idx_ds_new, ctg_ds_new


class TestHelpers(unittest.TestCase):
    """
    Test the helper functions
    """
    def assertDictNumpyEqual(self, d1, d2):
        try:
            np.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.tmpdir = Path(self.temporary_directory.name)
        data_setup(self.tmpdir)
        self.ctg_old = xr.open_dataset(self.tmpdir / "contiguous_RA_old.nc")
        self.idx_new = xr.open_dataset(self.tmpdir / "indexed_RA_new.nc")
        self.idx_old = xr.open_dataset(self.tmpdir / "indexed_RA_old.nc")
        self.ctg_new = xr.open_dataset(self.tmpdir / "contiguous_RA_new.nc")

    def tearDown(self):
        self.temporary_directory.cleanup()
        self.ctg_old.close()
        self.idx_new.close()
        self.idx_old.close()
        self.ctg_new.close()

    def test_var_order(self):
        """
        Test var_order
        """
        # print(list(self.ctg_old.variables))
        ctg_expected = [
            "row_size",
            "lon",
            "lat",
            "alt",
            "location_id",
            "location_description",
            "time",
            "sat_id",
            "temp",
            "humidity",
        ]
        idx_expected = ctg_expected.copy()
        idx_expected.remove("row_size")
        idx_expected.insert(0, "locationIndex")

        self.assertEqual(list(var_order(self.ctg_old).variables), ctg_expected)
        self.assertEqual(list(var_order(self.idx_new).variables), idx_expected)
        self.assertEqual(list(var_order(self.idx_old).variables), idx_expected)

    def test_dataset_ra_type(self):
        """
        Test dataset_ra_type
        """
        self.assertEqual(dataset_ra_type(self.ctg_old), "contiguous")
        self.assertEqual(dataset_ra_type(self.idx_new), "indexed")
        self.assertEqual(dataset_ra_type(self.idx_old), "indexed")
        neither = self.ctg_old.copy()
        neither = neither.drop_vars(["row_size"])
        self.assertRaises(ValueError, dataset_ra_type, neither)
        neither.close()

    def test_set_attributes(self):
        """
        Test set_attributes
        """
        expected = {
            "row_size": {
                "long_name": "number of observations at this location",
                "sample_dimension": "obs",
            },
            "locationIndex": {
                "long_name": "which location this observation is for",
                "sample_dimension": "locations",
            },
            "lon": {
                "standard_name": "longitude",
                "long_name": "location longitude",
                "units": "degrees_east",
                "valid_range": np.array([-180, 180], dtype=float),
            },
            "lat": {
                "standard_name": "latitude",
                "long_name": "location latitude",
                "units": "degrees_north",
                "valid_range": np.array([-90, 90], dtype=float),
            },
            "alt": {
                "standard_name": "height",
                "long_name": "vertical distance above the surface",
                "units": "m",
                "positive": "up",
                "axis": "Z",
            },
            "time": {
                "standard_name": "time",
                "long_name": "time of measurement",
            },
            "location_id": {
            },
            "location_description": {
            },
        }

        for ds in [self.ctg_old, self.idx_new, self.idx_old]:
            # test that setting attrs works with no extra attrs
            with set_attributes(ds.copy()) as setted:
                for var in setted.variables:
                    if var in expected:
                        self.assertDictNumpyEqual(setted[var].attrs, expected[var])
                    else:
                        self.assertDictNumpyEqual(setted[var].attrs, {})

            # test that setting attrs works with extra attrs
            extra_attrs = {
                "temp": {"foo": "bar"},
                "time": {"spam": "eggs"},
            }

            with set_attributes(ds.copy(), extra_attrs) as setted:
                for var in setted.variables:
                    if var in extra_attrs:
                        self.assertDictNumpyEqual(setted[var].attrs, extra_attrs[var])
                    elif var in expected:
                        self.assertDictNumpyEqual(setted[var].attrs, expected[var])
                    else:
                        self.assertDictNumpyEqual(setted[var].attrs, {})

    def test_create_encoding(self):
        """
        Test create_encoding
        """
        expected = {
            # we expect int32 dtypes below rather than int64 because no dtypes were
            # specified when actually writing the test data to file in data_setup.
            # If they were (using set_encoding), then the dtypes would be int64.
            "row_size": {
                "dtype": np.dtype("int32"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            },
            "locationIndex": {
                "dtype": np.dtype("int32"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            },
            "lon": {
                "dtype": "float32",
                "_FillValue": None,
                "zlib": True,
                "complevel": 4,
            },
            "lat": {
                "dtype": "float32",
                "_FillValue": None,
                "zlib": True,
                "complevel": 4,
            },
            "alt": {
                "dtype": "float32",
                "_FillValue": None,
                "zlib": True,
                "complevel": 4,
            },
            "location_id": {
                "dtype": np.dtype("int32"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            },
            "time": {
                "dtype": "float64",
                "units": "days since 1900-01-01 00:00:00",
                "_FillValue": None,
                "zlib": True,
                "complevel": 4,
            },
            "location_description": {
                "dtype": np.dtype("O"),
                "zlib": False,
                "complevel": 4,
                "_FillValue": None,
            },
            "sat_id": {
                "dtype": np.dtype("int32"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            },
            "temp": {
                "dtype": np.dtype("int32"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            },
            "humidity": {
                "dtype": np.dtype("float64"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": None,
            },
        }
        for ds in [self.ctg_old, self.idx_new, self.idx_old]:
            ds_encoding = create_encoding(ds)
            for var in ds.variables:
                try:
                    self.assertDictNumpyEqual(ds_encoding[var], expected[var])
                except:
                    print(var)
                    raise


    # @classmethod
    # def tearDownClass(cls):
    #     cls.ctg_old.close()
    #     cls.idx_new.close()
    #     cls.idx_old.close()


class TestConversion(unittest.TestCase):
    """
    Test the conversion functions
    """

    def assertNanEqual(self, d1, d2):
        try:
            np.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    def setUp(self):
        """
        Set up the test data
        """
        self.temporary_directory = TemporaryDirectory()
        self.tmpdir = Path(self.temporary_directory.name)
        data_setup(self.tmpdir)
        self.ctg_old = xr.open_dataset(self.tmpdir / "contiguous_RA_old.nc")
        self.idx_new = xr.open_dataset(self.tmpdir / "indexed_RA_new.nc")
        self.idx_old = xr.open_dataset(self.tmpdir / "indexed_RA_old.nc")
        self.ctg_new = xr.open_dataset(self.tmpdir / "contiguous_RA_new.nc")

    def tearDown(self):
        self.temporary_directory.cleanup()
        self.ctg_old.close()
        self.idx_new.close()
        self.idx_old.close()
        self.ctg_new.close()

    def test_indexed_to_contiguous(self):
        """
        Test conversion of indexed ragged array to contiguous ragged array
        """
        converted = indexed_to_contiguous(self.idx_old)
        for var in converted.variables:
            self.assertNanEqual(converted[var].values, self.ctg_old[var].values)

    def test_contiguous_to_indexed(self):
        """
        Test conversion of contiguous ragged array to indexed ragged array
        """
        converted = contiguous_to_indexed(self.ctg_old)
        for var in converted.variables:
            self.assertNanEqual(converted[var].values, self.idx_old[var].values)

    # @classmethod
    # def tearDownClass(cls):
    #     cls.ctg_old.close()
    #     cls.idx_new.close()
    #     cls.idx_old.close()

class TestMerge(unittest.TestCase):
    """
    Test the merge function
    """

    def assertNanEqual(self, d1, d2):
        try:
            np.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    # @classmethod
    # def setUpClass(cls):
    #     cls.ctg_old_fname = self.tmpdir / "contiguous_RA_old.nc"
    #     cls.idx_new_fname = self.tmpdir / "indexed_RA_new.nc"
    #     cls.idx_old_fname = self.tmpdir / "indexed_RA_old.nc"
    #     cls.ctg_new_fname = self.tmpdir / "contiguous_RA_new.nc"

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.tmpdir = Path(self.temporary_directory.name)
        self.ctg_ds_old, self.idx_ds_old, self.idx_ds_new, self.ctg_ds_new = data_setup(self.tmpdir)
        self.ctg_old_fname = self.tmpdir / "contiguous_RA_old.nc"
        self.idx_new_fname = self.tmpdir / "indexed_RA_new.nc"
        self.idx_old_fname = self.tmpdir / "indexed_RA_old.nc"
        self.ctg_new_fname = self.tmpdir / "contiguous_RA_new.nc"

    def tearDown(self):
        self.temporary_directory.cleanup()
        self.ctg_ds_old.close()
        self.idx_ds_old.close()
        self.idx_ds_new.close()
        self.ctg_ds_new.close()

    def test_merge_netCDFs(self):
        """
        Test merging of netCDF files
        """
        # Test merging indexed with indexed
        # merged = merge_netCDFs([self.idx_old_fname, self.idx_new_fname])
        # with xr.open_mfdataset(
        #     [self.idx_old_fname, self.idx_new_fname],
        #     concat_dim="locations",
        #     combine="nested",
        #     preprocess=lambda ds: ds[[var for var in ds.variables if "locations" in ds[var].dims]],
        # ) as merged:
        #     print(merged)
        merged = RACollection([self.idx_old_fname, self.idx_new_fname]).merge()
        expected = {
            "row_size": np.array([2, 3, 4, 7, 2]),
            "lon": np.array([-114.3, -119.59, -113.61, -106.27, -110.61]),
            "lat": np.array([38.97, 37.75, 48.68, 32.82, 44.44]),
            "alt": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            "location_id": np.array([1001, 1002, 1003, 1004, 1005]),
            "location_description": np.array(["Great Basin", "Yosemite", "Glacier", "White Sands", "Yellowstone"]),
            "sat_id": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            "temp": np.array([31, 32, 24, 23, 23, 18, 16, 17, 17, 35, 34, 34, 35, 31, 32, 32, 19, 22]),
            "humidity": np.array([0.0, 0.0, 0.44, 0.32, 0.21, 0.97, 0.86, 0.22, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33, 0.21]),
        }
        # print(np.argsort(merged["time"].values))
        time_order = np.array([2,  3,  4,  0,  1,  9, 10, 11, 12, 13, 16, 14, 15, 17,  5,  6,  7,  8])
        expected_times = np.concatenate((self.idx_ds_old["time"].values, self.idx_ds_new["time"].values))
        for var in expected:
            self.assertNanEqual(merged[var].values, expected[var])
        self.assertNanEqual(merged["time"].values[time_order], expected_times)
        merged.close()

        # Test merging indexed with contiguous
        merged = merge_netCDFs([self.ctg_old_fname, self.idx_new_fname])
        expected = {
            "row_size": np.array([2, 3, 4, 7, 2]),
            "lon": np.array([-114.3, -119.59, -113.61, -106.27, -110.61]),
            "lat": np.array([38.97, 37.75, 48.68, 32.82, 44.44]),
            "alt": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            "location_id": np.array([1001, 1002, 1003, 1004, 1005]),
            "location_description": np.array(["Great Basin", "Yosemite", "Glacier", "White Sands", "Yellowstone"]),
            "sat_id": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            "temp": np.array([31, 32, 24, 23, 23, 18, 16, 17, 17, 35, 34, 34, 35, 31, 32, 32, 19, 22]),
            "humidity": np.array([0.0, 0.0, 0.44, 0.32, 0.21, 0.97, 0.86, 0.22, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33, 0.21]),
        }
        time_order = np.array([2, 3, 4, 0, 1, 9, 10, 11, 12, 13, 16, 14, 15, 17, 5, 6, 7, 8])
        expected_times = np.concatenate((self.idx_ds_old["time"].values, self.idx_ds_new["time"].values))
        for var in expected:
            self.assertNanEqual(merged[var].values, expected[var])
        self.assertNanEqual(merged["time"].values[time_order], expected_times)
        merged.close()

        # Test merging contiguous with contiguous with same data
        merged = merge_netCDFs([self.ctg_old_fname, self.ctg_old_fname])
        # for var in merged.variables:
        #     print(f"\"{var}\": np.array([{', '.join(merged[var].values.astype(str))}]),")

        expected = {
            "row_size": np.array([2, 3, 4]),
            "lon": np.array([-114.3, -119.59, -106.27]),
            "lat": np.array([38.97, 37.75, 32.82]),
            "alt": np.array([np.nan, np.nan, np.nan]),
            "location_id": np.array([1001, 1002, 1004]),
            "location_description": np.array(["Great Basin", "Yosemite", "White Sands"]),
            "sat_id": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            "temp": np.array([31, 32, 24, 23, 23, 35, 34, 34, 35]),
            "humidity": np.array([0.0, 0.0, 0.44, 0.32, 0.21, 0.0, 0.0, 0.0, 0.0]),
        }
        time_order = np.array([2, 3, 4, 0, 1, 5, 6, 7, 8])
        expected_times = self.ctg_ds_old["time"].values
        for var in expected:
            self.assertNanEqual(merged[var].values, expected[var])
        self.assertNanEqual(merged["time"].values[time_order], expected_times)
        merged.close()

        # Test merging contiguous with contiguous with different satellites
        # Not passing a dupe_window here means that only the values from different
        # satellites should be included, as the others will be filtered out as duplicates
        merged = merge_netCDFs([self.ctg_old_fname, self.ctg_new_fname])
        # for var in merged.variables:
        #     print(f"\"{var}\": np.array([{', '.join(merged[var].values.astype(str))}]),")

        # print(np.argsort(merged["time"].values))
        expected = {
            "row_size": np.array([2, 3, 8]),
            "lon": np.array([-114.3, -119.59, -106.27]),
            "lat": np.array([38.97, 37.75, 32.82]),
            "alt": np.array([np.nan, np.nan, np.nan]),
            "location_id": np.array([1001, 1002, 1004]),
            "location_description": np.array(["Great Basin", "Yosemite", "White Sands"]),
            "sat_id": np.array([1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2]),
            "temp": np.array([31, 32, 24, 23, 23, 35, 35, 34, 34, 34, 34, 35, 35]),
            "humidity": np.array([0.0, 0.0, 0.44, 0.32, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        time_order = np.array([2, 3, 4, 0, 1, 5, 6, 7, 8, 9, 10, 11, 12])
        expected_times = np.sort(np.concatenate((self.ctg_ds_old["time"].values,
                                         self.ctg_ds_new["time"].values[-4:])))
        for var in expected:
            self.assertNanEqual(merged[var].values, expected[var])
        self.assertNanEqual(merged["time"].values[time_order], expected_times)
        merged.close()

        # Test merging contiguous with contiguous with different satellites
        # Passing a smaller dupe_window here will include more values from the
        # satellite in the first dataset
        merged = merge_netCDFs([self.ctg_old_fname, self.ctg_new_fname],
                               dupe_window=np.timedelta64(1, "m"))
        # for var in merged.variables:
        #     print(f"\"{var}\": np.array([{', '.join(merged[var].values.astype(str))}]),")

        expected = {
            "row_size": np.array([4, 6, 8]),
            "lon": np.array([-114.3, -119.59, -106.27]),
            "lat": np.array([38.97, 37.75, 32.82]),
            "alt": np.array([np.nan, np.nan, np.nan]),
            "location_id": np.array([1001, 1002, 1004]),
            "location_description": np.array(["Great Basin", "Yosemite", "White Sands"]),
            "sat_id": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2]),
            "temp": np.array([31, 31, 32, 32, 24, 24, 23, 23, 23, 23, 35, 35, 34, 34, 34, 34, 35, 35]),
            "humidity": np.array([0.0, 0.0, 0.0, 0.0, 0.44, 0.44, 0.32, 0.32, 0.21, 0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        time_order = np.array([4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17])
        expected_times = np.sort(np.concatenate((self.ctg_ds_old["time"].values,
                                         self.ctg_ds_new["time"].values)))
        for var in expected:
            self.assertNanEqual(merged[var].values, expected[var])
        self.assertNanEqual(merged["time"].values[time_order], expected_times)
        merged.close()

if __name__ == "__main__":
    unittest.main()
