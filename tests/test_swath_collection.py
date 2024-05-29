#!/usr/bin/env python3

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np
import dask

from pyresample.geometry import SwathDefinition, AreaDefinition
from fibgrid.realization import FibGrid

from pygeogrids.netcdf import load_grid

import ascat.read_native.generate_test_data as gtd

from ascat.read_native.swath_collection import SwathFile
from ascat.read_native.swath_collection import SwathGridFiles

def gen_dummy_swathfiles(dir, sat_name=None):
    if sat_name is not None:
        dir = dir / sat_name
        dir.mkdir(parents=True, exist_ok=True)
    gtd.swath_ds.to_netcdf(dir / "swath.nc")
    gtd.swath_ds_2.to_netcdf(dir / "swath_2.nc")

class TestSwathFile(unittest.TestCase):
    """
    """

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        self.assertEqual(ra.filename, swath_path)
        self.assertIsNone(ra.ds)

        ra_chunked = SwathFile(swath_path, chunks={"locations": 2})
        self.assertEqual(ra_chunked.filename, swath_path)
        self.assertIsNone(ra_chunked.ds)
        self.assertEqual(ra_chunked.chunks, {"locations": 2})

    def test_read(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        ra.read()
        self.assertIsInstance(ra.ds, xr.Dataset)
        self.assertIn("longitude", ra.ds)
        self.assertIn("latitude", ra.ds)
        # self.assertIn("time", ra.ds)
        self.assertIn("obs", ra.ds.dims)

    def test__ensure_obs(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        ds = xr.open_dataset(swath_path)
        ds = ra._ensure_obs(ds)
        self.assertIn("obs", ds.dims)
        self.assertNotIn("time", ds.dims)
        # print(original_dim.values)
        # np.testing.assert_equal(ds["obs"].values, original_dim.values)

    def test__trim_to_gpis(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        ds = xr.open_dataset(swath_path)
        ds = ds.chunk({"obs": 1_000_000})
        valid_gpis = [1100178, 1102762]
        grid = FibGrid(12.5)


        gpi_lookup = np.zeros(grid.gpis.max()+1, dtype=bool)
        gpi_lookup[valid_gpis] = 1
        # trimmed_ds = ra._trim_to_gpis(ds, valid_gpis)
        trimmed_ds = ra._trim_to_gpis(ds, lookup_vector=gpi_lookup)
        self.assertTrue(np.all(np.isin(trimmed_ds["location_id"].values, valid_gpis)))
        new_obs_gpis = trimmed_ds["location_id"].values
        np.testing.assert_array_equal(new_obs_gpis, np.repeat(np.array(valid_gpis), [1, 1]))

    def test__trim_var_range(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        ds = xr.open_dataset(swath_path)
        ds = ra._ensure_obs(ds)
        ds = ds.chunk({"obs": 1_000_000})
        start_dt = np.datetime64("2020-11-15T09:04:50")
        end_dt = np.datetime64("2020-11-15T09:04:52")
        trimmed_ds = ra._trim_var_range(ds, "time", start_dt, end_dt)
        self.assertTrue(np.all(trimmed_ds["time"].values < end_dt))
        self.assertTrue(np.all(trimmed_ds["time"].values >= start_dt))


    def test_merge(self):
        fname1 = self.tempdir_path / "swath.nc"
        fname2 = self.tempdir_path / "swath_2.nc"
        ra1 = SwathFile(fname1)
        ra2 = SwathFile(fname2)
        ds1 = ra1.read()
        ds2 = ra2.read()
        merged = ra1.merge([ra1.ds, ra2.ds])
        self.assertTrue(np.all(merged["location_id"].values == np.concatenate([ds1["location_id"].values, ds2["location_id"].values])))

class TestSwathGridFiles(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()


    def test_extract(self):
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        grid = FibGrid(6.25)
        sf = SwathGridFiles(
            swath_path,
            cls=SwathFile,
            fn_templ="W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H129_C_LIIB_{date}_{placeholder}_{placeholder1}____.nc",
            sf_templ={"year_folder": "{year}"},
            grid=grid,
            fn_read_fmt= lambda timestamp: {
                "date": timestamp.strftime("%Y%m%d*"),
                "sat": "[ABC]",
                "placeholder": "*",
                "placeholder1": "*"
            },
            sf_read_fmt = lambda timestamp:{
                "year_folder": {
                    "year": f"{timestamp.year}"
                },
            },
        )
        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S"
        )
        bbox=(-180, -4, -70, 20)

        merged_ds = sf.extract(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            bbox = bbox,
            date_field_fmt="%Y%m%d%H%M%S"
        )
