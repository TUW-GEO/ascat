#!/usr/bin/env python3

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory
from time import time

import xarray as xr
import numpy as np

import ascat.read_native.generate_test_data as gtd

from ascat.swath import Swath
from ascat.swath import SwathGridFiles
from ascat.product_info import AscatH129Swath
from get_path import get_testdata_path

TESTDATA_PATH = get_testdata_path()

def gen_dummy_swathfiles(directory, sat_name=None):
    if sat_name is not None:
        directory = directory / sat_name
        directory.mkdir(parents=True, exist_ok=True)
    gtd.swath_ds.to_netcdf(directory / "swath.nc")
    gtd.swath_ds_2.to_netcdf(directory / "swath_2.nc")


class TestSwath(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        self.real_swaths_path = Path(
            TESTDATA_PATH / "hsaf/h129/swaths/metop_a/2021/01"
        )
        gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = Swath(swath_path)
        self.assertEqual(ra.filenames[0], swath_path)

        # ra_chunked = SwathFile(swath_path, chunks={"locations": 2})
        # self.assertEqual(ra_chunked.filename, swath_path)
        # self.assertIsNone(ra_chunked.ds)
        # self.assertEqual(ra_chunked.chunks, {"locations": 2})

    def test_read(self):
        # Basic open of dummy file
        swath_path = self.tempdir_path / "swath.nc"
        ra = Swath(swath_path)
        ds = ra.read()
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("longitude", ds)
        self.assertIn("latitude", ds)
        # self.assertIn("time", ra.ds)
        self.assertIn("obs", ds.dims)

        # Open of multiple real files
        swath_paths = list(self.real_swaths_path.glob("*.nc"))
        ra = Swath(swath_paths)
        ds1 = ra.read().load()
        ds2 = ra.read(parallel=True).load()

        reconstructed = xr.open_mfdataset(
            swath_paths,
            concat_dim="obs",
            combine="nested",
            combine_attrs="drop_conflicts",
        ).load()

        ds3 = [data for data in ra.iter_read_nbytes(1_000_000_000_000)][0].load()

        xr.testing.assert_identical(ds1, reconstructed)
        xr.testing.assert_identical(ds2, reconstructed)
        xr.testing.assert_identical(ds3, reconstructed)

    def test__ensure_obs(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = Swath(swath_path)
        ds = xr.open_dataset(swath_path)
        ds = ra._ensure_obs(ds)
        self.assertIn("obs", ds.dims)
        self.assertNotIn("time", ds.dims)
        # print(original_dim.values)
        # np.testing.assert_equal(ds["obs"].values, original_dim.values)

    def test_merge(self):
        fname1 = self.tempdir_path / "swath.nc"
        fname2 = self.tempdir_path / "swath_2.nc"
        ra1 = Swath(fname1)
        ra2 = Swath(fname2)
        ds1 = ra1.read()
        ds2 = ra2.read()
        merged = ra1.merge([ds1, ds2])
        self.assertTrue(
            np.all(
                merged["location_id"].values
                == np.concatenate(
                    [ds1["location_id"].values, ds2["location_id"].values]
                )
            )
        )

    def test__nbytes(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = Swath(swath_path)
        ds = ra.read()
        nbytes = ra._nbytes(ds)
        self.assertGreater(nbytes, 0)

    def test_iter_read_nbytes(self):
        # TODO adapt and move to test_file_handling.TestFilenames

        fname1 = self.tempdir_path / "swath.nc"
        fname2 = self.tempdir_path / "swath_2.nc"
        ra = Swath([fname1, fname2])

        iterations = list(ra.iter_read_nbytes(max_nbytes=0))
        assert len(iterations) == 2

        ds1, ds2 = iterations
        total_size = ra._nbytes(ds1) + ra._nbytes(ds2)

        iterations = list(ra.iter_read_nbytes(max_nbytes=total_size))
        assert len(iterations) == 1

        iterations = list(ra.iter_read_nbytes(max_nbytes=total_size - 1))
        assert len(iterations) == 2


class TestSwathGridFiles(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):

        def _fn_read_fmt(timestamp, sat="[ABC]"):
            sat = sat.upper()
            return {
                "date": timestamp.strftime("%Y%m%d*"),
                "sat": sat,
                "placeholder": "*",
                "placeholder1": "*",
            }

        def _sf_read_fmt(timestamp, sat="[abc]"):
            sat = sat.lower()
            output_fmt = {
                "satellite_folder": {"satellite": f"metop_{sat}"},
                "year_folder": {"year": f"{timestamp.year}"},
                "month_folder": {"month": f"{timestamp.month:02d}"},
            }
            # if sat is not None:
            #     output_fmt["sat_folder"] = {
            #         "sat": f"metop_{sat.lower()}"
            #     }

            return output_fmt

        # we can create a SwathGridFiles object that points directly to a directory
        # and read the files within it, without passing a "sat" argument to
        # sf.search_period().
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths/metop_a/2021/01"

        sf = SwathGridFiles(
            swath_path,
            fn_templ="W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H129_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc",
            sf_templ={"satellite_folder": "metop_[abc]", "year_folder": "{year}", "month_folder": "{month}"},
            date_field_fmt="%Y%m%d%H%M%S",
            grid_name="fibgrid_6.25",
            fn_read_fmt=_fn_read_fmt,
            sf_read_fmt=_sf_read_fmt,
        )

        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S",
            # sat="a"
        )
        self.assertGreater(len(files), 0)

        # We can also set the path to the root of the product directory and
        # optionally specify "sat" to search_period() to filter the files by
        # satellite.
        # This would be a regex, so if we just wanted metop b and c we could
        # pass sat="[bc]".
        # The default value is "[abc]" which will take all three (or whatever
        # is available)
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths"
        sf = SwathGridFiles(
            swath_path,
            fn_templ="W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H129_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc",
            sf_templ={"satellite_folder": "metop_[abc]", "year_folder": "{year}", "month_folder": "{month}"},
            date_field_fmt="%Y%m%d%H%M%S",
            grid_name="fibgrid_6.25",
            fn_read_fmt=_fn_read_fmt,
            sf_read_fmt=_sf_read_fmt,
        )

        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S",
            sat="a",
        )
        self.assertGreater(len(files), 0)

    def test_from_product_id(self):
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        files = sf.search_period(
            datetime(2021, 1, 15), datetime(2021, 1, 30), date_field_fmt="%Y%m%d%H%M%S"
        )
        print(files)
        self.assertGreater(len(files), 0)

    def test_from_io_class(self):
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_class(swath_path, AscatH129Swath)
        files = sf.search_period(
            datetime(2021, 1, 15), datetime(2021, 1, 30), date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)

    def test_read(self):
        # test read
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        files = sf.search_period(
            datetime(2021, 1, 15), datetime(2021, 1, 30), date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)
        bbox = (-180, -4, -70, 20)

        merged_ds = sf.read(
            (datetime(2021, 1, 15), datetime(2021, 1, 30)),
            bbox=bbox,
        )
        merged_ds.load()

        self.assertLess(merged_ds.time.max().values, np.datetime64(datetime(2021, 1, 30)))
        self.assertGreater(merged_ds.time.min().values, np.datetime64(datetime(2021, 1, 15)))
        self.assertLess(merged_ds.latitude.max().values, bbox[1])
        self.assertGreater(merged_ds.latitude.min().values, bbox[0])
        self.assertLess(merged_ds.longitude.max().values, bbox[3])
        self.assertGreater(merged_ds.longitude.min().values, bbox[2])

        # test extract from main folder
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths/metop_a/2021/01"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        files = sf.search_period(
            datetime(2021, 1, 15), datetime(2021, 1, 30), date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)
        bbox = (-180, -4, -70, 20)

        merged_ds = sf.read(
            (datetime(2021, 1, 15), datetime(2021, 1, 30)),
            bbox=bbox,
        )
        merged_ds.load()

        self.assertLess(merged_ds.time.max().values, np.datetime64(datetime(2021, 1, 30)))
        self.assertGreater(merged_ds.time.min().values, np.datetime64(datetime(2021, 1, 15)))
        self.assertLess(merged_ds.latitude.max().values, bbox[1])
        self.assertGreater(merged_ds.latitude.min().values, bbox[0])
        self.assertLess(merged_ds.longitude.max().values, bbox[3])
        self.assertGreater(merged_ds.longitude.min().values, bbox[2])


    # def test__trim_to_gpis(self):
    #     swath_path = self.tempdir_path / "swath.nc"
    #     ra = Swath(swath_path)
    #     ds = xr.open_dataset(swath_path)
    #     ds = ds.chunk({"obs": 1_000_000})
    #     valid_gpis = [1100178, 1102762]
    #     grid = FibGrid(12.5)

    #     gpi_lookup = np.zeros(grid.gpis.max()+1, dtype=bool)
    #     gpi_lookup[valid_gpis] = 1
    #     # trimmed_ds = ra._trim_to_gpis(ds, valid_gpis)
    #     trimmed_ds = ra._trim_to_gpis(ds, lookup_vector=gpi_lookup)
    #     self.assertTrue(np.all(np.isin(trimmed_ds["location_id"].values, valid_gpis)))
    #     new_obs_gpis = trimmed_ds["location_id"].values
    #     np.testing.assert_array_equal(new_obs_gpis, np.repeat(np.array(valid_gpis), [1, 1]))

    # def test__trim_var_range(self):
    #     swath_path = self.tempdir_path / "swath.nc"
    #     ra = Swath(swath_path)
    #     ds = xr.open_dataset(swath_path)
    #     ds = ra._ensure_obs(ds)
    #     ds = ds.chunk({"obs": 1_000_000})
    #     start_dt = np.datetime64("2020-11-15T09:04:50")
    #     end_dt = np.datetime64("2020-11-15T09:04:52")
    #     trimmed_ds = ra._trim_var_range(ds, "time", start_dt, end_dt)
    #     self.assertTrue(np.all(trimmed_ds["time"].values < end_dt))
    #     self.assertTrue(np.all(trimmed_ds["time"].values >= start_dt))

    def test_stack_to_cell_files(self):
        swath_path = TESTDATA_PATH / "hsaf/h129/swaths/"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        out_dir = self.tempdir_path / "cells_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cells_to_test = [2587, 2588]
        sf.stack_to_cell_files(
            out_dir,
            4 * (1024**3),
            date_range=(datetime(2021, 1, 1), datetime(2021, 1, 15)),
            cells=cells_to_test,
        )

        # assert that the cell files were created ( and no others )
        assert len(list(out_dir.rglob("*.nc"))) == len(cells_to_test)
        assert all(
            [
                f"{c}.nc" in [f.name for f in out_dir.rglob("*.nc")]
                for c in cells_to_test
            ]
        )

        # assert that they are not empty
        for cell, cell_file in zip(cells_to_test, out_dir.rglob("*.nc")):
            ds = xr.open_dataset(cell_file, decode_cf=True, mask_and_scale=False)
            assert len(ds.obs) > 0
            # assert that they contain all the data that the swaths did (for these cells)
            cell_ds_from_swath = sf.read(
                date_range=(datetime(2021, 1, 1), datetime(2021, 1, 15)),
                cell=[cell],
                read_kwargs={"mask_and_scale": False, "decode_cf": True},
            )
            assert len(cell_ds_from_swath.obs) == len(ds.obs)
            for variable in cell_ds_from_swath.data_vars:
                if variable in ["location_id", "lat", "lon", "latitude", "longitude"]:
                    continue
                np.testing.assert_array_equal(
                    cell_ds_from_swath[variable].values, ds[variable].values
                )
                xr.testing.assert_identical(
                    cell_ds_from_swath[variable].reset_coords(drop=True),
                    ds[variable].reset_coords(drop=True),
                )

        # reprocess to contiguous
        from ascat.cell import CellGridFiles
        cf = CellGridFiles.from_product_id(out_dir, "h129")

        contig_out_dir = self.tempdir_path / "contig_cells_out"
        contig_out_dir.mkdir(parents=True, exist_ok=True)

        cf.convert_to_contiguous(contig_out_dir, parallel=True)

        # assert that the cell files were created ( and no others )
        assert len(list(contig_out_dir.rglob("*.nc"))) == len(cells_to_test)
        assert all(
            [
                f"{c}.nc" in [f.name for f in contig_out_dir.rglob("*.nc")]
                for c in cells_to_test
            ]
        )

        # assert that the data is the same as the original
        for cell, cell_file in zip(cells_to_test, out_dir.rglob("*.nc")):
            idx_ds = xr.open_dataset(cell_file, decode_cf=True, mask_and_scale=False).cf_geom.to_indexed_ragged()
            ctg_ds = xr.open_dataset(contig_out_dir / f"{cell}.nc", decode_cf=True, mask_and_scale=False,)
            round_trip_ds = ctg_ds.cf_geom.to_indexed_ragged()

            # time will not be sorted after a round trip
            idx_ds = idx_ds.sortby("time")
            round_trip_ds = round_trip_ds.sortby("time")

            for var in idx_ds.variables:
                xr.testing.assert_identical(idx_ds[var], round_trip_ds[var])
