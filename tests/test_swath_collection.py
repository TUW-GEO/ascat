#!/usr/bin/env python3

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np

from fibgrid.realization import FibGrid

import ascat.read_native.generate_test_data as gtd

from ascat.read_native.swath_collection import SwathFile
from ascat.read_native.swath_collection import SwathGridFiles
# from ascat.read_native.cell_collection import RaggedArrayCell
from ascat.read_native.product_info import AscatH129Swath


def gen_dummy_swathfiles(directory, sat_name=None):
    if sat_name is not None:
        directory = directory / sat_name
        directory.mkdir(parents=True, exist_ok=True)
    gtd.swath_ds.to_netcdf(directory / "swath.nc")
    gtd.swath_ds_2.to_netcdf(directory / "swath_2.nc")


class TestSwathFile(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        self.assertEqual(ra.filenames[0], swath_path)

        # ra_chunked = SwathFile(swath_path, chunks={"locations": 2})
        # self.assertEqual(ra_chunked.filename, swath_path)
        # self.assertIsNone(ra_chunked.ds)
        # self.assertEqual(ra_chunked.chunks, {"locations": 2})

    def test_read(self):
        swath_path = self.tempdir_path / "swath.nc"
        ra = SwathFile(swath_path)
        ds = ra.read()
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("longitude", ds)
        self.assertIn("latitude", ds)
        # self.assertIn("time", ra.ds)
        self.assertIn("obs", ds.dims)

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
        merged = ra1.merge([ds1, ds2])
        self.assertTrue(np.all(merged["location_id"].values == np.concatenate([ds1["location_id"].values, ds2["location_id"].values])))


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
                "placeholder1": "*"
            }

        def _sf_read_fmt(timestamp, sat="[abc]"):
            sat = sat.lower()
            output_fmt = {
                "satellite_folder": {
                    "satellite": f"metop_{sat}"
                },
                "year_folder": {
                    "year": f"{timestamp.year}"
                },
            }
            # if sat is not None:
            #     output_fmt["sat_folder"] = {
            #         "sat": f"metop_{sat.lower()}"
            #     }

            return output_fmt

        # we can create a SwathGridFiles object that points directly to a directory
        # and read the files within it, without passing a "sat" argument to
        # sf.search_period().
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths/metop_a/2021"

        sf = SwathGridFiles(
            swath_path,
            file_class=SwathFile,
            fn_templ="W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H129_C_LIIB_{date}_{placeholder}_{placeholder1}____.nc",
            sf_templ={"satellite_folder": "metop_[abc]", "year_folder": "{year}"},
            date_field_fmt="%Y%m%d%H%M%S",
            grid_name="Fib6.25",
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
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        sf = SwathGridFiles(
            swath_path,
            file_class=SwathFile,
            fn_templ="W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H129_C_LIIB_{date}_{placeholder}_{placeholder1}____.nc",
            sf_templ={"satellite_folder": "metop_[abc]", "year_folder": "{year}"},
            date_field_fmt="%Y%m%d%H%M%S",
            grid_name="Fib6.25",
            fn_read_fmt = _fn_read_fmt,
            sf_read_fmt = _sf_read_fmt
        )

        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S",
            sat="a",
        )
        self.assertGreater(len(files), 0)

    def test_from_product_id(self):
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)

    def test_from_io_class(self):
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        sf = SwathGridFiles.from_io_class(swath_path, AscatH129Swath)
        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)

    def test_extract(self):
        # test extract
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)
        bbox = (-180, -4, -70, 20)

        merged_ds = sf.extract(
            (datetime(2021, 1, 15), datetime(2021, 1, 30)),
            bbox=bbox,
        )

        merged_ds.load()
        self.assertLess(merged_ds.time.max(), np.datetime64(datetime(2021, 1, 30)))
        self.assertGreater(merged_ds.time.min(), np.datetime64(datetime(2021, 1, 15)))
        self.assertLess(merged_ds.latitude.max(), bbox[1])
        self.assertGreater(merged_ds.latitude.min(), bbox[0])
        self.assertLess(merged_ds.longitude.max(), bbox[3])
        self.assertGreater(merged_ds.longitude.min(), bbox[2])

        # test extract from main folder
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths/metop_a/2021"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
        files = sf.search_period(
            datetime(2021, 1, 15),
            datetime(2021, 1, 30),
            date_field_fmt="%Y%m%d%H%M%S"
        )
        self.assertGreater(len(files), 0)
        bbox = (-180, -4, -70, 20)

        merged_ds = sf.extract(
            (datetime(2021, 1, 15), datetime(2021, 1, 30)),
            bbox=bbox,
        )

        merged_ds.load()
        self.assertLess(merged_ds.time.max(), np.datetime64(datetime(2021, 1, 30)))
        self.assertGreater(merged_ds.time.min(), np.datetime64(datetime(2021, 1, 15)))
        self.assertLess(merged_ds.latitude.max(), bbox[1])
        self.assertGreater(merged_ds.latitude.min(), bbox[0])
        self.assertLess(merged_ds.longitude.max(), bbox[3])
        self.assertGreater(merged_ds.longitude.min(), bbox[2])

    def test_stack_to_cell_files(self):
        return
        # cell_path = self.tempdir_path / "cell"
        # cell_path.mkdir(parents=True, exist_ok=True)
        # swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        # swath_path = "ascat_test_data/hsaf/h129/swaths"

        swath_path = "/home/charriso/Projects/ascat/tests/ascat_test_data/hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")

        # sf.stack_to_cell_files("/home/charriso/test_cells/", RaggedArrayCell, datetime(2021, 1, 1), datetime(2021, 2, 1), processes=12, mode="a", chunk=True, load=True)

        sf.stack_to_cell_files(
            "/home/charriso/test_cells/",
            RaggedArrayCell,
            (datetime(2021, 1, 1), datetime(2021, 2, 1)),
            fnames=list(Path(swath_path).rglob("*.nc"))[:2],
            processes=12,
            # sat="b",
            mode="w"
        )
        return

        # sf.stack_to_cell_files_dask(
        #     "/home/charriso/test_cells/",
        #     RaggedArrayCell,
        #     datetime(2021, 1, 1),
        #     datetime(2021, 2, 1),
        #     processes=12,
        #     # sat="b",
        #     # mode="a"
        # )
        #
        # ds = sf.extract(
        #     datetime(2021, 1, 1),
        #     datetime(2021, 1, 2),
        #     # sat="[bc]"
        # )
        # print("hi")
        # #############################
        swath_path = "/home/charriso/p14/data-write/RADAR/hsaf/h129_v1.0/swaths/"
        sf = SwathGridFiles.from_product_id(swath_path, "h129_v1.0")

        # for f in sf.swath_search(datetime(2021, 1, 1), datetime(2021, 1, 7)):
        #     with xr.open_dataset(f) as ds:
        #         if "backscatter_flag" not in ds.variables:
        #             print(f)

        sf.stack_to_cell_files_2("/home/charriso/test_cells/magic/",
                                 RaggedArrayCell,
                                 (datetime(2021, 1, 7), datetime(2021, 1, 14)),
                                 processes=12,
                                 sat="c",
                                 mode="a"
                                 )

        # check that the time var in all files in /test_cells/ are monotonic ascending
        # output_cells = list(Path("/home/charriso/test_cells/").rglob("*.nc"))

        # for cell in output_cells:
        #     ds = xr.open_dataset(cell)
        #     if ds.time.size > 1:
        #         print(cell)
        # ds = sf.extract(datetime(2021, 1, 1), datetime(2021, 2, 1))
        # ds = ds.assign_coords(
        #     {"cell": ("obs", sf.grid.gpi2cell(ds["location_id"].values))}
        # )
        # ds = ds.set_xindex("cell")
        # print(ds)
        #

# import dask.array as da

# all_cells = np.unique(ds.cell.values)

# def m1(ds, cells):
#     for c in cells:
#         ds.sel(cell=c)

# def m2(ds, cells):
#     for c in cells:
#         ds.isel(obs=np.where(ds.cell.values==c)[0])

# def m3(ds, cells):
#     for c in cells:
#         ds.isel(obs=da.where(ds.cell.values==c)[0].compute())

# def m4(ds, cells):
#     for c in cells:
#         ds.where(ds.cell==c, drop=True)

# def m5(ds, cells):
#     for c in ds.groupby("cell"):
#         pass