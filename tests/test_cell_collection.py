#!/usr/bin/env python3

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np

from fibgrid.realization import FibGrid

from pygeogrids.netcdf import load_grid

import ascat.read_native.generate_test_data as gtd

from ascat.read_native.product_info import cell_io_catalog
from ascat.read_native.cell_collection import grid_cache

from ascat.read_native.cell_collection import RaggedArrayCell
from ascat.read_native.cell_collection import CellGridFiles
from ascat.read_native.cell_collection import RaggedArrayFiles
from ascat.read_native.cell_collection import OrthoMultiCell
from ascat.read_native.cell_collection import OrthoMultiArrayFiles


def add_sat_id(ds, sat_name):
    name_dict = {"metop_a": 3, "metop_b": 4, "metop_c": 5}
    if sat_name is not None:
        ds["sat_id"] = ("time", np.repeat(name_dict[sat_name], ds.sizes["time"]))
    return ds

def gen_dummy_cellfiles(dir, sat_name=None):
    contiguous_dir = dir / "contiguous"
    if sat_name is not None:
        contiguous_dir = contiguous_dir / sat_name
    contiguous_dir.mkdir(parents=True, exist_ok=True)
    indexed_dir = dir / "indexed"
    if sat_name is not None:
        indexed_dir = indexed_dir / sat_name
    indexed_dir.mkdir(parents=True, exist_ok=True)
    add_sat_id(gtd.contiguous_ragged_ds_2588, sat_name).to_netcdf(contiguous_dir / "2588.nc")
    add_sat_id(gtd.indexed_ragged_ds_2588, sat_name).to_netcdf(indexed_dir / "2588.nc")
    add_sat_id(gtd.contiguous_ragged_ds_2587, sat_name).to_netcdf(contiguous_dir / "2587.nc")
    add_sat_id(gtd.indexed_ragged_ds_2587, sat_name).to_netcdf(indexed_dir / "2587.nc")

class TestRaggedArrayCell(unittest.TestCase):
    """
    Test the merge function
    """

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # print("hi")
        gen_dummy_cellfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        contiguous_ragged_path = self.tempdir_path/ "contiguous" / "2588_contiguous_ragged.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        self.assertEqual(ra.filename, contiguous_ragged_path)
        self.assertIsNone(ra.ds)

        ra_chunked = RaggedArrayCell(contiguous_ragged_path, chunks={"locations": 2})
        self.assertEqual(ra_chunked.filename, contiguous_ragged_path)
        self.assertIsNone(ra_chunked.ds)
        self.assertEqual(ra_chunked.chunks, {"locations": 2})

    def test_read(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        ra.read()
        self.assertIsInstance(ra.ds, xr.Dataset)
        self.assertIn("lon", ra.ds)
        self.assertIn("lat", ra.ds)
        self.assertIn("locationIndex", ra.ds)
        # self.assertIn("time", ra.ds)
        self.assertIn("obs", ra.ds.dims)
        # assert chunk size
        self.assertEqual(ra.ds["lon"].data.chunksize, (5,))

    def test__ensure_obs(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        ds = xr.open_dataset(contiguous_ragged_path)
        # original_dim = ds["time"]
        self.assertNotIn("obs", ds.dims)
        self.assertIn("time", ds.dims)
        ds = ra._ensure_obs(ds)
        self.assertIn("obs", ds.dims)
        self.assertNotIn("time", ds.dims)
        # print(original_dim.values)
        # np.testing.assert_equal(ds["obs"].values, original_dim.values)

    def test__indexed_or_contiguous(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        ds = xr.open_dataset(contiguous_ragged_path)
        self.assertEqual(ra._indexed_or_contiguous(ds), "contiguous")

    def test__ensure_indexed(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        ds = xr.open_dataset(contiguous_ragged_path)
        self.assertNotIn("locationIndex", ds)
        self.assertIn("row_size", ds)

        self.assertIsNone(ra._ensure_indexed(None))

        indexed_ds = ra._ensure_indexed(ds)
        self.assertIn("locationIndex", indexed_ds)
        self.assertNotIn("row_size", indexed_ds)

        xr.testing.assert_equal(ra._ensure_indexed(indexed_ds), indexed_ds)

        np.testing.assert_array_equal(indexed_ds["locationIndex"].values,
                                      np.array([0, 0, 1, 2, 2, 3, 3, 3, 3, 4],
                                               dtype=np.int32))

    def test__ensure_contiguous(self):
        indexed_ragged_path = self.tempdir_path / "indexed" / "2588.nc"
        ra = RaggedArrayCell(indexed_ragged_path)
        ds = xr.open_dataset(indexed_ragged_path)
        ds = ds.chunk({"time": 1_000_000})
        self.assertIn("locationIndex", ds)
        self.assertNotIn("row_size", ds)

        self.assertIsNone(ra._ensure_contiguous(None))

        contiguous_ds = ra._ensure_contiguous(ds)
        self.assertNotIn("locationIndex", contiguous_ds)
        self.assertIn("row_size", contiguous_ds)

        xr.testing.assert_equal(ra._ensure_contiguous(contiguous_ds), contiguous_ds)

        np.testing.assert_array_equal(contiguous_ds["row_size"].values,
                                        np.array([2, 1, 2, 4, 1], dtype=np.int32))

    def test__trim_to_gpis(self):
        indexed_ragged_path = self.tempdir_path / "indexed" / "2588.nc"
        ra = RaggedArrayCell(indexed_ragged_path)
        ds = xr.open_dataset(indexed_ragged_path)
        ds = ra._ensure_obs(ds)
        ds = ds.chunk({"obs": 1_000_000})
        valid_gpis = [1549346, 1555912]
        trimmed_ds = ra._trim_to_gpis(ds, valid_gpis)
        self.assertTrue(np.all(np.isin(trimmed_ds["location_id"].values, valid_gpis)))
        new_obs_gpis = trimmed_ds["location_id"].values[trimmed_ds["locationIndex"].values]
        np.testing.assert_array_equal(new_obs_gpis, np.repeat(np.array([1549346, 1555912]), [2, 4]))

        # DELETE THIS
        # real_file = Path("/home/charriso/p14/data-write/RADAR/charriso/sig0_12.5/stack_cell_merged_sig0/metop_a")/"2343.nc"
        # grid = FibGrid(12.5)
        # date_range = (np.datetime64("2016-01-01T00:00:00"), np.datetime64("2017-01-01T00:00:00"))
        # ds2 = RaggedArray(real_file).read()
        # valid_gpis = list(ds2["location_id"].data[:500].compute())
        # gpi_lookup = np.zeros(grid.gpis.max()+1, dtype=bool)
        # gpi_lookup[valid_gpis] = True
        # print(ds2)
        # from time import time
        # start = time()
        # datetrim = ra._trim_var_range(ds2, "time", *date_range)
        # print("time to trim dates to a year")
        # print(time() - start)
        # start = time()
        # # gpitrim = ra._trim_to_gpis(ds2, valid_gpis)
        # gpitrim = ra._trim_to_gpis(ds2, lookup_vector=gpi_lookup)
        # print(f"time to trim to {len(valid_gpis)} gpis")
        # print(time() - start)
        # print(f"time to trimp to {len(valid_gpis)} gpis after trimming to a year")
        # start = time()
        # # gpitrim2 = ra._trim_to_gpis(datetrim, valid_gpis)
        # gpitrim2 = ra._trim_to_gpis(datetrim, lookup_vector=gpi_lookup)
        # print(time() - start)
        # print(f"time to trim dates to a year after trimming to {len(valid_gpis)} gpis")
        # start = time()
        # datetrim2 = ra._trim_var_range(gpitrim, "time", *date_range)
        # print(time() - start)

    def test__trim_var_range(self):
        indexed_ragged_path = self.tempdir_path / "indexed" / "2588.nc"
        ra = RaggedArrayCell(indexed_ragged_path)
        ds = xr.open_dataset(indexed_ragged_path)
        ds = ra._ensure_obs(ds)
        ds = ds.chunk({"obs": 1_000_000})
        start_dt = np.datetime64("2020-01-01T00:00:02")
        end_dt = np.datetime64("2020-01-01T00:00:04")
        trimmed_ds = ra._trim_var_range(ds, "time", start_dt, end_dt)
        self.assertTrue(np.all(trimmed_ds["time"].values < end_dt))
        self.assertTrue(np.all(trimmed_ds["time"].values >= start_dt))


    def test_back_and_forth(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2587.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        orig_ds = xr.open_dataset(contiguous_ragged_path)
        ds = ra._ensure_contiguous(orig_ds)
        xr.testing.assert_equal(orig_ds, ds)

    def test_merge(self):
        fname1 = self.tempdir_path / "contiguous" / "2588.nc"
        fname2 = self.tempdir_path / "contiguous" / "2587.nc"
        ra1 = RaggedArrayCell(fname1)
        ra2 = RaggedArrayCell(fname2)
        ra1.read()
        ra2.read()
        merged = ra1.merge([ra1.ds, ra2.ds])
        np.testing.assert_array_equal(merged["locationIndex"].values,
                                        np.array([5, 5, 6, 7, 7, 8, 8, 8, 8, 9, 0, 0, 0, 1, 2, 3, 3, 3, 4, 4],
                                                 dtype=np.int32))
        contig = ra1._ensure_contiguous(merged)
        np.testing.assert_array_equal(contig["row_size"].values,
                                        np.array([3, 1, 1, 3, 2, 2, 1, 2, 4, 1], dtype=np.int32))

        self.assertIsNone(ra1.merge([]))

    # def test__trim_var_range(self):
    #     contiguous_ragged_fname = "2588_contiguous_ragged.nc"
    #     ra = RaggedArray(self.tempdir_path / contiguous_ragged_fname)
    #     ra.read()

    # def test__location_vars_from_ds_list(self):
    #     contiguous_ragged_fname = "2588_contiguous_ragged.nc"
    #     ra = RaggedArray(self.tempdir_path / contiguous_ragged_fname)
    #     ra.read()
    #     # print(ra._location_vars_from_ds_list([ra.ds]))

    # def test__only_locations(self):
    #     contiguous_ragged_fname = "2588_contiguous_ragged.nc"
    #     ra = RaggedArray(self.tempdir_path / contiguous_ragged_fname)
    #     ra.read()
    #     xr.testing.assert_equal(ra._only_locations(ra.ds),
    #                             ra.ds[["lon", "lat", "location_id"]])
        # print(ra._only_locations(ra.ds))


    # def test_read_period(self):
        # ra = RaggedArray(self.tempdir_path / "contiguous_ragged.nc")
        # ra.read_period(datetime(2020, 1, 1), datetime(2020, 1, 2))

class TestCellGridFiles(unittest.TestCase):
    """
    Test a collection of cell files
    """

    @staticmethod
    def _init_options(root_path, sf_templ=None, sf_read_fmt=None):
        return {
            "root_path": root_path,
            "cls": RaggedArrayCell,
            "fn_templ": "{cell_id}.nc",
            "sf_templ": sf_templ,
            "grid_name": "Fib12.5",
            "fn_read_fmt": lambda cell: {"cell_id": f"{cell:04d}"},
            "sf_read_fmt": sf_read_fmt,
        }

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        gen_dummy_cellfiles(self.tempdir_path)
        gen_dummy_cellfiles(self.tempdir_path, "metop_a")
        gen_dummy_cellfiles(self.tempdir_path, "metop_b")
        gen_dummy_cellfiles(self.tempdir_path, "metop_c")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        contig_collection = CellGridFiles(
            **self._init_options(self.tempdir_path / "contiguous")
        )
        self.assertEqual(contig_collection.fn_read_fmt(2588), {"cell_id": "2588"})
        self.assertIsNone(contig_collection.sf_read_fmt)
        self.assertEqual(contig_collection.root_path, self.tempdir_path / "contiguous")
        self.assertEqual(contig_collection.cls, RaggedArrayCell)

    # def test_read(self):
    #     contig_collection = CellGridFiles(
    #         **self._init_options(self.tempdir_path / "contiguous")
    #     )
    #     ds_2588 = contig_collection.read(2588)
    #     self.assertIsInstance(ds_2588, xr.Dataset)
    #     self.assertIn("lon", ds_2588)
    def test__cells_for_location_id(self):
        contig_collection = CellGridFiles(
            **self._init_options(self.tempdir_path / "contiguous")
        )
        self.assertEqual(contig_collection._cells_for_location_id([1549346]), [2588])
        self.assertEqual(contig_collection._cells_for_location_id([1493629]), [2587])
        # self.assertEqual(contig_collection._cells_for_location_id([1549346, 1493629]), [2588, 2587])
        self.assertListEqual(list(contig_collection._cells_for_location_id([1549346, 1493629])), [2588, 2587])

    def test_search(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path)
        )
        self.assertEqual(contig_collection.spatial_search(),
                         [str(root_path/"2587.nc"), str(root_path/"2588.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=1549346),
                         [str(root_path/"2588.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=1493629),
                         [str(root_path/"2587.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=[1549346, 1493629]),
                         [str(root_path/"2587.nc"), str(root_path/"2588.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=[1493629, 1549346, 1493629]),
                         [str(root_path/"2587.nc"), str(root_path/"2588.nc")])

    def test_extract(self):
        # sf_pattern = {
        #     "satellite_folder": "metop_[abc]",
        # }
        # sf_read_fmt = lambda x: {
        #     "satellite_folder": {"satellite": "metop_[abc]"},
        # }
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        ds_2588 = contig_collection.extract(cell=2588)
        self.assertIsInstance(ds_2588, xr.Dataset)

        ds_2587 = contig_collection.extract(cell=2587)
        self.assertIsInstance(ds_2587, xr.Dataset)

        ds_cell_merged = contig_collection.extract(cell=[2587, 2588])
        self.assertIsInstance(ds_cell_merged, xr.Dataset)

        # # DELETE THIS
        # from time import time
        # collection_path = Path("/home/charriso/p14/data-write/RADAR/charriso/sig0_12.5/stack_cell_merged_sig0/")
        # real_collection = CellGridFiles(
        #     **self._init_options(collection_path)
        # )
        # bbox = (-7, -4, -69, -65)
        # # real_merged = real_collection.extract(cell=[2587, 2588])
        # start = time()
        # real_merged = real_collection.extract(bbox=bbox)
        # print(real_merged)



class TestRaggedArrayFiles(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        gen_dummy_cellfiles(self.tempdir_path)
        gen_dummy_cellfiles(self.tempdir_path, "metop_a")
        gen_dummy_cellfiles(self.tempdir_path, "metop_b")
        gen_dummy_cellfiles(self.tempdir_path, "metop_c")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        contig_collection = RaggedArrayFiles(
            self.tempdir_path / "contiguous",
            product_id="sig0_12.5",
        )
        self.assertEqual(contig_collection.fn_read_fmt(2588), {"cell_id": "2588"})
        self.assertIsNone(contig_collection.sf_read_fmt)
        self.assertEqual(contig_collection.root_path, self.tempdir_path / "contiguous")
        self.assertEqual(contig_collection.cls, RaggedArrayCell)

    def test_search(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = RaggedArrayFiles(
            root_path,
            product_id="sig0_12.5",
        )
        self.assertEqual(contig_collection.spatial_search(),
                         [str(root_path/"2587.nc"), str(root_path/"2588.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=1549346),
                         [str(root_path/"2588.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=1493629),
                         [str(root_path/"2587.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=[1549346, 1493629]),
                         [str(root_path/"2587.nc"), str(root_path/"2588.nc")])
        self.assertEqual(contig_collection.spatial_search(location_id=[1493629, 1549346, 1493629]),
                         [str(root_path/"2587.nc"), str(root_path/"2588.nc")])

    def test_extract(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = RaggedArrayFiles(
            root_path,
            product_id="sig0_12.5",
        )
        real_merged = contig_collection.extract(cell=[2587, 2588])
        self.assertIsInstance(real_merged, xr.Dataset)

        root_path = self.tempdir_path
        allsats_collection = RaggedArrayFiles(
            root_path,
            product_id="sig0_12.5",
            all_sats=True,
        )
        real_merged = allsats_collection.extract(cell=[2587, 2588])
        self.assertIsNone(real_merged)

    def test_convert_dir_to_contiguous(self):
        root_path = self.tempdir_path / "indexed"
        indexed_collection = RaggedArrayFiles(
            root_path,
            product_id="sig0_12.5",
        )
        converted_dir = self.tempdir_path / "converted_contiguous"
        converted_dir.mkdir(parents=True, exist_ok=True)
        indexed_collection.convert_dir_to_contiguous(converted_dir, num_processes=-1)
        # show all files in the converted directory
        # print(list(converted_dir.rglob("*")))



# test adding new cell types used with OrthoMultiArray
era5_grid = load_grid("tests/ascat_test_data/warp/era5_land_2023/grid.nc")
grid_cache.fetch_or_store("Era5Land", era5_grid)

gldas_grid = load_grid("tests/ascat_test_data/warp/gldas_2023/grid.nc")
grid_cache.fetch_or_store("GLDAS", gldas_grid)

cci_passive_grid = load_grid("tests/ascat_test_data/warp/cci_passive_v07.1/grid.nc")
grid_cache.fetch_or_store("CCI_PASSIVE", cci_passive_grid)

class ERA5Cell():
    grid_name = "Era5Land"
    grid_info = grid_cache.fetch_or_store(grid_name)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

class GLDASCell():
    grid_name = "GLDAS"
    grid_info = grid_cache.fetch_or_store(grid_name)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

class CCI_PassiveCell():
    grid_name = "CCI_PASSIVE"
    grid_info = grid_cache.fetch_or_store(grid_name)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

cell_io_catalog["ERA5"] = ERA5Cell
cell_io_catalog["GLDAS"] = GLDASCell
cell_io_catalog["CCI_PASSIVE"] = CCI_PassiveCell


class TestOrthoMultiCell(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        warp_path = Path("tests/ascat_test_data/warp")
        self.era5_path = warp_path / "era5_land_2023"
        self.gldas_path = warp_path / "gldas_2023"
        self.cci_passive_path = warp_path / "cci_passive_v07.1"
        # print("hi")
        # gen_dummy_cellfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        om = OrthoMultiCell(self.era5_path / "0030.nc")
        self.assertEqual(om.filename, self.era5_path / "0030.nc")
        self.assertIsNone(om.ds)

        # ra_chunked = RaggedArrayCell(contiguous_ragged_path, chunks={"locations": 2})
        # self.assertEqual(ra_chunked.filename, contiguous_ragged_path)
        # self.assertIsNone(ra_chunked.ds)
        # self.assertEqual(ra_chunked.chunks, {"locations": 2})

    def test_read(self):
        om = OrthoMultiCell(self.era5_path / "0030.nc")
        om.read()
        self.assertIsInstance(om.ds, xr.Dataset)

        om.read(valid_gpis=[905400, 905401])
        self.assertTrue(np.all(np.isin(om.ds.location_id.values, [905400, 905401]))
                        and len(om.ds.location_id.values) == 2)
        self.assertTrue(np.all(np.isin([905400, 905401], om.ds.location_id.values)))
        self.assertIn("lon", om.ds)
        self.assertIn("lat", om.ds)
        self.assertIn("time", om.ds)
        self.assertIn("locations", om.ds.dims)
        # # assert chunk size
        self.assertEqual(om.ds["location_id"].data.chunksize, (2,))

    def test_merge(self):
        fname1 = self.era5_path / "0029.nc"
        fname2 = self.era5_path / "0030.nc"
        om1 = OrthoMultiCell(fname1)
        om2 = OrthoMultiCell(fname2)
        om1.read()
        print(era5_grid.gpis.max())
        print(om1.ds.location_id.values.max())
        om2.read()
        # print(om1.ds.location_id.values)
        merged = om1.merge([om1.ds, om2.ds])
        # print(merged.location_id.values)

        fname1 = self.gldas_path / "0029.nc"
        fname2 = self.gldas_path / "0030.nc"
        om1 = OrthoMultiCell(fname1)
        om2 = OrthoMultiCell(fname2)
        om1.read()
        print(gldas_grid.gpis.max())
        om2.read()
        print(om2.ds.location_id.values.max())
        merged = om1.merge([om1.ds, om2.ds])

        # fname1 = self.cci_passive_path / "0030.nc"
        # fname2 = self.cci_passive_path / "0031.nc"
        # om1 = OrthoMultiCell(fname1)
        # print(cci_passive_grid.gpis.max())
        # print(om1.ds.location_id.values.max())
        # om2 = OrthoMultiCell(fname2)

        # merged = om1.merge([om1.ds, om2.ds])



class TestOrthoMultiArrayFiles(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        warp_path = Path("tests/ascat_test_data/warp")
        self.era5_path = warp_path / "era5_land_2023"
        self.gldas_path = warp_path / "gldas_2023"
        self.cci_passive_path = warp_path / "cci_passive_v07.1"
        # gen_dummy_cellfiles(self.tempdir_path)
        # gen_dummy_cellfiles(self.tempdir_path, "metop_a")
        # gen_dummy_cellfiles(self.tempdir_path, "metop_b")
        # gen_dummy_cellfiles(self.tempdir_path, "metop_c")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        om_collection = OrthoMultiArrayFiles(
            self.era5_path,
            product_id="ERA5",
        )
        self.assertEqual(om_collection.fn_read_fmt(29), {"cell_id": "0029"})
        self.assertIsNone(om_collection.sf_read_fmt)
        self.assertEqual(om_collection.root_path, self.era5_path)
        self.assertEqual(om_collection.cls, OrthoMultiCell)

    def test_search(self):
        om_collection = OrthoMultiArrayFiles(
            self.era5_path,
            product_id="ERA5",
        )
        self.assertEqual(om_collection.spatial_search(),
                         [str(self.era5_path/"0029.nc"),
                          str(self.era5_path/"0030.nc"),
                          str(self.era5_path/"0140.nc")])
        self.assertEqual(om_collection.spatial_search(location_id=1085400),
                         [str(self.era5_path/"0029.nc")])
        self.assertEqual(om_collection.spatial_search(location_id=1081849),
                         [str(self.era5_path/"0030.nc")])
        self.assertEqual(om_collection.spatial_search(location_id=[1085400, 1081849]),
                         [str(self.era5_path/"0029.nc"),
                          str(self.era5_path/"0030.nc")])
        self.assertEqual(om_collection.spatial_search(location_id=[1081849, 1085400, 1081849]),
                         [str(self.era5_path/"0029.nc"),
                          str(self.era5_path/"0030.nc")])


    def test_extract(self):
        om_collection = OrthoMultiArrayFiles(
            self.era5_path,
            product_id="ERA5",
        )
        real_merged = om_collection.extract(cell=[29, 30])
        self.assertIsInstance(real_merged, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
