#!/usr/bin/env python3

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np

import ascat.read_native.generate_test_data as gtd

import ascat.read_native.product_info as product_info

from ascat.read_native.grid_registry import GridRegistry
from ascat.read_native.cell_collection import RaggedArrayCell
from ascat.read_native.cell_collection import CellGridFiles


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

class TestRaggedArrayCellFile(unittest.TestCase):
    """
    Test the merge function
    """

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        gen_dummy_cellfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        contiguous_ragged_path = self.tempdir_path/ "contiguous" / "2588_contiguous_ragged.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        self.assertEqual(ra.filenames[0], contiguous_ragged_path)
        # self.assertIsNone(ra.ds)

    def test_read(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        ds = ra.read(chunks={"obs": 3})
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("lon", ds)
        self.assertIn("lat", ds)
        self.assertIn("locationIndex", ds)
        # self.assertIn("time", ds)
        self.assertIn("obs", ds.dims)
        # assert chunk size
        # self.assertEqual(ds["lon"].data.chunksize, (5,))

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
        ds1 = ra1.read(return_format="point")
        ds2 = ra2.read(return_format="point")
        merged = ra1.merge([ds1, ds2]).cf_geom.to_indexed_ragged()
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

grid_registry = GridRegistry()
class TestCellGridFiles(unittest.TestCase):
    """
    Test a collection of cell files
    """

    @staticmethod
    def _init_options(root_path, sf_templ=None, sf_read_fmt=None):
        return {
            "root_path": root_path,
            "file_class": RaggedArrayCell,
            "fn_format": "{:04d}.nc",
            #"sf_format": sf_templ,
            "grid": grid_registry.get("fibgrid_12.5"),
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
        self.assertEqual(contig_collection._fn(2588).name, "2588.nc")
        self.assertEqual(contig_collection.root_path, self.tempdir_path / "contiguous")
        self.assertEqual(contig_collection.file_class, RaggedArrayCell)

    def test_spatial_search(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path)
        )
        self.assertEqual(contig_collection.spatial_search(),
                         [root_path/"2587.nc", root_path/"2588.nc"])
        self.assertEqual(contig_collection.spatial_search(location_id=1549346),
                         [root_path/"2588.nc"])
        self.assertEqual(contig_collection.spatial_search(location_id=1493629),
                         [root_path/"2587.nc"])
        self.assertEqual(contig_collection.spatial_search(location_id=[1549346, 1493629]),
                         [root_path/"2587.nc", root_path/"2588.nc"])
        self.assertEqual(contig_collection.spatial_search(location_id=[1493629, 1549346, 1493629]),
                         [root_path/"2587.nc", root_path/"2588.nc"])

    def test_read(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        ds_2588 = contig_collection.read(cell=2588)
        self.assertIsInstance(ds_2588, xr.Dataset)

        ds_2587 = contig_collection.read(cell=2587)
        self.assertIsInstance(ds_2587, xr.Dataset)

        ds_cell_merged = contig_collection.read(cell=[2587, 2588])
        self.assertIsInstance(ds_cell_merged, xr.Dataset)

    def test_real(self):
        cells_path = Path("/data-write/RADAR/hsaf/h121_v2.0/time_series/")
        collection = CellGridFiles.from_product_id(cells_path, "h121_v1.0")
        ds_cell_merged = collection.read(bbox=(-3, 4, 109, 113))
        print(ds_cell_merged)

if __name__ == "__main__":
    unittest.main()
