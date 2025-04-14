#!/usr/bin/env python3

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np

import ascat.read_native.generate_test_data as gtd

from ascat.grids import GridRegistry, NamedFileGridRegistry
from ascat.product_info import RaggedArrayCellProduct, OrthoMultiArrayCellProduct
from ascat.cell import RaggedArrayTs
from ascat.cell import OrthoMultiTimeseriesCell
from ascat.cell import CellGridFiles

from get_path import get_testdata_path

TESTDATA_PATH = get_testdata_path()

class RaggedArrayDummyCellProduct(RaggedArrayCellProduct):
    sample_dim = "time"
    grid_name = "fibgrid_12.5"

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

class ERA5Cell(OrthoMultiArrayCellProduct):
    grid_name = "era5land"

    @classmethod
    def preprocessor(cls, ds):
        ds["location_id"].attrs["cf_role"] = "timeseries_id"
        return ds

class GLDASCell(OrthoMultiArrayCellProduct):
    grid_name = "gldas"

    @classmethod
    def preprocessor(cls, ds):
        ds["location_id"].attrs["cf_role"] = "timeseries_id"
        return ds

class TestOrthoMultiCellFile(unittest.TestCase):
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
        # contiguous_ragged_path = self.tempdir_path/ "contiguous" / "2588_contiguous_ragged.nc"
        gldas_path = TESTDATA_PATH / "warp/gldas_2023/"
        gldas_0029_path = TESTDATA_PATH / "warp/gldas_2023/0029.nc"
        gldas_0029 = OrthoMultiTimeseriesCell(gldas_0029_path)
        self.assertTrue(gldas_0029_path.samefile(gldas_0029.filenames[0]))

        cellnum_glob = "[0-9]" * 4 + ".nc"
        gldas_files = list(Path(gldas_path).glob(cellnum_glob))
        gldas = OrthoMultiTimeseriesCell(gldas_files)
        self.assertTrue(all([f in gldas.filenames
                             for f in gldas_files]))
        self.assertFalse("grid.nc" in [f.name for f in gldas.filenames])
        # self.assertIsNone(ra.ds))

    def test_read(self):
        gldas_path = TESTDATA_PATH / "warp/gldas_2023/"
        cellnum_glob = "[0-9]" * 4 + ".nc"
        gldas_files = list(Path(gldas_path).glob(cellnum_glob))
        gldas = OrthoMultiTimeseriesCell(gldas_files)
        # just make sure it works for now
        gldas.read()


    def test_cellgridfiles_read(self):
        gldas_path = TESTDATA_PATH / "warp/gldas_2023/"
        grid = gldas_path / "grid.nc"
        NamedFileGridRegistry.register("gldas", str(grid))
        del grid
        from time import time
        t1 = time()
        grid = GridRegistry().get("gldas")
        t2 = time()

        first_load_time = t2-t1

        for i in range(100):
            t1 = time()
            grid = GridRegistry().get("gldas")
            t2 = time()
            assert t2-t1 < first_load_time

        gldas_files = CellGridFiles.from_product_class(gldas_path, GLDASCell)
        ds = gldas_files.read()
        # assert it's a dataset and not empty
        assert isinstance(ds, xr.Dataset)
        assert len(ds) > 0

    def test_to_raster(self):
        gldas_path = TESTDATA_PATH / "warp/gldas_2023/"
        grid = gldas_path / "grid.nc"
        NamedFileGridRegistry.register("gldas", str(grid))
        del grid

        gldas_files = CellGridFiles.from_product_class(gldas_path, GLDASCell)
        gldas_files.read().cf_geom.to_raster(x_var="lon", y_var="lat")





class TestRaggedArrayCellFile(unittest.TestCase):
    """
    Test the merge function
    """

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        self.indexed_cells_path = TESTDATA_PATH / "hsaf/h129/stack_cells/"
        gen_dummy_cellfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        contiguous_ragged_path = self.tempdir_path/ "contiguous" / "2588_contiguous_ragged.nc"
        ra = RaggedArrayTs(contiguous_ragged_path)
        self.assertEqual(ra.filenames[0], contiguous_ragged_path)
        # self.assertIsNone(ra.ds)

    def test_read(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayTs(contiguous_ragged_path)
        ds = ra.read(chunks={"obs": 3})
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("lon", ds)
        self.assertIn("lat", ds)
        self.assertIn("row_size", ds)
        # self.assertIn("time", ds)
        self.assertIn("obs", ds.dims)
        # assert chunk size
        # self.assertEqual(ds["lon"].data.chunksize, (5,))
        #

    def test__ensure_obs(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayTs(contiguous_ragged_path)
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
        ra = RaggedArrayTs(contiguous_ragged_path)
        ds = xr.open_dataset(contiguous_ragged_path)
        self.assertEqual(ra._indexed_or_contiguous(ds), "contiguous")

    def test__ensure_indexed(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayTs(contiguous_ragged_path)
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
        ra = RaggedArrayTs(indexed_ragged_path)
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
        ra = RaggedArrayTs(indexed_ragged_path)
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
        ra = RaggedArrayTs(indexed_ragged_path)
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
        ra = RaggedArrayTs(contiguous_ragged_path)
        orig_ds = xr.open_dataset(contiguous_ragged_path)
        ds = ra._ensure_contiguous(orig_ds)
        xr.testing.assert_equal(orig_ds, ds)

    def test_merge(self):
        fname1 = self.tempdir_path / "contiguous" / "2588.nc"
        fname2 = self.tempdir_path / "contiguous" / "2587.nc"
        ra1 = RaggedArrayTs(fname1)
        ra2 = RaggedArrayTs(fname2)
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

    def test__merge_contiguous(self):
        fname1 = self.tempdir_path / "contiguous" / "2588.nc"
        fname2 = self.tempdir_path / "contiguous" / "2587.nc"
        ra = RaggedArrayTs([fname1, fname2])
        ds = ra.read()


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
            "file_class": RaggedArrayTs,
            "fn_format": "{:04d}.nc",
            #"sf_format": sf_templ,
            "grid": grid_registry.get("fibgrid_12.5"),
        }

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # self.indexed_cells_path = TESTDATA_PATH / "hsaf/h129/stack_cells/"
        gen_dummy_cellfiles(self.tempdir_path)
        gen_dummy_cellfiles(self.tempdir_path, "metop_a")
        gen_dummy_cellfiles(self.tempdir_path, "metop_b")
        gen_dummy_cellfiles(self.tempdir_path, "metop_c")
        self.indexed_cells_path = self.tempdir_path / "indexed" / "metop_a"
        self.contiguous_cells_path = self.tempdir_path / "contiguous" / "metop_a"



    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        contig_collection = CellGridFiles(
            **self._init_options(self.tempdir_path / "contiguous")
        )
        self.assertEqual(contig_collection._fn(2588).name, "2588.nc")
        self.assertEqual(contig_collection.root_path, self.tempdir_path / "contiguous")
        self.assertEqual(contig_collection.file_class, RaggedArrayTs)

    def test_spatial_search(self):
        root_path = self.tempdir_path / "contiguous" / "metop_a"
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


    def test_read_with_one_coord(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        coord = (175.8, 70.01)
        ds = contig_collection.read(coords=coord)
        ds.load()
        assert len(ds.time) >= 1

    def test_read_with_two_valid_coords(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        coords = (np.array([175.8, 175.4]),
                  np.array([70.01, 70.05]))
        ds = contig_collection.read(coords=coords)
        assert len(ds.time) >= 1

    def test_read_with_invalid_coords(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        coords = (np.array([1, 2, 0]),
                  np.array([10, 20, 0]))
        ds = contig_collection.read(coords=coords)
        assert ds is None

    def test_read_with_two_valid_coords_and_invalid_coord(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        coords = (np.array([175.8, 175.4, 0]),
                  np.array([70.01, 70.05, 0]))
        ds = contig_collection.read(coords=coords)
        ds.load()
        assert len(ds.time) >= 1

    def test_read_indexed(self):
        root_path = self.tempdir_path / "indexed"
        indexed_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        ds = indexed_collection.read(bbox=(70.5, 75, 175.5, 179))

    def test_read_contiguous(self):
        root_path = self.tempdir_path / "contiguous"
        indexed_collection = CellGridFiles(
            **self._init_options(root_path, {"sat_str": "{sat}"}, {"sat_str": {"sat": "metop_[abc]"}})
        )
        ds = indexed_collection.read(bbox=(70.5, 75, 175.5, 179))
        assert len(ds.time) >= 1

    def test_read_single_ts_indexed(self):
        files = list(self.indexed_cells_path.glob("*.nc"))
        first_file_ds = xr.open_dataset(files[0])
        one_valid_gpi = [first_file_ds["location_id"][first_file_ds["locationIndex"][5]].values]
        collection = CellGridFiles.from_product_class(self.indexed_cells_path, RaggedArrayDummyCellProduct)
        ds = collection.read(location_id=one_valid_gpi)
        assert len(ds.cf_geom.to_indexed_ragged().locations) == 1
        assert len(ds.time) >= 1

    def test_read_n_ts_from_one_cell_indexed(self):
        n = 5
        files = list(self.indexed_cells_path.glob("*.nc"))
        first_file_ds = xr.open_dataset(files[0])

        n_valid_gpis = np.unique([first_file_ds["location_id"][first_file_ds["locationIndex"]].values])[:n]
        collection = CellGridFiles.from_product_class(self.indexed_cells_path, RaggedArrayDummyCellProduct)

        # Try reading all at once
        ds = collection.read(location_id=n_valid_gpis)
        idxed_ds = ds.cf_geom.to_indexed_ragged().load()
        assert len(idxed_ds.locations) == n

        # Try reading one by one in a loop
        # This access pattern is slow because under the hood we convert to point array on every read.
        # Better to read all the data at once and then select the points when doing indexed.
        for gpi in n_valid_gpis:
            collection.read(location_id=gpi, return_format="point")

    def test_read_2n_ts_from_two_cells_indexed(self):
        n = 5
        files = list(self.indexed_cells_path.glob("*.nc"))
        first_file_ds = xr.open_dataset(files[0])
        second_file_ds = xr.open_dataset(files[1])

        n_valid_gpis_1 = np.unique([first_file_ds["location_id"][first_file_ds["locationIndex"]].values])[:n]
        n_valid_gpis_2 = np.unique([second_file_ds["location_id"][second_file_ds["locationIndex"]].values])[:n]
        n_times_2_valid_gpis = np.concatenate([n_valid_gpis_1, n_valid_gpis_2])

        collection = CellGridFiles.from_product_class(self.indexed_cells_path, RaggedArrayDummyCellProduct)

        # Try reading all at once
        ds = collection.read(location_id=n_times_2_valid_gpis)
        idxed_ds = ds.cf_geom.to_indexed_ragged().load()
        assert len(idxed_ds.locations) == 2*n

    def test_read_one_cell_indexed(self):
        files = list(self.indexed_cells_path.glob("*.nc"))
        first_cell = int(files[0].stem)

        collection = CellGridFiles.from_product_class(self.indexed_cells_path, RaggedArrayDummyCellProduct)
        ds = collection.read(cell=first_cell)
        assert len(ds.time) >= 1

    def test_read_two_cells_indexed(self):
        files = list(self.indexed_cells_path.glob("*.nc"))
        first_cell = int(files[0].stem)
        second_cell = int(files[1].stem)

        collection = CellGridFiles.from_product_class(self.indexed_cells_path, RaggedArrayDummyCellProduct)
        ds = collection.read(cell=[first_cell, second_cell])
        assert len(ds.time) >= 1

    def test_read_single_ts_contiguous(self):
        files = list(self.contiguous_cells_path.glob("*.nc"))
        first_file_ds = xr.open_dataset(files[0])
        valid_gpis = np.unique(np.repeat(first_file_ds["location_id"].values, first_file_ds["row_size"].values))
        one_valid_gpi = [valid_gpis[0]]
        collection = CellGridFiles.from_product_class(self.contiguous_cells_path, RaggedArrayDummyCellProduct)
        ds = collection.read(location_id=one_valid_gpi)
        assert len(ds.time) >= 1

    def test_read_n_ts_from_one_cell_contiguous(self):
        n = 5
        files = list(self.contiguous_cells_path.glob("*.nc"))
        first_file_ds = xr.open_dataset(files[0])

        valid_gpis = np.unique(np.repeat(first_file_ds["location_id"].values, first_file_ds["row_size"].values))
        n_valid_gpis = valid_gpis[:n]
        collection = CellGridFiles.from_product_class(self.contiguous_cells_path, RaggedArrayDummyCellProduct)

        # Try reading all at once
        ds = collection.read(location_id=n_valid_gpis)
        idxed_ds = ds.cf_geom.to_contiguous_ragged().load()
        assert len(idxed_ds.locations) == n

        # Try reading one by one in a loop
        for gpi in n_valid_gpis:
            new_ds = collection.read(location_id=gpi, return_format="point")
            assert len(new_ds.time) >= 1

    def test_read_2n_ts_from_two_cells_contiguous(self):
        n = 5
        files = list(self.contiguous_cells_path.glob("*.nc"))
        first_file_ds = xr.open_dataset(files[0])
        second_file_ds = xr.open_dataset(files[1])

        valid_gpis_1 = np.unique(np.repeat(first_file_ds["location_id"].values, first_file_ds["row_size"].values))
        n_valid_gpis_1 = valid_gpis_1[:n]
        valid_gpis_2 = np.unique(np.repeat(second_file_ds["location_id"].values, second_file_ds["row_size"].values))
        n_valid_gpis_2 = valid_gpis_2[:n]
        n_times_2_valid_gpis = np.concatenate([n_valid_gpis_1, n_valid_gpis_2])

        collection = CellGridFiles.from_product_class(self.contiguous_cells_path, RaggedArrayDummyCellProduct)

        # Try reading all at once
        ds = collection.read(location_id=n_times_2_valid_gpis)
        idxed_ds = ds.cf_geom.to_contiguous_ragged().load()
        assert len(idxed_ds.locations) == 2*n

    def test_read_one_cell_contiguous(self):
        files = list(self.contiguous_cells_path.glob("*.nc"))
        first_cell = int(files[0].stem)

        collection = CellGridFiles.from_product_class(self.contiguous_cells_path, RaggedArrayDummyCellProduct)
        ds = collection.read(cell=first_cell)
        assert len(ds.time) >= 1

    def test_read_two_cells_contiguous(self):
        files = list(self.contiguous_cells_path.glob("*.nc"))
        first_cell = int(files[0].stem)
        second_cell = int(files[1].stem)

        collection = CellGridFiles.from_product_class(self.contiguous_cells_path, RaggedArrayDummyCellProduct)
        ds = collection.read(cell=[first_cell, second_cell])
        assert len(ds.time) >= 1

if __name__ == "__main__":
    unittest.main()
