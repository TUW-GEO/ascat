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

# from ascat.read_native.product_info import cell_io_catalog
from ascat.read_native.product_info import register_cell_grid_reader
#from ascat.read_native.product_info import grid_cache

from ascat.read_native.cell_collection import RaggedArrayCell
#from ascat.read_native.cell_collection import IndexedRaggedArrayFile
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



import unittest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import patch, MagicMock
from ascat.read_native.cell_collection import RaggedArrayCell

# class TestRaggedArrayCellFile(unittest.TestCase):

#     def setUp(self):
#         self.test_file = 'test_file.nc'
#         self.test_ds = xr.Dataset({
#             'location_id': ('locations', [1, 2, 3]),
#             'lon': ('locations', [10, 20, 30]),
#             'lat': ('locations', [40, 50, 60]),
#             'alt': ('locations', [np.nan, np.nan, np.nan]),
#             'time': ('obs', [np.datetime64('2020-01-01'), np.datetime64('2020-01-02')]),
#             'data': ('obs', [1.1, 2.2]),
#             'locationIndex': ('obs', [0, 1])
#         })

#     def test_read(self):
#         with patch('xarray.open_dataset', return_value=self.test_ds):
#             raf = RaggedArrayCellFile([self.test_file])
#             result = raf.read()
#             self.assertIsInstance(result, xr.Dataset)
#             self.assertEqual(len(result['obs']), 2)

#     def test_read_with_date_range(self):
#         with patch('xarray.open_dataset', return_value=self.test_ds):
#             raf = RaggedArrayCellFile([self.test_file])
#             result = raf.read(date_range=(np.datetime64('2020-01-01'), np.datetime64('2020-01-02')))
#             self.assertIsInstance(result, xr.Dataset)
#             self.assertEqual(len(result['obs']), 2)

#     def test_read_with_valid_gpis(self):
#         with patch('xarray.open_dataset', return_value=self.test_ds):
#             raf = RaggedArrayCellFile([self.test_file])
#             result = raf.read(valid_gpis=[1, 2])
#             self.assertIsInstance(result, xr.Dataset)
#             self.assertEqual(len(result['locations']), 2)

#     def test_ensure_indexed(self):
#         contiguous_ds = xr.Dataset({
#             'location_id': ('locations', [1, 2, 3]),
#             'lon': ('locations', [10, 20, 30]),
#             'lat': ('locations', [40, 50, 60]),
#             'alt': ('locations', [np.nan, np.nan, np.nan]),
#             'row_size': ('locations', [2, 1, 0]),
#             'data': ('obs', [1.1, 2.2, 3.3]),
#             'time': ('obs', [np.datetime64('2020-01-01'), np.datetime64('2020-01-02'), np.datetime64('2020-01-03')]),
#         })
#         raf = RaggedArrayCellFile([self.test_file])
#         result = raf._ensure_indexed(contiguous_ds)
#         self.assertIn('locationIndex', result.data_vars)
#         self.assertNotIn('row_size', result.data_vars)

#     def test_ensure_contiguous(self):
#         indexed_ds = xr.Dataset({
#             'location_id': ('locations', [1, 2, 3]),
#             'lon': ('locations', [10, 20, 30]),
#             'lat': ('locations', [40, 50, 60]),
#             'alt': ('locations', [np.nan, np.nan, np.nan]),
#             'locationIndex': ('obs', [0, 1, 1]),
#             'data': ('obs', [1.1, 2.2, 3.3]),
#             'time': ('obs', [np.datetime64('2020-01-01'), np.datetime64('2020-01-02'), np.datetime64('2020-01-03')])
#         })
#         result = RaggedArrayCellFile._ensure_contiguous(indexed_ds)
#         self.assertIn('row_size', result.data_vars)
#         self.assertNotIn('locationIndex', result.data_vars)

#     def test_merge(self):
#         ds1 = xr.Dataset({
#             'location_id': ('locations', [1, 2]),
#             'lon': ('locations', [10, 20]),
#             'lat': ('locations', [40, 50]),
#             'alt': ('locations', [np.nan, np.nan]),
#             'data': ('obs', [1.1, 2.2]),
#             'locationIndex': ('obs', [0, 1]),
#             'time': ('obs', [np.datetime64('2020-01-01'), np.datetime64('2020-01-02')]),
#         })
#         ds2 = xr.Dataset({
#             'location_id': ('locations', [2, 3]),
#             'lon': ('locations', [20, 30]),
#             'lat': ('locations', [50, 60]),
#             'alt': ('locations', [np.nan, np.nan]),
#             'data': ('obs', [3.3, 4.4]),
#             'locationIndex': ('obs', [0, 1]),
#             'time': ('obs', [np.datetime64('2020-01-02'), np.datetime64('2020-01-03')]),
#         })
#         raf = RaggedArrayCellFile([self.test_file])
#         result = raf._merge([ds1, ds2])
#         self.assertEqual(len(result['locations']), 3)
#         self.assertEqual(len(result['obs']), 4)

#     def test_trim_to_gpis(self):
#         raf = RaggedArrayCellFile([self.test_file])
#         result = raf._trim_to_gpis(self.test_ds, gpis=[1, 2])
#         self.assertEqual(len(result['locations']), 2)

#     def test_write(self):
#         raf = RaggedArrayCellFile([self.test_file])
#         raf.ds = self.test_ds
#         with patch('xarray.Dataset.to_netcdf') as mock_to_netcdf:
#             raf.write('output.nc')
#             mock_to_netcdf.assert_called_once()

#     def test_write_append_mode(self):
#         raf = RaggedArrayCellFile([self.test_file])
#         raf.ds = self.test_ds
#         raf.write('output.nc')
#         with patch('ascat.utils.append_to_netcdf') as mock_append:
#             raf.write('output.nc', mode='a', ra_type='indexed')
#             mock_append.assert_called_once()

from time import time
from ascat.read_native.cell_collection import ContiguousRaggedArrayHandler
from ascat.read_native.cell_collection import OneDimArrayHandler
from ascat.read_native.cell_collection import IndexedRaggedArrayHandler



class TestContiguousRaggedArrayHandler(unittest.TestCase):
    def setUp(self):
        #self.path1 = Path("tests/ascat_test_data/hsaf/h129/stack_cells/")
        self.contiguous_path = Path("/data-write/RADAR/hsaf/h121_v2.0/time_series/")

    # def test_one_location(self):
    #     all_cells = list(self.contiguous_path.glob("100*.nc"))
    #     cell = RaggedArrayCell(all_cells[0:10])
    #     # locations = [5974490, 5974545, 5974634, 5974778,5974867, 5974922, 5975011, 5975155, 5975244,5696947, 5697934, 5698544, 5698921, 5699531, 5700518, 5701128, 5701505, 5701738, 5702115]
    #     # ds = cell.read(location_id=locations)
    #     #
    #     ds = cell.read(date_range = (np.datetime64("2020-01-01"), np.datetime64("2020-01-15")))
    #     print(ds)

    def test_to_1d_array(self):
        handler = ContiguousRaggedArrayHandler
        contiguous_cells = list(self.contiguous_path.glob("100*.nc"))
        ds = xr.open_dataset(contiguous_cells[0])
        start = time()
        ds_1d = handler.to_1d_array(ds)
        t1 = time() - start
        ds_reverted = handler.from_1d_array(ds_1d)
        t2 = time() - t1

        num_locations = 100
        random_indices = np.random.randint(0, len(ds_1d), num_locations)
        random_location_ds = ds_1d["location_id"].values[random_indices]

        print("to_1d_array", t1)
        print("from_1d_array to contiguous", t2)

        start = time()
        handler.trim_to_gpis(ds, random_location_ds).compute()
        print(f"read {num_locations} locations from original contiguous", time() - start)

        start = time()
        handler.trim_to_gpis(ds_reverted, random_location_ds).compute()
        print(f"read {num_locations} locations from reverted contiguous", time() - start)

        start = time()
        OneDimArrayHandler.trim_to_gpis(ds_1d, gpis=random_location_ds).compute()
        print(f"read {num_locations} locations from 1d with gpis", time() - start)
        lookup_vector = np.zeros(ds_1d["location_id"].max().values + 1, dtype=bool)
        lookup_vector[random_indices] = True
        start = time()
        OneDimArrayHandler.trim_to_gpis(ds_1d, lookup_vector=lookup_vector).compute()
        print(f"read {num_locations} locations from 1d with lookup vector", time() - start)



        # print(ds)
        # print(ds_1d)
        # print(ds_reverted)

        handler = IndexedRaggedArrayHandler
        start = time()
        ds_indexed = handler.from_1d_array(ds_1d)
        t3 = time() - start
        print("from_1d_array to indexed", t3)

        start = time()
        handler.trim_to_gpis(ds_indexed, random_location_ds).compute()
        print(f"read {num_locations} locations from reverted indexed", time() - start)



class TestRaggedArrayCellFile_new(unittest.TestCase):
    def setUp(self):
        #self.path1 = Path("tests/ascat_test_data/hsaf/h129/stack_cells/")
        self.contiguous_path = Path("/data-write/RADAR/hsaf/h121_v2.0/time_series/")
        self.indexed_path = Path("/data-write/RADAR/hsaf/h129_v1.0/ts/")

    # def test_one_location(self):
    #     all_cells = list(self.contiguous_path.glob("100*.nc"))
    #     cell = RaggedArrayCell(all_cells[0:10])
    #     # locations = [5974490, 5974545, 5974634, 5974778,5974867, 5974922, 5975011, 5975155, 5975244,5696947, 5697934, 5698544, 5698921, 5699531, 5700518, 5701128, 5701505, 5701738, 5702115]
    #     # ds = cell.read(location_id=locations)
    #     #
    #     ds = cell.read(date_range = (np.datetime64("2020-01-01"), np.datetime64("2020-01-15")))
    #     print(ds)

    def test_read(self):
        # date_range
        # # Make sure for both RA types, with single and multiple cells, that "date_range"
        # # is faster than reading without date_range and filtering afterward.
        contiguous_cells = list(self.contiguous_path.glob("100*.nc"))
        indexed_cells = list(self.indexed_path.glob("100*.nc"))

        # # contiguous, single cell
        cell = RaggedArrayCell(contiguous_cells[0:1])
        start = time()
        ds = cell.read(date_range=(np.datetime64("2020-01-01"), np.datetime64("2020-01-15")))
        t1 = time() - start
        print("contiguous, single cell, date_range", t1)




class TestRaggedArrayCellFile(unittest.TestCase):
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
        self.assertEqual(ra.filenames[0], contiguous_ragged_path)
        # self.assertIsNone(ra.ds)

    def test_read(self):
        contiguous_ragged_path = self.tempdir_path / "contiguous" / "2588.nc"
        ra = RaggedArrayCell(contiguous_ragged_path)
        ra.read(chunks={"obs": 3})
        print(ra.ds)
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
        def _fn_read_fmt(cell, sat=None):
            return {"cell_id": f"{cell:04d}"}
        return {
            "root_path": root_path,
            "cls": RaggedArrayCell,
            "fn_templ": "{cell_id}.nc",
            "sf_templ": sf_templ,
            "grid_name": "Fib12.5",
            "fn_read_fmt": _fn_read_fmt,
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
        # self.assertIsNone(contig_collection.sf_read_fmt)
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
        contig_collection = RaggedArrayFiles.from_product_id(
            self.tempdir_path / "contiguous",
            product_id="sig0_12.5",
        )
        self.assertEqual(contig_collection.fn_read_fmt(2588), {"cell_id": "2588"})
        # self.assertIsNone(contig_collection.sf_read_fmt)
        self.assertEqual(contig_collection.root_path, self.tempdir_path / "contiguous")
        self.assertEqual(contig_collection.cls, RaggedArrayCell)

    def test_search(self):
        root_path = self.tempdir_path / "contiguous"
        contig_collection = RaggedArrayFiles.from_product_id(
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
        contig_collection = RaggedArrayFiles.from_product_id(
            root_path,
            product_id="sig0_12.5",
        )
        real_merged = contig_collection.read(cell=[2587, 2588])
        self.assertIsInstance(real_merged, xr.Dataset)

        root_path = self.tempdir_path
        allsats_collection = RaggedArrayFiles.from_product_id(
            root_path,
            product_id="sig0_12.5",
            # all_sats=True,
        )
        real_merged = allsats_collection.read(cell=[2587, 2588],
                                                 fmt_kwargs={"sat": ["[ABC]"]})
        self.assertIsNone(real_merged)

    def test_convert_dir_to_contiguous(self):
        root_path = self.tempdir_path / "indexed"
        indexed_collection = RaggedArrayFiles.from_product_id(
            root_path,
            product_id="sig0_12.5",
        )
        converted_dir = self.tempdir_path / "converted_contiguous"
        converted_dir.mkdir(parents=True, exist_ok=True)
        indexed_collection.convert_dir_to_contiguous(converted_dir, num_processes=None)

        converted_collection = RaggedArrayFiles.from_product_id(
            converted_dir,
            product_id="sig0_12.5",
        )

        ref_data = indexed_collection.read().load()
        converted_data = converted_collection.read().load()

        xr.testing.assert_equal(ref_data, converted_data)


        # show all files in the converted directory
        # print(list(converted_dir.rglob("*")))



# test adding new cell types used with OrthoMultiArray
class ERA5Cell():
    grid_name = "Era5Land"
    grid_info = None
    grid = None
    fn_format = "{:04d}.nc"
    possible_cells = None
    max_cell = None
    min_cell = None

    @staticmethod
    def fn_read_fmt(cell, sat=None):
        return {"cell_id": f"{cell:04d}"}

    @staticmethod
    def sf_read_fmt(cell, sat=None):
        if sat is None:
            return None
        return {"sat_str": {"sat": sat}}

class GLDASCell():
    grid_name = "GLDAS"
    grid_info = None
    grid = None
    fn_format = "{:04d}.nc"
    possible_cells = None
    max_cell = None
    min_cell = None

    @staticmethod
    def fn_read_fmt(cell, sat=None):
        return {"cell_id": f"{cell:04d}"}

    @staticmethod
    def sf_read_fmt(cell, sat=None):
        if sat is None:
            return None
        return {"sat_str": {"sat": sat}}

class CCI_PassiveCell():
    grid_name = "CCI_PASSIVE"
    grid_info = None
    grid = None
    fn_format = "{:04d}.nc"
    possible_cells = None
    max_cell = None
    min_cell = None

    @staticmethod
    def fn_read_fmt(cell, sat=None):
        return {"cell_id": f"{cell:04d}"}

    @staticmethod
    def sf_read_fmt(cell, sat=None):
        if sat is None:
            return None
        return {"sat_str": {"sat": sat}}

era5_grid = load_grid("tests/ascat_test_data/warp/era5_land_2023/grid.nc")
gldas_grid = load_grid("tests/ascat_test_data/warp/gldas_2023/grid.nc")
cci_passive_grid = load_grid("tests/ascat_test_data/warp/cci_passive_v07.1/grid.nc")

register_cell_grid_reader(ERA5Cell, era5_grid, "ERA5")
register_cell_grid_reader(GLDASCell, gldas_grid, "GLDAS")
register_cell_grid_reader(CCI_PassiveCell, cci_passive_grid, "CCI_PASSIVE")


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
        om_collection = OrthoMultiArrayFiles.from_product_id(
            self.era5_path,
            product_id="ERA5",
        )
        self.assertEqual(om_collection.fn_read_fmt(29), {"cell_id": "0029"})
        self.assertIsNone(om_collection.sf_read_fmt(29))
        self.assertEqual(om_collection.sf_read_fmt(29, "metop_a"), {"sat_str": {"sat": "metop_a"}})
        self.assertEqual(om_collection.root_path, self.era5_path)
        self.assertEqual(om_collection.cls, OrthoMultiCell)

    def test_search(self):
        om_collection = OrthoMultiArrayFiles.from_product_id(
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
        om_collection = OrthoMultiArrayFiles.from_product_id(
            self.era5_path,
            product_id="ERA5",
        )
        real_merged = om_collection.extract(cell=[29, 30])
        self.assertIsInstance(real_merged, xr.Dataset)


if __name__ == "__main__":
    unittest.main()
