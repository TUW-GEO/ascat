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

from ascat.read_native import xarray_ext as xae
from ascat.read_native.product_info import cell_io_catalog
from ascat.read_native.product_info import grid_cache
from ascat.read_native.swath_collection import SwathGridFiles
from ascat.read_native.cell_collection import RaggedArrayFiles


class TestSwathAccessor(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        all_files = list(Path(swath_path).rglob("*.nc"))
        first_file = all_files[0]
        ds = xr.open_dataset(first_file)
        print(ds)
        with self.assertRaises(ValueError):
            ds.swath

        ds.attrs["grid_name"] = cell_io_catalog["H129"].grid_name
        ds.swath


    def test_sel_spatial(self):
        swath_path = "tests/ascat_test_data/hsaf/h129/swaths"
        sf = SwathGridFiles.from_product_id(swath_path, "h129")
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
            # date_field_fmt="%Y%m%d%H%M%S"
        )
        merged_ds.load()

        # test with cell
        cell_sel = merged_ds.swath.sel_spatial(cell=1296)
        cell_sel_cells = cell_sel.swath.grid.gpi2cell(cell_sel["location_id"].values)
        np.testing.assert_array_equal(
            np.array([1296]),
            np.unique(cell_sel_cells)
        )

        cells_sel = merged_ds.swath.sel_spatial(cell=[871, 1296])
        cells_sel_cells = cells_sel.swath.grid.gpi2cell(cells_sel["location_id"].values)
        np.testing.assert_array_equal(
            np.array([871, 1296]),
            np.unique(cells_sel_cells)
        )

        # test with location_id
        gpi_sel = merged_ds.swath.sel_spatial(
            location_id=6600008
        )
        np.testing.assert_array_equal(
            np.unique(gpi_sel["location_id"].values),
            np.array([6600008])
        )

        gpis_sel = merged_ds.swath.sel_spatial(
            location_id=[6600008, 7982189]
        )
        np.testing.assert_array_equal(
            np.unique(gpis_sel["location_id"].values),
            np.array([6600008, 7982189])
        )


        # test with coords
        coord = (0.17, -89.9)
        coords_list = ([0.2, -30, -59], [-90, -70, -53])
        coord_sel = merged_ds.swath.sel_spatial(coords=coord)
        np.testing.assert_array_equal(
            np.unique(coord_sel["location_id"].values),
            np.array([6600008])
        )

        coords_sel = merged_ds.swath.sel_spatial(
            coords=coords_list
        )
        np.testing.assert_array_equal(
            np.unique(coords_sel["location_id"].values),
            np.array([7003497, 7940379])
        )

        coords_sel_limited = merged_ds.swath.sel_spatial(
            coords=coords_list,
            max_coord_dist=100
        )
        np.testing.assert_array_equal(
            np.unique(coords_sel_limited["location_id"].values),
            np.array([])
        )

        # test with bbox
        smaller_bbox = (-90, -30, -30, 10)
        small_sel = merged_ds.swath.sel_spatial(bbox=smaller_bbox)
        self.assertGreater(small_sel["latitude"].min(), smaller_bbox[0])
        self.assertLess(small_sel["latitude"].max(), smaller_bbox[1])
        self.assertGreater(small_sel["longitude"].min(), smaller_bbox[2])
        self.assertLess(small_sel["longitude"].max(), smaller_bbox[3])

        # test with geometry once it's implemented

class TestIndexedRaggedAccessor(unittest.TestCase):

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        cell_path = "tests/ascat_test_data/hsaf/h129/stack_cells"
        all_files = list(Path(cell_path).rglob("*.nc"))
        first_file = all_files[0]
        ds = xr.open_dataset(first_file)
        with self.assertRaises(ValueError):
            ds.idx_ragged

        ds.attrs["grid_name"] = cell_io_catalog["H129"].grid_name
        ds.idx_ragged

    def test_sel_spatial(self):
        cell_path = "tests/ascat_test_data/hsaf/h129/stack_cells"
        ra = RaggedArrayFiles(cell_path, "h129")
        bbox=(58, 70, -179, -170)

        merged_ds = ra.extract(
            # bbox=bbox,
            cell=[30, 31],
            # date_range=(np.datetime64(datetime(2021, 1, 15)),
            # np.datetime64(datetime(2021, 1, 30))),
            # date_field_fmt="%Y%m%d%H%M%S"
        )
        merged_ds.load()
        print(merged_ds)

        cell_sel = merged_ds.idx_ragged.sel_spatial(cell=31)
        print(cell_sel)

        cells_sel = merged_ds.idx_ragged.sel_spatial(cell=[30, 31, 29])
        print(cells_sel)

        bbox_sel = merged_ds.idx_ragged.sel_spatial(bbox=bbox)
        print(bbox_sel)

    def test_extract_spatial(self):
        cell_path = "tests/ascat_test_data/hsaf/h129/stack_cells"
        ra = RaggedArrayFiles(cell_path, "h129")
        bbox=(58, 70, -179, -170)

        merged_ds = ra.extract(
            # bbox=bbox,
            cell=[30, 31],
            # date_range=(np.datetime64(datetime(2021, 1, 15)),
            # np.datetime64(datetime(2021, 1, 30))),
            # date_field_fmt="%Y%m%d%H%M%S"
        )
        merged_ds.load()
        # print(merged_ds)

        cell_ext = merged_ds.idx_ragged.extract_spatial(cell=31)
        # print(cell_ext)

        cells_ext = merged_ds.idx_ragged.extract_spatial(cell=[30, 31, 29])
        # print(cells_ext)

        bbox_ext = merged_ds.idx_ragged.extract_spatial(bbox=bbox)
        # print(bbox_ext)


class TestOrthoMultiAccessor(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)
        # gen_dummy_swathfiles(self.tempdir_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_init(self):
        cell_path = "tests/ascat_test_data/warp/era5_land_2023"
        all_files = list(Path(cell_path).rglob("*.nc"))
        first_file = all_files[0]
        ds = xr.open_dataset(first_file)
        with self.assertRaises(ValueError):
            ds.orthomulti

        ds.attrs["grid_name"] = cell_io_catalog["H129"].grid_name
        ds.orthomulti
        return


