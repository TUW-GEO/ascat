#!/usr/bin/env python3

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr
import numpy as np


from ascat.read_native import generate_test_data as gtd

# from ascat.read_native.product_info import cell_io_catalog

from ascat.accessors import *
from ascat.cell import RaggedArrayTs


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
    add_sat_id(gtd.contiguous_ragged_ds_2588, sat_name).to_netcdf(
        contiguous_dir / "2588.nc"
    )
    add_sat_id(gtd.indexed_ragged_ds_2588, sat_name).to_netcdf(indexed_dir / "2588.nc")
    add_sat_id(gtd.contiguous_ragged_ds_2587, sat_name).to_netcdf(
        contiguous_dir / "2587.nc"
    )
    add_sat_id(gtd.indexed_ragged_ds_2587, sat_name).to_netcdf(indexed_dir / "2587.nc")


class TestCFDiscreteGeometryAccessor(unittest.TestCase):
    @staticmethod
    def _init_options(root_path, sf_templ=None, sf_read_fmt=None):
        return {
            "root_path": root_path,
            "file_class": RaggedArrayTs,
            "fn_format": "{:04d}.nc",
            # "sf_format": sf_templ,
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

    def test_array_type(self):
        ds = xr.open_dataset(self.tempdir_path / "contiguous" / "2588.nc")

        ds.attrs["grid_mapping_name"] = "fibgrid_12.5"
        ds["row_size"] = ds.row_size.assign_attrs({"sample_dimension": "time"})
        ds["location_id"] = ds["location_id"].assign_attrs({"cf_role": "timeseries_id"})

        # ds = ds.set_coords(["lat", "lon", "alt"])
        # ds.cf_geom.set_coord_vars(["lat", "lon", "alt"])
        # assert ds.cf_geom._coord_vars == ["lat", "lon", "alt"]

        assert ds.cf_geom.array_type == "contiguous"

        # to indexed_ragged
        ids = ds.cf_geom.to_indexed_ragged()
        assert ids.cf_geom.array_type == "indexed"

        # to indexed_ragged with different index_var name
        ids2 = ds.copy().cf_geom.to_indexed_ragged(index_var="BIGTEST")
        assert ids2.cf_geom.array_type == "indexed"

        # now back to_contiguous_ragged
        cds = ids2.cf_geom.to_contiguous_ragged()
        assert cds.cf_geom.array_type == "contiguous"
        xr.testing.assert_identical(cds, ds)

        # now back to indexed
        ids3 = cds.cf_geom.to_indexed_ragged()
        assert ids3.cf_geom.array_type == "indexed"
        assert ids3.cf_geom._obj._index_var != "BIGTEST"

        # original to point array
        pads = ds.cf_geom.to_point_array()
        assert pads.cf_geom.array_type == "point"


class TestPyGeoGriddedArrayAccessor(unittest.TestCase):
    @staticmethod
    def _init_options(root_path, sf_templ=None, sf_read_fmt=None):
        return {
            "root_path": root_path,
            "file_class": RaggedArrayTs,
            "fn_format": "{:04d}.nc",
            # "sf_format": sf_templ,
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

    def test_array_type(self):
        ds = xr.open_dataset(self.tempdir_path / "contiguous" / "2588.nc")

        ds.attrs["grid_mapping_name"] = "fibgrid_12.5"
        ds["row_size"] = ds.row_size.assign_attrs({"sample_dimension": "time"})
        ds["location_id"] = ds["location_id"].assign_attrs({"cf_role": "timeseries_id"})


        # select from original contiguous
        gpi_selection = ds.pgg.sel_gpis([1549346, 1556056])
        assert gpi_selection.cf_geom.array_type == ds.cf_geom.array_type
        np.testing.assert_array_equal(
            gpi_selection["location_id"].values, [1549346, 1556056]
        )

        # select from indexed
        idx_gpi_selection = (
            ds.load().cf_geom.to_indexed_ragged().pgg.sel_gpis([1549346, 1556056])
        )
        assert idx_gpi_selection.cf_geom.array_type == "indexed"
        xr.testing.assert_identical(
            idx_gpi_selection.cf_geom.to_point_array(),
            gpi_selection.cf_geom.to_point_array(),
        )

        # select from point array
        point_gpi_selection = ds.cf_geom.to_point_array().pgg.sel_gpis(
            [1549346, 1556056]
        )
        assert point_gpi_selection.cf_geom.array_type == "point"

        # contig_ra = RaggedArray(ds)
        # self.assertEqual(contig_ra.array_type, "contiguous")
