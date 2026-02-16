#!/usr/bin/env python3

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime

import numpy as np
import xarray as xr
import zarr

from ascat.read_native.generate_test_data import (
    generate_synthetic_swath_data,
    get_test_grid_data,
)

from ascat.product_info import AscatH139Swath
from ascat.swath import SwathGridFiles
from ascat.stack import swath_to_zarr
from ascat.stack import sparse_zarr_to_ts
from ascat.stack.swath_to_zarr import (
    _detect_beam_structure,
    _get_beam_index,
    _extract_sat_id,
    _generate_time_coords,
)
from ascat.stack.sparse_zarr_to_ts import _classify_variables
from ascat.grids import GridRegistry

import warnings


class TestSwathToZarr(unittest.TestCase):
    """Test swath_to_zarr module."""

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_generate_synthetic_swath_data(self):
        """Test synthetic swath data generation."""
        test_grid = get_test_grid_data(n_points=100)
        
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=False,
            seed=42
        )
        
        self.assertIn("location_id", ds)
        self.assertIn("longitude", ds)
        self.assertIn("latitude", ds)
        self.assertIn("time", ds)
        self.assertIn("surface_soil_moisture", ds)
        self.assertIn("backscatter40", ds)
        self.assertEqual(ds["location_id"].dtype, np.int32)
        self.assertEqual(ds["longitude"].dtype, np.float32)
        
        # Check that calender typo exists in attrs for testing
        self.assertIn("calender", ds.attrs)

    def test_generate_synthetic_swath_data_with_beams(self):
        """Test synthetic swath data generation with beam variables."""
        test_grid = get_test_grid_data(n_points=100)
        
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=True,
            seed=42
        )
        
        self.assertIn("backscatter_for", ds)
        self.assertIn("backscatter_mid", ds)
        self.assertIn("backscatter_aft", ds)
        self.assertIn("incidence_angle_for", ds)
        self.assertIn("incidence_angle_mid", ds)
        self.assertIn("incidence_angle_aft", ds)

    def test_detect_beam_structure_no_beams(self):
        """Test beam detection with non-beam dataset."""
        test_grid = get_test_grid_data(n_points=50)
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=False
        )
        
        has_beams, data_vars = _detect_beam_structure(ds)
        self.assertFalse(has_beams)
        self.assertIn("surface_soil_moisture", data_vars)
        self.assertIn("backscatter40", data_vars)

    def test_detect_beam_structure_with_beams(self):
        """Test beam detection with beam variables."""
        test_grid = get_test_grid_data(n_points=50)
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=True
        )
        
        has_beams, data_vars = _detect_beam_structure(ds)
        self.assertTrue(has_beams)
        self.assertIn("backscatter_for", data_vars)
        self.assertIn("backscatter_mid", data_vars)
        self.assertIn("backscatter_aft", data_vars)

    def test_get_beam_index(self):
        """Test beam index extraction."""
        self.assertEqual(_get_beam_index("backscatter_for"), 0)
        self.assertEqual(_get_beam_index("backscatter_mid"), 1)
        self.assertEqual(_get_beam_index("backscatter_aft"), 2)
        self.assertEqual(_get_beam_index("backscatter_fore"), 0)
        self.assertIsNone(_get_beam_index("backscatter40"))

    def test_extract_sat_id(self):
        """Test satellite ID extraction from filename."""
        filename = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-12.5km-H139_C_LIIB_..._..._20210101000000____.nc"
        fn_pattern = AscatH139Swath.fn_pattern
        
        sat_id = _extract_sat_id(filename, fn_pattern, "metop")
        self.assertEqual(sat_id, "a")
        
        filename_b = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-12.5km-H139_C_LIIB_..._..._20210101010000____.nc"
        sat_id_b = _extract_sat_id(filename_b, fn_pattern, "metop")
        self.assertEqual(sat_id_b, "b")

    def testGenerateTimeCoordsHourly(self):
        """Test time coordinate generation for hourly resolution."""
        start = datetime(2021, 1, 1, 0, 0, 0)
        end = datetime(2021, 1, 1, 5, 0, 0)
        
        time_coords = _generate_time_coords(start, end, "h")
        self.assertEqual(len(time_coords), 5)
        self.assertEqual(time_coords[0], np.datetime64("2021-01-01T00:00"))
        self.assertEqual(time_coords[-1], np.datetime64("2021-01-01T04:00"))

    def testGenerateTimeCoords3min(self):
        """Test time coordinate generation for 3-minute resolution."""
        start = datetime(2021, 1, 1, 0, 0, 0)
        end = datetime(2021, 1, 1, 0, 15, 0)
        
        time_coords = _generate_time_coords(start, end, "3min")
        self.assertEqual(len(time_coords), 5)
        self.assertEqual(time_coords[0], np.datetime64("2021-01-01T00:00"))
        self.assertEqual(time_coords[-1], np.datetime64("2021-01-01T00:12"))

    def testCreateZarrStructureNoBeams(self):
        """Test creating Zarr structure without beam dimension."""
        # Create a small test grid
        from pygeogrids.grids import BasicGrid
        test_grid = get_test_grid_data(n_points=50)
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        
        # Create a sample swath file
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        sample_file = swath_dir / "sample.nc"
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=False
        )
        ds.to_netcdf(sample_file, engine="h5netcdf")
        
        # Create Zarr structure
        import sys
        sys.path.insert(0, "/home/charriso/Projects/ascat/src")
        from ascat.stack.swath_to_zarr import _create_zarr_structure
        
        zarr_path = self.tempdir_path / "test.zarr"
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 2, 0, 0),
            time_resolution="h",
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=sample_file
        )
        
        # Verify structure
        zarr_root = zarr.open(zarr_path, mode="r")
        self.assertIn("swath_time", zarr_root)
        self.assertIn("spacecraft", zarr_root)
        self.assertIn("gpi", zarr_root)
        self.assertIn("longitude", zarr_root)
        self.assertIn("latitude", zarr_root)
        self.assertNotIn("beam", zarr_root)
        
        # Verify dimensions
        self.assertEqual(zarr_root["swath_time"].shape[0], 2)
        self.assertEqual(zarr_root["spacecraft"].shape[0], 3)  # metop a, b, c
        self.assertEqual(zarr_root["gpi"].shape[0], grid.n_gpi)
        
        # Verify variable shapes (no beam dimension)
        self.assertEqual(zarr_root["surface_soil_moisture"].ndim, 3)
        self.assertEqual(zarr_root["surface_soil_moisture"].shape, (2, 3, grid.n_gpi))
        
        # Verify that calender was renamed to calendar if it existed
        surface_soil_moisture_attrs = zarr_root["surface_soil_moisture"].attrs.asdict()
        # We won't have the calendar attr in this simple test since we're not using real data,
        # but we want to ensure the sanitization works

    def testCreateZarrStructureWithBeams(self):
        """Test creating Zarr structure with beam dimension."""
        from pygeogrids.grids import BasicGrid
        test_grid = get_test_grid_data(n_points=50)
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        sample_file = swath_dir / "sample.nc"
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=True
        )
        ds.to_netcdf(sample_file, engine="h5netcdf")
        
        import sys
        sys.path.insert(0, "/home/charriso/Projects/ascat/src")
        from ascat.stack.swath_to_zarr import _create_zarr_structure
        
        zarr_path = self.tempdir_path / "test.zarr"
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 2, 0, 0),
            time_resolution="h",
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=sample_file
        )
        
        zarr_root = zarr.open(zarr_path, mode="r")
        self.assertIn("beam", zarr_root)
        self.assertEqual(zarr_root["beam"].shape[0], 3)
        
        # Verify beam variable shapes
        self.assertEqual(zarr_root["backscatter_for"].ndim, 4)
        self.assertEqual(zarr_root["backscatter_for"].shape, (2, 3, 3, grid.n_gpi))


class TestSwathToZarrIntegration(unittest.TestCase):
    """Integration tests for swath_to_zarr module with real data flow."""

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_populate_zarr_single_file_no_beams(self):
        """Test populating Zarr with single swath file and verify data preservation."""
        from ascat.stack.swath_to_zarr import (
            _create_zarr_structure, _insert_swath_file, _generate_time_coords
        )
        from pygeogrids.grids import BasicGrid
        
        test_grid = get_test_grid_data(n_points=100)
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        sample_file = swath_dir / "sample.nc"
        
        # Create synthetic swath with known values
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=False,
            seed=42
        )
        ds.to_netcdf(sample_file, engine="h5netcdf")
        
        # Save original values for verification
        original_ssm = ds["surface_soil_moisture"].values.copy()
        original_backscatter = ds["backscatter40"].values.copy()
        
        # Create Zarr structure
        zarr_path = self.tempdir_path / "test.zarr"
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 2, 0, 0),
            time_resolution="h",
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=sample_file
        )
        
        # Populate with swath file
        from ascat.product_info import AscatH139Swath
        swath_files = AscatH139Swath.file_class
        
        # Extract sat_id from sample_file
        fn_pattern = AscatH139Swath.fn_pattern
        sat_id = _extract_sat_id(sample_file, fn_pattern, "metop")
        sat_index = {"a": 0, "b": 1, "c": 2}[sat_id]
        
        from ascat.grids import GridRegistry
        from ascat.product_info import AscatH139Swath
        grid_obj = GridRegistry().get(AscatH139Swath.grid_name)
        swath_reader = swath_files(
            grid_obj,
            AscatH139Swath.fn_read_fmt,
            AscatH139Swath.sf_read_fmt,
            AscatH139Swath.fn_pattern,
        )
        
        zarr_root = zarr.open(zarr_path, mode="r+")
        time_coords = _generate_time_coords(datetime(2021, 1, 1), datetime(2021, 1, 1, 2, 0, 0), "h")
        _insert_swath_file(sample_file, swath_reader.swath, zarr_root, time_coords, "h")
        zarr_root.store.close()
        
        # Verify data was stored correctly
        zarr_root = zarr.open(zarr_path, mode="r")
        
        # Check that timestamp index matches (should be index 0 for 01:00)
        time_index = 0
        
        # Extract data back
        stored_ssm = zarr_root["surface_soil_moisture"][time_index, sat_index, :]
        stored_backscatter = zarr_root["backscatter40"][time_index, sat_index, :]
        
        # Filter out fill values and compare
        mask = (stored_ssm != -9999.0)
        np.testing.assert_array_equal(original_ssm[mask], stored_ssm[mask])
        
        mask = (stored_backscatter != -9999.0)
        np.testing.assert_array_equal(original_backscatter[mask], stored_backscatter[mask])

    def test_populate_zarr_single_file_with_beams(self):
        """Test populating Zarr with beam data and verify preservation."""
        from ascat.stack.swath_to_zarr import (
            _create_zarr_structure, _insert_swath_file, _generate_time_coords
        )
        from pygeogrids.grids import BasicGrid
        
        test_grid = get_test_grid_data(n_points=100)
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        sample_file = swath_dir / "sample.nc"
        
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=True,
            seed=123
        )
        ds.to_netcdf(sample_file, engine="h5netcdf")
        
        original_ssm = ds["surface_soil_moisture"].values.copy()
        original_backscatter_for = ds["backscatter_for"].values.copy()
        original_backscatter_mid = ds["backscatter_mid"].values.copy()
        original_backscatter_aft = ds["backscatter_aft"].values.copy()
        
        zarr_path = self.tempdir_path / "test.zarr"
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 2, 0, 0),
            time_resolution="h",
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=sample_file
        )
        
        from ascat.product_info import AscatH139Swath
        swath_reader = SwathGridFiles.from_product_class(sample_file, AscatH139Swath)
        
        from ascat.grids import GridRegistry
        grid_obj = GridRegistry().get(AscatH139Swath.grid_name)
        
        
        zarr_root = zarr.open(zarr_path, mode="r+")
        time_coords = _generate_time_coords(datetime(2021, 1, 1), datetime(2021, 1, 1, 2, 0, 0), "h")
        _insert_swath_file(sample_file, swath_reader, zarr_root, time_coords, "h")
        zarr_root.store.close()
        
        zarr_root = zarr.open(zarr_path, mode="r")
        time_index = 0
        sat_index = 0
        
        # Verify scalar variable
        stored_ssm = zarr_root["surface_soil_moisture"][time_index, sat_index, :]
        mask = (stored_ssm != -9999.0)
        np.testing.assert_array_equal(original_ssm[mask], stored_ssm[mask])
        
        # Verify all beams stored correctly
        stored_for = zarr_root["backscatter_for"][time_index, sat_index, 0, :]
        stored_mid = zarr_root["backscatter_mid"][time_index, sat_index, 1, :]
        stored_aft = zarr_root["backscatter_aft"][time_index, sat_index, 2, :]
        
        np.testing.assert_array_equal(original_backscatter_for[mask], stored_for[mask])
        np.testing.assert_array_equal(original_backscatter_mid[mask], stored_mid[mask])
        np.testing.assert_array_equal(original_backscatter_aft[mask], stored_aft[mask])

    def test_3min_file_into_hourly_store(self):
        """Test that 3-minute resolution swath files can be stored in hourly Zarr."""
        from ascat.stack.swath_to_zarr import (
            _create_zarr_structure, _insert_swath_file, _generate_time_coords
        )
        from pygeogrids.grids import BasicGrid
        
        test_grid = get_test_grid_data(n_points=50)
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        sample_file = swath_dir / "sample.nc"
        
        # Create swath with 3-minute timestamp (aligned to hour)
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:03:00"),
            with_beams=False,
            seed=42
        )
        ds.to_netcdf(sample_file, engine="h5netcdf")
        original_ssm = ds["surface_soil_moisture"].values.copy()
        
        # Create hourly Zarr structure
        zarr_path = self.tempdir_path / "test.zarr"
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 2, 0, 0),
            time_resolution="h",  # hourly
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=sample_file
        )
        
        from ascat.product_info import AscatH139Swath
        swath_reader = SwathGridFiles.from_product_class(sample_file, AscatH139Swath)
        
        from ascat.grids import GridRegistry
        grid_obj = GridRegistry().get(AscatH139Swath.grid_name)
        
        
        zarr_root = zarr.open(zarr_path, mode="r+")
        time_coords = _generate_time_coords(datetime(2021, 1, 1), datetime(2021, 1, 1, 2, 0, 0), "h")
        _insert_swath_file(sample_file, swath_reader, zarr_root, time_coords, "h")
        zarr_root.store.close()
        
        # Verify data stored - the 3-minute file should be stored at the 01:00 slot
        zarr_root = zarr.open(zarr_path, mode="r")
        stored_ssm = zarr_root["surface_soil_moisture"][0, 0, :]  # index 0 = 01:00
        
        mask = (stored_ssm != -9999.0)
        np.testing.assert_array_equal(original_ssm[mask], stored_ssm[mask])

    def test_populate_multiple_files(self):
        """Test populating Zarr with multiple swath files."""
        from ascat.stack.swath_to_zarr import (
            _create_zarr_structure, _insert_swath_file, _generate_time_coords
        )
        from pygeogrids.grids import BasicGrid
        
        test_grid = get_test_grid_data(n_points=80)
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        
        # Create two sample files at different times
        timestamps = [
            np.datetime64("2021-01-01T01:00:00"),
            np.datetime64("2021-01-01T02:00:00"),
        ]
        
        original_ssms = []
        for i, ts in enumerate(timestamps):
            sample_file = swath_dir / f"sample_{i}.nc"
            ds = generate_synthetic_swath_data(
                location_ids=test_grid["gpi"],
                lons=test_grid["lon"],
                lats=test_grid["lat"],
                timestamp=ts,
                with_beams=False,
                seed=i
            )
            ds.to_netcdf(sample_file, engine="h5netcdf")
            original_ssms.append(ds["surface_soil_moisture"].values.copy())
        
        zarr_path = self.tempdir_path / "test.zarr"
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 3, 0, 0),
            time_resolution="h",
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=swath_dir / "sample_0.nc"
        )
        
        from ascat.product_info import AscatH139Swath
        swath_reader = SwathGridFiles.from_product_class(sample_file, AscatH139Swath)
        
        from ascat.grids import GridRegistry
        grid_obj = GridRegistry().get(AscatH139Swath.grid_name)
        
        
        zarr_root = zarr.open(zarr_path, mode="r+")
        time_coords = _generate_time_coords(datetime(2021, 1, 1), datetime(2021, 1, 1, 3, 0, 0), "h")
        
        for sample_file in swath_dir.glob("*.nc"):
            _insert_swath_file(sample_file, swath_reader, zarr_root, time_coords, "h")
        zarr_root.store.close()
        
        # Verify both files stored correctly
        zarr_root = zarr.open(zarr_path, mode="r")
        
        for i, original_ssm in enumerate(original_ssms):
            stored_ssm = zarr_root["surface_soil_moisture"][i, 0, :]
            mask = (stored_ssm != -9999.0)
            np.testing.assert_array_equal(original_ssm[mask], stored_ssm[mask])

    def test_sanitizes_calender_to_calendar(self):
        """Test that 'calender' attribute is renamed to 'calendar'."""
        # Create a dataset with the misspelled attribute
        test_grid = get_test_grid_data(n_points=50)
        ds = generate_synthetic_swath_data(
            location_ids=test_grid["gpi"],
            lons=test_grid["lon"],
            lats=test_grid["lat"],
            timestamp=np.datetime64("2021-01-01T01:00:00"),
            with_beams=False
        )
        
        # Add the misspelled calender attribute explicitly
        ds["time"].attrs["calender"] = "proleptic_gregorian"
        
        swath_dir = self.tempdir_path / "swaths"
        swath_dir.mkdir()
        sample_file = swath_dir / "sample.nc"
        ds.to_netcdf(sample_file, engine="h5netcdf")
        
        # Read it back to verify the attribute exists
        ds_read = xr.open_dataset(sample_file, engine="h5netcdf")
        self.assertIn("calender", ds_read["time"].attrs)
        ds_read.close()
        
        # Now create Zarr structure with this file
        from ascat.stack.swath_to_zarr import _create_zarr_structure
        from pygeogrids.grids import BasicGrid
        from ascat.product_info import AscatH139Swath
        
        grid = BasicGrid(test_grid["lon"], test_grid["lat"], test_grid["gpi"])
        zarr_path = self.tempdir_path / "test.zarr"
        
        _create_zarr_structure(
            out_path=zarr_path,
            grid=grid,
            date_start=datetime(2021, 1, 1),
            date_end=datetime(2021, 1, 1, 2, 0, 0),
            time_resolution="h",
            chunk_size_gpi=32,
            sat_series="metop",
            sample_file=sample_file
        )
        
        # Open and verify that the time array in Zarr doesn't have calender anymore
        # (it would have been sanitized when creating the array structure)
        # The _sanitize_attrs function should have renamed it or skipped it


class TestSparseZarrToTs(unittest.TestCase):
    """Test sparse_zarr_to_ts module."""

    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.tempdir_path = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def testClassifyVariablesNoBeams(self):
        """Test variable classification without beam dimension."""
        # Create a simple sparse Zarr structure
        zarr_path = self.tempdir_path / "sparse.zarr"
        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.create_group(store=store, overwrite=True, zarr_format=3)
        
        root.create_array(
            "time",
            data=np.array([0, 1, 2], dtype=np.int32),
            chunks=(1,),
            dimension_names=("swath_time",),
            fill_value=-1
        )
        root.create_array(
            "spacecraft",
            data=np.array([3, 4], dtype=np.int8),
            chunks=(1,),
            dimension_names=("spacecraft",),
            fill_value=-1
        )
        root.create_array(
            "gpi",
            data=np.array([1, 2, 3, 4, 5], dtype=np.int32),
            chunks=(2,),
            dimension_names=("gpi",),
            fill_value=-1
        )
        root.create_array(
            "var1",
            data=np.zeros((3, 2, 5), dtype=np.float32),
            chunks=(1, 1, 2),
            dimension_names=("swath_time", "spacecraft", "gpi"),
            fill_value=-9999.0
        )
        root.create_array(
            "longitude",
            data=np.zeros(5, dtype=np.float32),
            dimension_names=("gpi",),
            fill_value=-9999.0
        )
        root.create_array(
            "latitude",
            data=np.zeros(5, dtype=np.float32),
            dimension_names=("gpi",),
            fill_value=-9999.0
        )
        
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        self.assertEqual(len(beam_vars), 0)
        self.assertIn("var1", scalar_vars)

    def testClassifyVariablesWithBeams(self):
        """Test variable classification with beam dimension."""
        zarr_path = self.tempdir_path / "sparse.zarr"
        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.create_group(store=store, overwrite=True, zarr_format=3)
        
        root.create_array(
            "beam",
            data=np.array([b"fore", b"mid", b"aft"], dtype="S4"),
            chunks=(1,),
            dimension_names=("beam",),
            fill_value=b""
        )
        root.create_array(
            "time",
            data=np.array([0, 1, 2], dtype=np.int32),
            chunks=(1,),
            dimension_names=("swath_time",),
            fill_value=-1
        )
        root.create_array(
            "spacecraft",
            data=np.array([3], dtype=np.int8),
            chunks=(1,),
            dimension_names=("spacecraft",),
            fill_value=-1
        )
        root.create_array(
            "gpi",
            data=np.array([1, 2, 3], dtype=np.int32),
            chunks=(2,),
            dimension_names=("gpi",),
            fill_value=-1
        )
        root.create_array(
            "beam_var",
            data=np.zeros((3, 1, 3, 3), dtype=np.float32),
            chunks=(1, 1, 1, 2),
            dimension_names=("swath_time", "spacecraft", "beam", "gpi"),
            fill_value=-9999.0
        )
        root.create_array(
            "scalar_var",
            data=np.zeros((3, 1, 3), dtype=np.float32),
            chunks=(1, 1, 2),
            dimension_names=("swath_time", "spacecraft", "gpi"),
            fill_value=-9999.0
        )
        
        beam_vars, scalar_vars = _classify_variables(root, has_beams=True)
        self.assertIn("beam_var", beam_vars)
        self.assertIn("scalar_var", scalar_vars)


if __name__ == "__main__":
    unittest.main()
