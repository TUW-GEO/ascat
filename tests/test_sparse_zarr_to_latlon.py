"""
Test suite for ascat.regrid.sparse_zarr_pyramids (regrid_to_latlon)

Tests are organized around the main behaviors of regrid_to_latlon and its
internal helpers.  All tests use synthetic sparse Zarr stores and mock grid
objects — no real FibGrid KDTree or NetCDF files required.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from ascat.regrid.sparse_zarr_pyramids import (
    _build_regular_grid,
    _classify_variables,
    _compute_nn_lookup,
    _create_level_arrays,
    _create_pyramid_store,
    _downsample_coords,
    _find_pending_slices,
    _gaussian_downsample_2d,
    _maybe_expand_pyramid_swath_time,
    regrid_to_latlon,
)
from conftest import (
    CHUNK_SIZE_GPI,
    N_GPI,
    N_SPACECRAFT,
    N_SWATH_TIME,
)

# Coarse resolution for fast tests
TEST_RESOLUTION = 5.0     # degrees — gives a ~36x72 grid
TEST_LAT_CHUNK = 4
TEST_LON_CHUNK = 4
TEST_MAX_DIST = 600_000   # 600 km — generous, ensures all cells get a GPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse_store_for_pyramid(tmp_path, populated_slots=None, var_name="surface_soil_moisture", n_swath_time=N_SWATH_TIME):
    """Minimal sparse store compatible with regrid_to_latlon."""
    from zarr.codecs import BloscCodec, BloscShuffle

    path = tmp_path
    store = zarr.storage.LocalStore(path)
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    shape = (n_swath_time, N_SPACECRAFT, N_GPI)
    inner_chunks = (1, 1, CHUNK_SIZE_GPI)
    blosc = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)

    root.create_array(
        name=var_name, dtype="float32", shape=shape, chunks=inner_chunks,
        dimension_names=("swath_time", "spacecraft", "gpi"),
        fill_value=np.float32(-9999.0), compressors=blosc,
    )
    root.create_array(
        name="time", dtype="float64", shape=shape, chunks=inner_chunks,
        dimension_names=("swath_time", "spacecraft", "gpi"),
        fill_value=np.float64(0.0), compressors=blosc,
    )
    root.create_array(
        "gpi", data=np.arange(N_GPI, dtype="int32"),
        chunks=(CHUNK_SIZE_GPI,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "longitude", data=np.linspace(-180, 180, N_GPI, dtype="float32"),
        chunks=(CHUNK_SIZE_GPI,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "latitude", data=np.linspace(-90, 90, N_GPI, dtype="float32"),
        chunks=(CHUNK_SIZE_GPI,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "swath_time", data=np.arange(n_swath_time, dtype="int64"),
        chunks=(1,), dimension_names=("swath_time",), compressors=None,
    )
    root.create_array(
        "spacecraft", data=np.array([3, 4], dtype="int8")[:N_SPACECRAFT],
        chunks=(1,), dimension_names=("spacecraft",), compressors=None,
    )
    root.create_array(
        "processed", shape=(n_swath_time, N_SPACECRAFT), dtype="bool",
        chunks=(1, N_SPACECRAFT), dimension_names=("swath_time", "spacecraft"),
        fill_value=False, compressors=None,
    )

    if populated_slots:
        for t_idx, s_idx, gpi_slice, values in populated_slots:
            root[var_name][t_idx, s_idx, gpi_slice] = values
            n = len(values)
            root["time"][t_idx, s_idx, gpi_slice] = np.full(n, float(t_idx + 1))
            root["processed"][t_idx, s_idx] = True

    return path


def _make_grid_mock(n_gpi=N_GPI):
    """Minimal grid mock that supports get_grid_points and find_nearest_gpi."""
    lats = np.linspace(-90.0, 90.0, n_gpi)
    lons = np.linspace(-180.0, 180.0, n_gpi)
    gpis = np.arange(n_gpi, dtype="int32")

    grid = MagicMock()
    grid.n_gpi = n_gpi
    grid.get_grid_points.return_value = (gpis, lons, lats, np.zeros(n_gpi, dtype="int32"))

    def _find_nearest(query_lons, query_lats):
        lons_arr = np.asarray(query_lons)
        idx = np.clip(
            np.round((lons_arr + 180.0) / 360.0 * (n_gpi - 1)).astype(int),
            0, n_gpi - 1,
        )
        dist = np.abs(lons_arr - lons[idx]) * 111_000.0
        return gpis[idx], dist.astype("float32")

    grid.find_nearest_gpi.side_effect = _find_nearest
    return grid


# ===========================================================================
# _build_regular_grid
# ===========================================================================

class TestBuildRegularGrid:

    def test_latitude_range(self):
        lats, lons = _build_regular_grid(1.0)
        assert lats[0] < 90.0
        assert lats[-1] > -90.0
        assert lats[0] > lats[-1]   # descending (north to south)

    def test_longitude_range(self):
        lats, lons = _build_regular_grid(1.0)
        assert lons[0] > -180.0
        assert lons[-1] < 180.0

    def test_step_size_matches_resolution(self):
        lats, lons = _build_regular_grid(2.0)
        lat_steps = np.diff(lats)
        assert np.allclose(lat_steps, -2.0)
        lon_steps = np.diff(lons)
        assert np.allclose(lon_steps, 2.0)

    def test_coarser_grid_has_fewer_points(self):
        lats_fine, lons_fine = _build_regular_grid(0.5)
        lats_coarse, lons_coarse = _build_regular_grid(1.0)
        assert len(lats_fine) > len(lats_coarse)


# ===========================================================================
# _compute_nn_lookup
# ===========================================================================

class TestComputeNNLookup:

    def test_output_shapes(self):
        grid = _make_grid_mock()
        lats, lons = _build_regular_grid(TEST_RESOLUTION)
        nn_idx, nn_mask = _compute_nn_lookup(grid, lats, lons, TEST_RESOLUTION, TEST_MAX_DIST)
        assert nn_idx.shape == (len(lats), len(lons))
        assert nn_mask.shape == (len(lats), len(lons))

    def test_valid_mask_is_boolean(self):
        grid = _make_grid_mock()
        lats, lons = _build_regular_grid(TEST_RESOLUTION)
        _, nn_mask = _compute_nn_lookup(grid, lats, lons, TEST_RESOLUTION, TEST_MAX_DIST)
        assert nn_mask.dtype == bool

    def test_distant_cells_masked(self):
        """With a very small max_dist, many cells should be masked out."""
        grid = _make_grid_mock()
        lats, lons = _build_regular_grid(TEST_RESOLUTION)
        _, nn_mask = _compute_nn_lookup(grid, lats, lons, TEST_RESOLUTION, max_dist_m=1.0)
        assert nn_mask.sum() < nn_mask.size

    def test_all_valid_within_generous_dist(self):
        grid = _make_grid_mock()
        lats, lons = _build_regular_grid(TEST_RESOLUTION)
        _, nn_mask = _compute_nn_lookup(grid, lats, lons, TEST_RESOLUTION, TEST_MAX_DIST)
        # With generous distance most cells should be valid
        assert nn_mask.sum() > nn_mask.size * 0.5


# ===========================================================================
# _gaussian_downsample_2d
# ===========================================================================

class TestGaussianDownsample2d:

    def test_output_is_half_input_size(self):
        data = np.random.rand(20, 40).astype("float32")
        result = _gaussian_downsample_2d(data, fill_val=-9999.0, sigma=1.0)
        assert result.shape == (10, 20)

    def test_fill_value_regions_preserved(self):
        data = np.ones((20, 20), dtype="float32")
        data[:, :] = -9999.0  # all fill
        result = _gaussian_downsample_2d(data, fill_val=-9999.0, sigma=1.0)
        assert (result == -9999.0).all()

    def test_valid_values_smoothed(self):
        """A constant non-fill field should downsample to the same constant."""
        data = np.full((20, 20), 5.0, dtype="float32")
        result = _gaussian_downsample_2d(data, fill_val=-9999.0, sigma=1.0)
        np.testing.assert_allclose(result, 5.0, atol=1e-4)

    def test_output_dtype_preserved(self):
        data = np.ones((20, 20), dtype="float32")
        result = _gaussian_downsample_2d(data, fill_val=-9999.0, sigma=1.0)
        assert result.dtype == np.float32

    def test_integer_dtype_rounds(self):
        data = np.full((20, 20), 3, dtype="int16")
        result = _gaussian_downsample_2d(data, fill_val=-9999, sigma=1.0)
        assert result.dtype == np.int16
        assert (result == 3).all()


# ===========================================================================
# _downsample_coords
# ===========================================================================

class TestDownsampleCoords:

    def test_output_length(self):
        coords = np.arange(20, dtype="float64")
        result = _downsample_coords(coords, scale=2)
        assert len(result) == 10

    def test_mean_correct(self):
        coords = np.array([0.0, 1.0, 2.0, 3.0])
        result = _downsample_coords(coords, scale=2)
        np.testing.assert_allclose(result, [0.5, 2.5])


# ===========================================================================
# _find_pending_slices
# ===========================================================================

class TestFindPendingSlices:

    def test_returns_only_unprocessed_in_pyramid(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(
            tmp_path / "sp",
            populated_slots=[(0, 0, slice(0, 5), np.ones(5, dtype="f4"))],
        )
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        # Build pyramid so we have a real output store
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION,
            max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1,
            lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
            n_workers=1,
        )
        # Now sparse has slot (0,0) processed; pyramid level 0 should too
        all_slices = [(t, s) for t in range(N_SWATH_TIME) for s in range(N_SPACECRAFT)]
        pending = _find_pending_slices(out_path, sparse_path, all_slices)
        # (0,0) is processed in both — should not be pending
        assert (0, 0) not in pending

    def test_unpopulated_sparse_slots_not_pending(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION,
            max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1,
            lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
            n_workers=1,
        )
        all_slices = [(t, s) for t in range(N_SWATH_TIME) for s in range(N_SPACECRAFT)]
        pending = _find_pending_slices(out_path, sparse_path, all_slices)
        assert len(pending) == 0


# ===========================================================================
# _maybe_expand_pyramid_swath_time
# ===========================================================================

class TestMaybeExpandPyramidSwathTime:

    def test_no_op_when_same_size(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=2, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
            n_workers=1,
        )
        sparse_root = zarr.open(str(sparse_path), mode="r")
        before = zarr.open(str(out_path), mode="r")["0"]["surface_soil_moisture"].shape[0]
        _maybe_expand_pyramid_swath_time(out_path, sparse_root)
        after = zarr.open(str(out_path), mode="r")["0"]["surface_soil_moisture"].shape[0]
        assert before == after

    def test_expansion_all_levels_resized(self, tmp_path):
        """When sparse store grows, all pyramid levels should expand."""
        # Build initial pyramid with N_SWATH_TIME slots
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=2, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
            n_workers=1,
        )

        # Build a larger sparse store (more swath_time slots)
        larger_sparse_path = _make_sparse_store_for_pyramid(
            tmp_path / "sp2", n_swath_time=N_SWATH_TIME + 2
        )
        # Rename to match the original path (simulate the sparse store growing)
        larger_sparse_root = zarr.open(str(larger_sparse_path), mode="r")

        _maybe_expand_pyramid_swath_time(out_path, larger_sparse_root)

        out_root = zarr.open(str(out_path), mode="r")
        for level in range(2):
            arr = out_root[str(level)]["surface_soil_moisture"]
            assert arr.shape[0] == N_SWATH_TIME + 2

    def test_swath_time_coords_updated(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
            n_workers=1,
        )
        larger_sparse_path = _make_sparse_store_for_pyramid(
            tmp_path / "sp2", n_swath_time=N_SWATH_TIME + 2
        )
        larger_sparse_root = zarr.open(str(larger_sparse_path), mode="r")
        _maybe_expand_pyramid_swath_time(out_path, larger_sparse_root)

        out_root = zarr.open(str(out_path), mode="r")
        assert out_root["0"]["swath_time"].shape[0] == N_SWATH_TIME + 2


# ===========================================================================
# regrid_to_latlon integration
# ===========================================================================

class TestRegridToLatlon:

    def test_creates_output_store(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        assert (out_path / "zarr.json").exists()

    def test_multiscales_metadata_present(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        root = zarr.open(str(out_path), mode="r")
        assert "multiscales" in root.attrs

    def test_correct_number_of_pyramid_levels(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=3, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        root = zarr.open(str(out_path), mode="r")
        for level in range(3):
            assert str(level) in root

    def test_level0_grid_shape_correct(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        lats, lons = _build_regular_grid(TEST_RESOLUTION)
        root = zarr.open(str(out_path), mode="r")
        arr = root["0"]["surface_soil_moisture"]
        assert arr.shape[2] == len(lats)
        assert arr.shape[3] == len(lons)

    def test_level1_grid_is_half_level0(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=2, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        root = zarr.open(str(out_path), mode="r")
        shape0 = root["0"]["surface_soil_moisture"].shape
        shape1 = root["1"]["surface_soil_moisture"].shape
        # Level 1 spatial dims should be roughly half of level 0
        assert shape1[2] <= shape0[2] // 2 + 1
        assert shape1[3] <= shape0[3] // 2 + 1

    def test_known_value_appears_at_correct_cell(self, tmp_path):
        """A constant field should produce the same value at the corresponding cell."""
        gpi_slice = slice(0, N_GPI)
        values = np.full(N_GPI, 42.0, dtype="f4")
        sparse_path = _make_sparse_store_for_pyramid(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        root = zarr.open(str(out_path), mode="r")
        data = root["0"]["surface_soil_moisture"][0, 0, :, :]
        # Valid cells should all be 42.0
        lats, lons = _build_regular_grid(TEST_RESOLUTION)
        _, valid_mask = _compute_nn_lookup(grid, lats, lons, TEST_RESOLUTION, TEST_MAX_DIST)
        valid_data = data[valid_mask]
        assert np.allclose(valid_data, 42.0)

    def test_processed_array_set_after_regrid(self, tmp_path):
        gpi_slice = slice(0, 5)
        values = np.ones(5, dtype="f4")
        sparse_path = _make_sparse_store_for_pyramid(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        root = zarr.open(str(out_path), mode="r")
        assert bool(root["0"]["processed"][0, 0])

    def test_unprocessed_slots_skipped_on_rerun(self, tmp_path):
        """Re-running on a complete store should process 0 pending slices."""
        gpi_slice = slice(0, 5)
        values = np.ones(5, dtype="f4")
        sparse_path = _make_sparse_store_for_pyramid(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"

        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        all_slices = [(t, s) for t in range(N_SWATH_TIME) for s in range(N_SPACECRAFT)]
        pending = _find_pending_slices(out_path, sparse_path, all_slices)
        assert len(pending) == 0

    def test_new_slots_processed_on_rerun_after_sparse_growth(self, tmp_path):
        """After sparse store gains new swath_time slots, a second call should
        process only the new slots."""
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"

        # First run — empty sparse store
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )

        # Simulate sparse store growing: add data to a new slot by writing
        # directly and marking processed
        sparse_root = zarr.open(str(sparse_path), mode="a")
        sparse_root["surface_soil_moisture"][1, 0, :5] = np.ones(5, dtype="f4")
        sparse_root["processed"][1, 0] = True

        # Second run — should pick up slot (1, 0)
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )
        out_root = zarr.open(str(out_path), mode="r")
        assert bool(out_root["0"]["processed"][1, 0])


class TestClassifyVariables:

    def test_scalar_vars_classified_correctly(self, tmp_path):
        """3D arrays (swath_time, spacecraft, gpi) go into scalar_vars."""
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        assert "surface_soil_moisture" in scalar_vars
        assert "time" in scalar_vars
        assert len(beam_vars) == 0

    def test_coord_arrays_excluded(self, tmp_path):
        """Coordinate arrays (gpi, latitude, etc.) must not appear in either set."""
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        coord_names = {"swath_time", "spacecraft", "beam", "gpi", "latitude", "longitude"}
        assert not (beam_vars | scalar_vars) & coord_names

    def test_has_beams_false_produces_no_beam_vars(self, tmp_path):
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        assert len(beam_vars) == 0


class TestRegridToLatlon_UnprocessedSlots:

    def test_unprocessed_sparse_slots_produce_no_output(self, tmp_path):
        """Slots where sparse processed=False must produce fill values in pyramid,
        even if the array position contains data (e.g. from a previous partial run).
        """
        # Write data but deliberately leave processed=False
        sparse_path = _make_sparse_store_for_pyramid(tmp_path / "sp")
        sparse_root = zarr.open(str(sparse_path), mode="a")
        sparse_root["surface_soil_moisture"][0, 0, :10] = np.ones(10, dtype="f4")
        # processed[0, 0] stays False

        grid = _make_grid_mock()
        out_path = tmp_path / "pyramid.zarr"
        regrid_to_latlon(
            sparse_path, out_path, grid=grid,
            resolution_deg=TEST_RESOLUTION, max_dist_m=TEST_MAX_DIST,
            n_pyramid_levels=1, lat_chunk=TEST_LAT_CHUNK, lon_chunk=TEST_LON_CHUNK,
        )

        root = zarr.open(str(out_path), mode="r")
        data = root["0"]["surface_soil_moisture"][0, 0, :, :]
        fill_val = root["0"]["surface_soil_moisture"].metadata.fill_value

        # Every cell should be fill value — the slot was not processed
        assert (data == fill_val).all(), (
            "Slot with processed=False should produce all fill values in pyramid"
        )
        # And the pyramid processed array should also be False
        assert not bool(root["0"]["processed"][0, 0])
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        assert len(beam_vars) == 0
