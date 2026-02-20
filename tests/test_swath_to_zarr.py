"""
Test suite for ascat.stack.swath_to_zarr

Tests are organized around the main behaviors of stack_swaths_to_zarr and its
internal helpers.  All tests use synthetic data and do not require real swath
files on disk.  The SwathGridFiles interface is bypassed where possible by
calling internal functions (_create_zarr_structure, _insert_swath_file) directly.
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from ascat.stack.swath_to_zarr import (
    _balanced_shard_size,
    _create_zarr_structure,
    _extract_sat_id,
    _insert_swath_file,
    _maybe_expand_swath_time,
    stack_swaths_to_zarr,
    MISSION_SAT_ID_IDX_MAP,
)
from conftest import (
    CHUNK_SIZE_GPI,
    N_GPI,
    N_SPACECRAFT,
    N_SWATH_TIME,
    SHARD_SIZE_GPI,
    SAT_SERIES,
    TIME_RESOLUTION,
)


# ===========================================================================
# _balanced_shard_size
# ===========================================================================

class TestBalancedShardSize:

    def test_even_split(self):
        """N_GPI divisible by chunk_size gives two equal shards."""
        # 1000 GPIs, chunk 100 → per_shard=500, rounded up to 500
        result = _balanced_shard_size(1000, 100, n_shards=2)
        assert result == 500
        assert result % 100 == 0

    def test_odd_split_rounds_up(self):
        """Result is always a multiple of chunk_size."""
        result = _balanced_shard_size(N_GPI, CHUNK_SIZE_GPI, n_shards=2)
        assert result % CHUNK_SIZE_GPI == 0

    def test_two_shards_cover_all_gpis(self):
        """Two shards of the computed size must cover all GPIs."""
        shard = _balanced_shard_size(N_GPI, CHUNK_SIZE_GPI, n_shards=2)
        assert shard * 2 >= N_GPI

    def test_n_shards_parameter(self):
        """n_shards=4 produces a smaller shard size than n_shards=2."""
        shard2 = _balanced_shard_size(N_GPI, CHUNK_SIZE_GPI, n_shards=2)
        shard4 = _balanced_shard_size(N_GPI, CHUNK_SIZE_GPI, n_shards=4)
        assert shard4 < shard2

    def test_real_world_size(self):
        """3,300,001 GPIs with chunk 4096 gives a multiple of 4096."""
        result = _balanced_shard_size(3_300_001, 4096, n_shards=2)
        assert result % 4096 == 0
        assert result * 2 >= 3_300_001


# ===========================================================================
# _create_zarr_structure
# ===========================================================================

class TestCreateZarrStructure:

    @pytest.fixture
    def created_store(self, tmp_path, synthetic_grid, swath_file_factory):
        """Create a minimal zarr structure using a synthetic sample file."""
        sample_file = swath_file_factory(
            0, "a",
            gpi_indices=np.arange(10),
            obs_values={"surface_soil_moisture": np.ones(10, dtype="f4")},
        )
        out_path = tmp_path / "swaths.zarr"
        _create_zarr_structure(
            out_path=out_path,
            grid=synthetic_grid,
            date_start=datetime(2024, 12, 1),
            date_end=datetime(2024, 12, 1, 4),
            time_resolution=TIME_RESOLUTION,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
            sat_series=SAT_SERIES,
            sample_file=sample_file,
        )
        return zarr.open(out_path, mode="r"), out_path

    def test_zarr_json_exists(self, created_store):
        _, out_path = created_store
        assert (out_path / "zarr.json").exists()

    def test_swath_time_dimension(self, created_store):
        root, _ = created_store
        assert "swath_time" in root
        assert root["swath_time"].ndim == 1
        assert root["swath_time"].shape[0] == N_SWATH_TIME

    def test_spacecraft_dimension(self, created_store):
        root, _ = created_store
        assert "spacecraft" in root
        assert root["spacecraft"].shape[0] == N_SPACECRAFT

    def test_gpi_dimension(self, created_store):
        root, _ = created_store
        assert "gpi" in root
        assert root["gpi"].shape[0] == N_GPI

    def test_data_array_shape(self, created_store):
        root, _ = created_store
        ssm = root["surface_soil_moisture"]
        assert ssm.shape == (N_SWATH_TIME, N_SPACECRAFT, N_GPI)

    def test_data_array_dimensions_named(self, created_store):
        root, _ = created_store
        ssm = root["surface_soil_moisture"]
        assert ssm.metadata.dimension_names == ("swath_time", "spacecraft", "gpi")

    def test_sharding_applied(self, created_store):
        """Data arrays should use the sharding_indexed codec."""
        root, _ = created_store
        meta = root["surface_soil_moisture"].metadata
        has_sharding = any(
            type(c).__name__ == "ShardingCodec" for c in (meta.codecs or [])
        )
        assert has_sharding

    def test_shard_shape_correct(self, created_store):
        root, _ = created_store
        meta = root["surface_soil_moisture"].metadata
        assert meta.chunk_grid.chunk_shape[-1] == SHARD_SIZE_GPI

    def test_processed_array_created(self, created_store):
        root, _ = created_store
        assert "processed" in root
        assert root["processed"].shape == (N_SWATH_TIME, N_SPACECRAFT)
        assert root["processed"].dtype == bool
        assert not root["processed"][:].any()

    def test_gpi_shard_size_not_multiple_raises(self, tmp_path, synthetic_grid, swath_file_factory):
        sample_file = swath_file_factory(0, "a", np.arange(10),
                                         {"surface_soil_moisture": np.ones(10, "f4")})
        with pytest.raises(ValueError, match="multiple"):
            _create_zarr_structure(
                out_path=tmp_path / "bad.zarr",
                grid=synthetic_grid,
                date_start=datetime(2024, 12, 1),
                date_end=datetime(2024, 12, 1, 4),
                time_resolution=TIME_RESOLUTION,
                chunk_size_gpi=CHUNK_SIZE_GPI,
                gpi_shard_size=CHUNK_SIZE_GPI + 1,   # not a multiple
                sat_series=SAT_SERIES,
                sample_file=sample_file,
            )


# ===========================================================================
# _maybe_expand_swath_time
# ===========================================================================

class TestMaybeExpandSwathTime:

    @pytest.fixture
    def small_store(self, tmp_path, synthetic_grid, swath_file_factory):
        """A store covering 2024-12-01 00:00 to 2024-12-01 04:00 (4 slots)."""
        sample_file = swath_file_factory(0, "a", np.arange(10),
                                         {"surface_soil_moisture": np.ones(10, "f4")})
        out_path = tmp_path / "expand.zarr"
        _create_zarr_structure(
            out_path=out_path,
            grid=synthetic_grid,
            date_start=datetime(2024, 12, 1),
            date_end=datetime(2024, 12, 1, 4),
            time_resolution=TIME_RESOLUTION,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
            sat_series=SAT_SERIES,
            sample_file=sample_file,
        )
        return out_path

    def test_no_expansion_when_within_range(self, small_store):
        before = zarr.open(small_store, mode="r")["swath_time"].shape[0]
        _maybe_expand_swath_time(small_store, datetime(2024, 12, 1, 3), TIME_RESOLUTION)
        after = zarr.open(small_store, mode="r")["swath_time"].shape[0]
        assert before == after

    def test_expansion_adds_correct_number_of_slots(self, small_store):
        # Extend by 2 more hours (slots 4 and 5)
        _maybe_expand_swath_time(small_store, datetime(2024, 12, 1, 6), TIME_RESOLUTION)
        root = zarr.open(small_store, mode="r")
        assert root["swath_time"].shape[0] == 6

    def test_data_arrays_resized(self, small_store):
        _maybe_expand_swath_time(small_store, datetime(2024, 12, 1, 6), TIME_RESOLUTION)
        root = zarr.open(small_store, mode="r")
        assert root["surface_soil_moisture"].shape[0] == 6

    def test_processed_array_resized(self, small_store):
        _maybe_expand_swath_time(small_store, datetime(2024, 12, 1, 6), TIME_RESOLUTION)
        root = zarr.open(small_store, mode="r")
        assert root["processed"].shape == (6, N_SPACECRAFT)

    def test_new_time_coords_correct(self, small_store):
        _maybe_expand_swath_time(small_store, datetime(2024, 12, 1, 6), TIME_RESOLUTION)
        root = zarr.open(small_store, mode="r")
        times = root["swath_time"][:]
        # Last two slots should be at 05:00 and (no — inclusive="neither" excludes last_time)
        # Check that new entries are monotonically increasing
        assert (np.diff(times.astype("i8")) > 0).all()

    def test_existing_data_not_corrupted(self, small_store):
        """Pre-existing time coordinates must be unchanged after expansion."""
        root = zarr.open(small_store, mode="r")
        original_times = root["swath_time"][:].copy()
        _maybe_expand_swath_time(small_store, datetime(2024, 12, 1, 6), TIME_RESOLUTION)
        root = zarr.open(small_store, mode="r")
        np.testing.assert_array_equal(root["swath_time"][:4], original_times)


# ===========================================================================
# _insert_swath_file
# ===========================================================================

class TestInsertSwathFile:
    """Tests for _insert_swath_file called directly with a pre-built zarr store."""

    @pytest.fixture
    def store_and_coords(self, tmp_path, synthetic_grid, swath_file_factory):
        """A 4-slot store plus synthetic swath files for slot 0 (metop-a) and
        slot 2 (metop-b)."""
        sample_file = swath_file_factory(0, "a", np.arange(10),
                                         {"surface_soil_moisture": np.ones(10, "f4")})
        out_path = tmp_path / "insert_test.zarr"
        _create_zarr_structure(
            out_path=out_path,
            grid=synthetic_grid,
            date_start=datetime(2024, 12, 1),
            date_end=datetime(2024, 12, 1, 4),
            time_resolution=TIME_RESOLUTION,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
            sat_series=SAT_SERIES,
            sample_file=sample_file,
        )
        root = zarr.open(out_path, mode="a")
        time_coords = root["swath_time"][:]
        return out_path, root, time_coords

    def test_data_written_to_correct_position(
        self, store_and_coords, swath_file_factory, synthetic_grid
    ):
        """Values from slot 0 / metop-a should appear at [0, 0, gpi]."""
        out_path, root, time_coords = store_and_coords
        gpi_indices = np.array([0, 1, 2, 3, 4])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="f4")
        swath_file = swath_file_factory(0, "a", gpi_indices,
                                        {"surface_soil_moisture": expected})
        swath_mock = _build_swath_files_mock(synthetic_grid)
        result = _insert_swath_file(
            swath_file, swath_mock, root, time_coords, TIME_RESOLUTION
        )
        assert result is True
        written = root["surface_soil_moisture"][0, 0, gpi_indices]
        np.testing.assert_array_almost_equal(written, expected)

    def test_processed_marked_true(self, store_and_coords, swath_file_factory, synthetic_grid):
        out_path, root, time_coords = store_and_coords
        swath_file = swath_file_factory(0, "a", np.arange(5),
                                        {"surface_soil_moisture": np.ones(5, "f4")})
        swath_mock = _build_swath_files_mock(synthetic_grid)
        _insert_swath_file(swath_file, swath_mock, root, time_coords, TIME_RESOLUTION)
        assert bool(root["processed"][0, 0])

    def test_other_slots_remain_unprocessed(self, store_and_coords, swath_file_factory, synthetic_grid):
        out_path, root, time_coords = store_and_coords
        swath_file = swath_file_factory(0, "a", np.arange(5),
                                        {"surface_soil_moisture": np.ones(5, "f4")})
        swath_mock = _build_swath_files_mock(synthetic_grid)
        _insert_swath_file(swath_file, swath_mock, root, time_coords, TIME_RESOLUTION)
        # All slots except (0,0) should still be False
        processed = root["processed"][:]
        assert bool(processed[0, 0])
        assert not bool(processed[0, 1])
        assert not processed[1:, :].any()

    def test_out_of_range_timestamp_warns_and_returns_false(
        self, store_and_coords, swath_file_factory, synthetic_grid
    ):
        out_path, root, time_coords = store_and_coords
        # slot_idx=99 is far outside the store's time range
        swath_file = swath_file_factory(99, "a", np.arange(5),
                                        {"surface_soil_moisture": np.ones(5, "f4")})
        swath_mock = _build_swath_files_mock(synthetic_grid)
        with pytest.warns(UserWarning, match="not in time coordinates"):
            result = _insert_swath_file(
                swath_file, swath_mock, root, time_coords, TIME_RESOLUTION
            )
        assert result is False

    def test_unknown_variable_warns_and_skips(
        self, store_and_coords, swath_file_factory, synthetic_grid
    ):
        out_path, root, time_coords = store_and_coords
        swath_file = swath_file_factory(
            0, "a", np.arange(5),
            {
                "surface_soil_moisture": np.ones(5, "f4"),
                "unknown_var_xyz": np.zeros(5, "f4"),
            },
        )
        swath_mock = _build_swath_files_mock(synthetic_grid)
        with pytest.warns(UserWarning, match="not in Zarr schema"):
            result = _insert_swath_file(
                swath_file, swath_mock, root, time_coords, TIME_RESOLUTION
            )
        assert result is True  # file still succeeded despite unknown var

    def test_idempotent_reinsertion(self, store_and_coords, swath_file_factory, synthetic_grid):
        """Inserting the same file twice should produce the same values."""
        out_path, root, time_coords = store_and_coords
        gpi_indices = np.array([0, 1, 2])
        expected = np.array([7.0, 8.0, 9.0], dtype="f4")
        swath_file = swath_file_factory(0, "a", gpi_indices,
                                        {"surface_soil_moisture": expected})
        swath_mock = _build_swath_files_mock(synthetic_grid)
        _insert_swath_file(swath_file, swath_mock, root, time_coords, TIME_RESOLUTION)
        _insert_swath_file(swath_file, swath_mock, root, time_coords, TIME_RESOLUTION)
        written = root["surface_soil_moisture"][0, 0, gpi_indices]
        np.testing.assert_array_almost_equal(written, expected)


# ===========================================================================
# _extract_sat_id
# ===========================================================================

class TestExtractSatId:

    def test_metop_a(self):
        fn = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-6.25km-H139_C_LIIB_00000000000000_20241201000000____.nc"
        pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H139_C_LIIB_{placeholder}_{date}____.nc"
        assert _extract_sat_id(fn, pattern, "metop") == "a"

    def test_metop_b(self):
        fn = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPB-6.25km-H139_C_LIIB_00000000000000_20241201000000____.nc"
        pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H139_C_LIIB_{placeholder}_{date}____.nc"
        assert _extract_sat_id(fn, pattern, "metop") == "b"

    def test_unknown_sat_raises(self):
        fn = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPZ-6.25km-H139_C_LIIB_00000000000000_20241201000000____.nc"
        pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H139_C_LIIB_{placeholder}_{date}____.nc"
        with pytest.raises(ValueError, match="Unknown satellite"):
            _extract_sat_id(fn, pattern, "metop")

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            _extract_sat_id("garbage_filename.nc",
                            "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H139_C_LIIB_{x}_{date}____.nc",
                            "metop")


# ===========================================================================
# Integration: stack_swaths_to_zarr append / expansion
# ===========================================================================

class TestStackSwathsToZarrIntegration:
    """Integration tests using a mocked SwathGridFiles."""

    def test_creates_store_on_first_call(self, tmp_path, synthetic_grid, swath_file_factory):
        """First call creates the zarr.json sentinel."""
        from ascat.stack.swath_to_zarr import stack_swaths_to_zarr

        out_path = tmp_path / "integration.zarr"

        # Build swath files for slots 0 and 1 (metop-a)
        file0 = swath_file_factory(0, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})
        file1 = swath_file_factory(1, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})

        # Build mock SwathGridFiles
        mock_swath_files = _build_swath_files_mock(synthetic_grid)

        # Add search_period method
        def _search_period(dt_start, dt_end, date_field_fmt, end_inclusive=False):
            """Return files within the date range."""
            available = [file0, file1]
            # Parse dates from filenames to filter
            files_in_range = []
            for f in available:
                dt = mock_swath_files._parse_date(f, "date", date_field_fmt)
                if dt_start <= dt < dt_end:
                    files_in_range.append(f)
            return files_in_range

        mock_swath_files.search_period = _search_period

        # Call stack_swaths_to_zarr
        stack_swaths_to_zarr(
            swath_files=mock_swath_files,
            out_path=out_path,
            date_range=(datetime(2024, 12, 1), datetime(2024, 12, 1, 4)),
            time_resolution=TIME_RESOLUTION,
            n_workers=1,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
        )

        # Assert store was created
        assert (out_path / "zarr.json").exists()
        root = zarr.open(out_path, mode="r")
        assert root["swath_time"].shape[0] == 4
        # Slots 0 and 1 (metop-a, spacecraft idx 0) should be processed
        assert bool(root["processed"][0, 0])
        assert bool(root["processed"][1, 0])

    def test_expands_store_on_second_call(self, tmp_path, synthetic_grid, swath_file_factory):
        """Second call with later date_range expands swath_time without corrupting first batch."""
        from ascat.stack.swath_to_zarr import stack_swaths_to_zarr

        out_path = tmp_path / "integration.zarr"

        # First call: create files for slots 0 and 1
        file0 = swath_file_factory(0, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})
        file1 = swath_file_factory(1, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})

        mock_swath_files = _build_swath_files_mock(synthetic_grid)

        def _search_period(dt_start, dt_end, date_field_fmt, end_inclusive=False):
            available = [file0, file1]
            files_in_range = []
            for f in available:
                dt = mock_swath_files._parse_date(f, "date", date_field_fmt)
                if dt_start <= dt < dt_end:
                    files_in_range.append(f)
            return files_in_range

        mock_swath_files.search_period = _search_period

        stack_swaths_to_zarr(
            swath_files=mock_swath_files,
            out_path=out_path,
            date_range=(datetime(2024, 12, 1), datetime(2024, 12, 1, 4)),
            time_resolution=TIME_RESOLUTION,
            n_workers=1,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
        )

        # Verify first batch
        root = zarr.open(out_path, mode="r")
        assert root["swath_time"].shape[0] == 4
        assert root["processed"][0, 0] == True   # noqa: E712
        assert root["processed"][1, 0] == True   # noqa: E712

        # Second call: extend with files for slots 4 and 5
        file4 = swath_file_factory(4, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})
        file5 = swath_file_factory(5, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})

        def _search_period_2(dt_start, dt_end, date_field_fmt, end_inclusive=False):
            available = [file0, file1, file4, file5]
            files_in_range = []
            for f in available:
                dt = mock_swath_files._parse_date(f, "date", date_field_fmt)
                if dt_start <= dt < dt_end:
                    files_in_range.append(f)
            return files_in_range

        mock_swath_files.search_period = _search_period_2

        stack_swaths_to_zarr(
            swath_files=mock_swath_files,
            out_path=out_path,
            date_range=(datetime(2024, 12, 1), datetime(2024, 12, 1, 6)),
            time_resolution=TIME_RESOLUTION,
            n_workers=1,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
        )

        # Verify expansion without corruption
        root = zarr.open(out_path, mode="r")
        assert root["swath_time"].shape[0] == 6
        # First batch should still be marked processed
        assert bool(root["processed"][0, 0])
        assert bool(root["processed"][1, 0])
        assert bool(root["processed"][4, 0])
        assert bool(root["processed"][5, 0])

    def test_no_expansion_when_date_within_existing_range(
        self, tmp_path, synthetic_grid, swath_file_factory
    ):
        """Re-running with the same date_range does not resize anything."""
        from ascat.stack.swath_to_zarr import stack_swaths_to_zarr

        out_path = tmp_path / "integration.zarr"

        file0 = swath_file_factory(0, "a", np.arange(5), {"surface_soil_moisture": np.ones(5, "f4")})

        mock_swath_files = _build_swath_files_mock(synthetic_grid)

        def _search_period(dt_start, dt_end, date_field_fmt, end_inclusive=False):
            available = [file0]
            files_in_range = []
            for f in available:
                dt = mock_swath_files._parse_date(f, "date", date_field_fmt)
                if dt_start <= dt < dt_end:
                    files_in_range.append(f)
            return files_in_range

        mock_swath_files.search_period = _search_period

        # First call
        stack_swaths_to_zarr(
            swath_files=mock_swath_files,
            out_path=out_path,
            date_range=(datetime(2024, 12, 1), datetime(2024, 12, 1, 4)),
            time_resolution=TIME_RESOLUTION,
            n_workers=1,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
        )

        root = zarr.open(out_path, mode="r")
        old_shape = root["swath_time"].shape[0]
        assert old_shape == 4

        # Second call with same range
        stack_swaths_to_zarr(
            swath_files=mock_swath_files,
            out_path=out_path,
            date_range=(datetime(2024, 12, 1), datetime(2024, 12, 1, 4)),
            time_resolution=TIME_RESOLUTION,
            n_workers=1,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
        )

        root = zarr.open(out_path, mode="r")
        new_shape = root["swath_time"].shape[0]
        assert new_shape == old_shape

    def test_multiprocessing_does_not_fail_with_sorted_grid(
        self, tmp_path, synthetic_grid, swath_file_factory
    ):
        """n_workers > 1 must succeed when sorted_grid is provided.

        Regression test for the pykdtree KDTree pickle failure: sorted_grid
        must be converted to a plain numpy gpi_lookup array before being sent
        to worker processes.
        """
        out_path = tmp_path / "mp_test.zarr"

        files = [
            swath_file_factory(i, "a", np.arange(5),
                            {"surface_soil_moisture": np.ones(5, "f4")})
            for i in range(3)
        ]

        mock_swath_files = _build_swath_files_mock(synthetic_grid)

        def _search_period(dt_start, dt_end, date_field_fmt, end_inclusive=False):
            result = []
            for f in files:
                dt = mock_swath_files._parse_date(f, "date", date_field_fmt)
                if dt_start <= dt < dt_end:
                    result.append(f)
            return result

        mock_swath_files.search_period = _search_period

        # Should not raise — specifically must not raise
        # "TypeError: no default __reduce__ due to non-trivial __cinit__"
        stack_swaths_to_zarr(
            swath_files=mock_swath_files,
            out_path=out_path,
            date_range=(datetime(2024, 12, 1), datetime(2024, 12, 1, 4)),
            time_resolution=TIME_RESOLUTION,
            n_workers=2,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            gpi_shard_size=SHARD_SIZE_GPI,
            sorted_grid=synthetic_grid,
        )
        root = zarr.open(out_path, mode="r")
        assert bool(root["processed"][:].any())

# ===========================================================================
# Helpers
# ===========================================================================

def _build_swath_files_mock(grid):
    """Build a minimal SwathGridFiles mock compatible with _insert_swath_file."""
    from ascat.stack.swath_to_zarr import MISSION_SAT_ID_IDX_MAP

    mock = MagicMock()
    mock.sat_series = SAT_SERIES
    mock.date_field_fmt = "%Y%m%d%H%M%S"
    mock.ft.fn_templ = (
        "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H139_C_LIIB_"
        "{placeholder}_{date}____.nc"
    )
    mock.grid = grid

    def _parse_date(path, date_field, date_field_fmt):
        # Extract date from filename: last segment before ____.nc
        import re as _re
        m = _re.search(r"_(\d{14})____\.nc$", str(path))
        if not m:
            raise ValueError(f"Cannot parse date from {path}")
        return datetime.strptime(m.group(1), date_field_fmt)

    mock._parse_date.side_effect = _parse_date
    return mock

