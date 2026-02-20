"""
Regression tests for ASCAT data pipeline using real H129 product data.

These tests validate both correctness and performance of the three-stage pipeline:
1. swath_to_zarr: Convert raw swath NetCDF files → sparse Zarr cube
2. sparse_to_dense: Convert sparse Zarr cube → dense time-series Zarr
3. regrid_to_latlon: Convert sparse Zarr cube → lat/lon multiscale pyramid Zarr

To establish or re-establish performance baselines:
    1. Delete tests/ascat-test-data/tests/regression_baseline.json
    2. Re-run this test file
    3. The first run will skip the comparison test and write the new baseline
    4. Re-run again to validate against the new baseline

These tests use pytest marks:
    - `slow`: Tests can be skipped with `-m "not slow"` in CI
    - `benchmark`: Focus on performance validation

All timing data is stored in regression_baseline.json relative to this file.
"""

from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

# Module-level dictionary for sharing timings between tests
# Simpler than fixtures for cross-test state and appropriate for module-scoped tests
_TIMINGS: dict[str, float] = {}

# Constants for performance validation
REGRESSION_TOLERANCE = 1.5  # Accept 50% slowdown before failing
IMPROVEMENT_THRESHOLD = 0.7  # Update baseline if >30% improvement

# Path to baseline JSON (relative to this test file)
_BASELINE_PATH = Path(__file__).parent / "regression_baseline.json"


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


@contextmanager
def record_timing():
    """Context manager to time a code block and return elapsed time.

    Yields a mutable `elapsed` container that is updated with the elapsed
    seconds when the block completes.

    Example:
        >>> with record_timing() as t:
        ...     expensive_operation()
        >>> elapsed = t.seconds
    """
    class _TimeRecorder:
        seconds: float = 0.0

    recorder = _TimeRecorder()
    start = time.perf_counter()
    try:
        yield recorder
    finally:
        recorder.seconds = time.perf_counter() - start


def _load_baseline() -> dict[str, float]:
    """Load timing baseline from JSON file.

    Returns
    -------
    dict
        Baseline timings, or empty dict if file doesn't exist.
    """
    if _BASELINE_PATH.exists():
        return json.loads(_BASELINE_PATH.read_text())
    return {}


def _save_baseline(data: dict[str, float]) -> None:
    """Write timing baseline to JSON file.

    Parameters
    ----------
    data : dict
        Baseline timings to write.
    """
    _BASELINE_PATH.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Module-scoped shared temporary directory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temporary directory shared by all tests.

    Allows stage 2 and 3 to consume stage 1's output without rebuilding.
    """
    return tmp_path_factory.getbasetemp() / "regression_tests"


@pytest.fixture(scope="module")
def grid():
    """Load the H129 fibgrid (6.25km spacing) for use in all tests."""
    from fibgrid.realization import FibGrid

    return FibGrid(6.25)


# ---------------------------------------------------------------------------
# Test helpers for fixture discovery
# ---------------------------------------------------------------------------


def _get_swath_files(data_dir: Path) -> list[Path]:
    """Discover available swath files in the test data directory.

    Returns
    -------
    list[Path]
        All .nc files found in the h129 test data directory.
    """
    return list(data_dir.glob("*.nc"))


# ---------------------------------------------------------------------------
# Pipeline stage tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_swath_to_zarr_runs_and_produces_correct_structure(shared_tmp: Path, grid):
    """Test swath_to_zarr stage on real H129 data.

    Validates:
    - Basic functionality (zarr.json exists, data inserted)
    - Structure correctness (dimensions, coordinate arrays)
    - Records timing for regression tracking.
    """
    from ascat.swath import SwathGridFiles
    from ascat.stack.swath_to_zarr import stack_swaths_to_zarr

    # Use appropriate date range from available test data
    # Based on actual files, the available dates are Jan 12, 18, 19, and 31, 2021
    date_start = datetime(2021, 1, 10)
    date_end = datetime(2021, 1, 20)

    out_path = shared_tmp / "sparse.zarr"

    with record_timing() as t:
        stack_swaths_to_zarr(
            swath_files=SwathGridFiles.from_product_id(
                Path(__file__).parent / "ascat-test-data" / "hsaf" / "h129" / "swaths",
                "h129",
            ),
            out_path=out_path,
            date_range=(date_start, date_end),
            time_resolution="60min",
            n_workers=1,  # Single worker for consistent timing
            chunk_size_gpi=2**15,
        )
    _TIMINGS["swath_to_zarr"] = t.seconds

    # Validate output structure
    assert out_path.exists(), "Output directory should exist"
    assert (out_path / "zarr.json").exists(), "Zarr sentinel file should exist"

    root = zarr.open(out_path, mode="r")

    # Validate dimensions
    assert "swath_time" in root, "swath_time dimension should exist"
    assert "spacecraft" in root, "spacecraft dimension should exist"
    assert "gpi" in root, "gpi dimension should exist"

    n_swath_time = root["swath_time"].shape[0]
    n_spacecraft = root["spacecraft"].shape[0]
    n_gpi = root["gpi"].shape[0]

    assert n_swath_time > 0, f"swath_time should have data, got {n_swath_time}"
    assert n_spacecraft > 0, f"spacecraft should have data, got {n_spacecraft}"
    assert n_gpi > 0, f"gpi should have data, got {n_gpi}"

    # Validate processed array
    assert "processed" in root, "processed array should exist"
    processed = root["processed"]
    assert processed.dtype == np.bool_, f"processed should be bool, got {processed.dtype}"
    assert np.any(processed[:]), "At least one entry should be marked processed"

    # Validate coordinate arrays
    assert "latitude" in root, "latitude coordinate should exist"
    assert "longitude" in root, "longitude coordinate should exist"
    assert "gpi" in root, "gpi coordinate should exist"

    assert root["latitude"].shape[0] == n_gpi, "latitude length should match gpi"
    assert root["longitude"].shape[0] == n_gpi, "longitude length should match gpi"
    assert root["gpi"].shape[0] == n_gpi, "gpi array length should match"


@pytest.mark.slow
@pytest.mark.benchmark
def test_sparse_to_dense_runs_and_produces_correct_structure(shared_tmp: Path):
    """Test sparse_to_dense stage on real sparse Zarr data.

    Validates:
    - n_obs counts non-zero for some GPIs
    - Values are finite (not fill values)
    - Time is monotonically non-decreasing for each GPI
    - Records timing for regression tracking.
    """
    from ascat.stack.sparse_zarr_to_ts import sparse_to_dense

    sparse_path = shared_tmp / "sparse.zarr"
    out_path = shared_tmp / "dense.zarr"

    with record_timing() as t:
        sparse_to_dense(
            sparse_path,
            out_path,
            n_workers=1,
            chunk_size_gpi=1024,
            chunk_size_obs=30,
        )
    _TIMINGS["sparse_to_dense"] = t.seconds

    # Validate output structure
    assert (out_path / "zarr.json").exists(), "Output zarr.json should exist"

    root = zarr.open(out_path, mode="r")

    # Validate n_obs array
    assert "n_obs" in root, "n_obs array should exist"
    n_obs = root["n_obs"]

    # At least some GPIs should have observations
    assert np.any(n_obs[:] > 0), "At least one GPI should have observations"

    # Find a GPI with multiple observations for time ordering check
    multi_obs_gpis = np.where(n_obs[:] > 1)[0]
    if len(multi_obs_gpis) > 0:
        gpi = multi_obs_gpis[0]
        n = n_obs[gpi]

        # Get surface_soil_moisture for this GPI
        ssm = root["surface_soil_moisture"][gpi, :n]
        # Check values are finite (not fill values)
        assert np.all(np.isfinite(ssm)), f"Values for GPI {gpi} should be finite"

        # Get time values and check monotonic non-decreasing
        time_vals = root["time"][gpi, :n]
        assert np.all(np.diff(time_vals) >= 0), (
            f"Time should be monotonically non-decreasing for GPI {gpi}"
        )
    else:
        # This is fine if dataset is very small - just ensure at least some values
        single_obs_gpis = np.where(n_obs[:] > 0)[0]
        if len(single_obs_gpis) > 0:
            gpi = single_obs_gpis[0]
            n = n_obs[gpi]
            ssm = root["surface_soil_moisture"][gpi, :n]
            assert np.all(np.isfinite(ssm)), f"Values for GPI {gpi} should be finite"


@pytest.mark.slow
@pytest.mark.benchmark
def test_regrid_to_latlon_runs_and_produces_correct_structure(shared_tmp: Path, grid):
    """Test regrid_to_latlon stage on real sparse Zarr data.

    Validates:
    - Multiscales metadata present
    - Level 0 has non-fill values
    - Level 1 is spatially smaller than level 0
    - Records timing for regression tracking.
    """
    from ascat.regrid.sparse_zarr_pyramids import regrid_to_latlon

    sparse_path = shared_tmp / "sparse.zarr"
    out_path = shared_tmp / "pyramids.zarr"

    with record_timing() as t:
        regrid_to_latlon(
            sparse_path,
            out_path,
            grid=grid,
            resolution_deg=0.25,  # ~25km resolution for H129
            max_dist_m=50_000,     # H129 has 6.25km spacing
            n_pyramid_levels=2,
            lat_chunk=10,
            lon_chunk=10,
            n_workers=1,  # Single worker for consistent timing
        )
    _TIMINGS["regrid_to_latlon"] = t.seconds

    # Validate output structure
    assert (out_path / "zarr.json").exists(), "Output zarr.json should exist"

    root = zarr.open(out_path, mode="r")

    # Validate multiscales metadata (it's directly in root.attrs in zarr v3)
    assert "multiscales" in root.attrs, "multiscales should exist in root.attrs"

    # Check level 0
    assert "0" in root, "Level 0 group should exist"
    level0 = root["0"]
    assert "surface_soil_moisture" in level0, "surface_soil_moisture should exist in level 0"

    ssm0 = level0["surface_soil_moisture"]
    assert len(ssm0.shape) == 4, f"Level 0 should have 4 dims, got {len(ssm0.shape)}"
    shape0 = ssm0.shape

    # Check that some cells have valid (non-fill) data
    # Get the first slice for a spacecraft
    data = ssm0[0, 0, :, :]
    # Find values that are not the fill value (-9999.0)
    valid = np.abs(data) < 9000.0
    valid_fraction = valid.sum() / valid.size
    assert valid_fraction > 0.01, (
        f"Level 0 should have at least 1% valid cells, got {valid_fraction:.1%}"
    )

    # Check level 1
    assert "1" in root, "Level 1 group should exist"
    level1 = root["1"]
    assert "surface_soil_moisture" in level1, "surface_soil_moisture should exist in level 1"

    ssm1 = level1["surface_soil_moisture"]
    assert len(ssm1.shape) == 4, f"Level 1 should have 4 dims, got {len(ssm1.shape)}"
    shape1 = ssm1.shape

    # Level 1 should be spatially smaller than level 0
    # First two dims are (swath_time, spacecraft), last two are (lat, lon)
    assert shape1[0] == shape0[0], "Level 1 should have same swath_time dim as level 0"
    assert shape1[1] == shape0[1], "Level 1 should have same spacecraft dim as level 0"
    assert shape1[2] < shape0[2], (
        f"Level 1 lat dim ({shape1[2]}) should be smaller than level 0 ({shape0[2]})"
    )
    assert shape1[3] < shape0[3], (
        f"Level 1 lon dim ({shape1[3]}) should be smaller than level 0 ({shape0[3]})"
    )


@pytest.mark.slow
@pytest.mark.benchmark
def test_performance_within_baseline():
    """Validate performance against saved baselines.

    This test must run after the three pipeline tests above. It:
    - Loads regression baselines from JSON
    - If no baseline exists, writes it and skips
    - If baseline exists, validates current times are within tolerance
    - Updates baseline for significant performance improvements (>30%)
    """
    baseline = _load_baseline()

    # If no baseline exists, this is the first run
    if not baseline:
        _save_baseline(_TIMINGS)
        pytest.skip(
            "No baseline file exists. "
            f"Written initial baselines to {_BASELINE_PATH.relative_to(Path(__file__).parent)}. "
            "Rerun to validate performance."
        )

    # Validate each timing against baseline
    updated = False
    for key, measured in _TIMINGS.items():
        if key not in baseline:
            warnings.warn(
                f"Timing '{key}' not found in baseline. "
                "This may indicate a new test was added. "
                f"Adding current value {measured:.2f}s to baseline.",
                stacklevel=2,
            )
            baseline[key] = measured
            updated = True
            continue

        baseline_time = baseline[key]

        # Check for regression (significant slowdown)
        if measured > baseline_time * REGRESSION_TOLERANCE:
            pytest.fail(
                f"Performance regression for '{key}': "
                f"{measured:.2f}s > baseline {baseline_time:.2f}s "
                f"(tolerance {REGRESSION_TOLERANCE}x)."
            )

        # Check for improvement (significant speedup)
        if measured < baseline_time * IMPROVEMENT_THRESHOLD:
            warnings.warn(
                f"Performance improved for '{key}': "
                f"{baseline_time:.2f}s → {measured:.2f}s. "
                "Updating baseline.",
                stacklevel=2,
            )
            baseline[key] = measured
            updated = True

    # If any baselines were updated, save the file
    if updated:
        _save_baseline(baseline)
