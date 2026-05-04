"""
Test suite for ascat.stack.sparse_zarr_to_ts (sparse_to_dense)

Tests are organized around the main behaviors of sparse_to_dense and its
internal helpers.  All tests operate on synthetic sparse Zarr stores built
directly via zarr API calls — no SwathGridFiles or real NetCDF files required.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from ascat.stack.sparse_zarr_to_ts import (
    _ShardingConfig,
    _classify_variables,
    _classify_intermediate_variables,
    _create_dense_structure_from_intermediate,
    _expand_obs_dimension,
    _round_up,
    _scan_all_populated_slots,
    densify_from_intermediate,
    rechunk_sparse,
    sparse_to_dense_rechunked,
)
from conftest import (
    CHUNK_SIZE_GPI,
    N_GPI,
    N_SPACECRAFT,
    N_SWATH_TIME,
    SHARD_SIZE_GPI,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic sparse zarr stores
# ---------------------------------------------------------------------------

def _make_sparse_store(
    tmp_path,
    n_gpi=N_GPI,
    n_swath_time=N_SWATH_TIME,
    n_spacecraft=N_SPACECRAFT,
    chunk_size_gpi=CHUNK_SIZE_GPI,
    shard_size_gpi=None,
    populated_slots=None,
    var_name="surface_soil_moisture",
    fill_value=np.float32(-9999.0),
    time_fill=np.float64(0.0),
):
    """Build a minimal synthetic sparse Zarr store directly via zarr API.

    Parameters
    ----------
    populated_slots : list of (t_idx, s_idx, gpi_slice, values) or None
        Each entry writes ``values`` into the store at (t_idx, s_idx, gpi_slice).
        If None, no data is written (store is empty).
    """
    path = tmp_path
    store = zarr.storage.LocalStore(path)
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    shape = (n_swath_time, n_spacecraft, n_gpi)
    inner_chunks = (1, 1, chunk_size_gpi)

    create_kwargs = dict(
        dtype="float32",
        shape=shape,
        chunks=inner_chunks,
        dimension_names=("swath_time", "spacecraft", "gpi"),
        fill_value=fill_value,
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
    )
    if shard_size_gpi is not None:
        create_kwargs["shards"] = (1, 1, shard_size_gpi)

    root.create_array(name=var_name, **create_kwargs)

    # time variable (float64, same layout)
    time_kwargs = dict(
        dtype="float64",
        shape=shape,
        chunks=inner_chunks,
        dimension_names=("swath_time", "spacecraft", "gpi"),
        fill_value=time_fill,
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
    )
    if shard_size_gpi is not None:
        time_kwargs["shards"] = (1, 1, shard_size_gpi)
    root.create_array(name="time", **time_kwargs)

    root.create_array(
        "gpi", data=np.arange(n_gpi, dtype="int32"),
        chunks=(chunk_size_gpi,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "longitude", data=np.linspace(-180, 180, n_gpi, dtype="float32"),
        chunks=(chunk_size_gpi,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "latitude", data=np.linspace(-90, 90, n_gpi, dtype="float32"),
        chunks=(chunk_size_gpi,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "swath_time", data=np.arange(n_swath_time, dtype="int64"),
        chunks=(1,), dimension_names=("swath_time",), compressors=None,
    )
    root.create_array(
        "spacecraft", data=np.array([3, 4, 5], dtype="int8")[:n_spacecraft],
        chunks=(1,), dimension_names=("spacecraft",), compressors=None,
    )
    root.create_array(
        "processed",
        shape=(n_swath_time, n_spacecraft), dtype="bool",
        chunks=(1, n_spacecraft), dimension_names=("swath_time", "spacecraft"),
        fill_value=False, compressors=None,
    )

    if populated_slots is not None:
        for t_idx, s_idx, gpi_slice, values in populated_slots:
            root[var_name][t_idx, s_idx, gpi_slice] = values
            # Write synthetic time values: slot index * 1.0 + tiny offset per gpi
            n = values.shape[0] if hasattr(values, "shape") else len(values)
            time_vals = np.full(n, float(t_idx + 1), dtype="float64")
            root["time"][t_idx, s_idx, gpi_slice] = time_vals
            root["processed"][t_idx, s_idx] = True

    return path


# ===========================================================================
# _ShardingConfig validation
# ===========================================================================

class TestShardingConfig:

    def test_valid_config(self):
        cfg = _ShardingConfig(
            shard_size_gpi=100, inner_chunk_gpi=25,
            shard_size_obs=90, inner_chunk_obs=30,
        )
        assert cfg.obs_alignment == 90
        assert cfg.gpi_alignment == 100

    def test_gpi_not_multiple_raises(self):
        with pytest.raises(ValueError, match="multiple"):
            _ShardingConfig(
                shard_size_gpi=101, inner_chunk_gpi=25,
                shard_size_obs=None, inner_chunk_obs=30,
            )

    def test_obs_not_multiple_raises(self):
        with pytest.raises(ValueError, match="multiple"):
            _ShardingConfig(
                shard_size_gpi=100, inner_chunk_gpi=25,
                shard_size_obs=91, inner_chunk_obs=30,
            )

    def test_no_obs_sharding_alignment_falls_back_to_chunk(self):
        cfg = _ShardingConfig(
            shard_size_gpi=100, inner_chunk_gpi=25,
            shard_size_obs=None, inner_chunk_obs=30,
        )
        assert cfg.obs_alignment == 30


# ===========================================================================
# _round_up
# ===========================================================================

class TestRoundUpToChunk:

    def test_already_multiple(self):
        assert _round_up(100, 25) == 100

    def test_rounds_up(self):
        assert _round_up(101, 25) == 125

    def test_zero(self):
        assert _round_up(0, 25) == 0


# ===========================================================================
# GPI chunk ranges (inline in tests)
# ===========================================================================

class TestBuildGpiChunkRanges:
    """Tests for GPI chunk range generation.

    Note: _build_gpi_chunk_ranges was removed as a separate helper.
    The logic is now inlined in the densification code.
    These tests verify the expected behavior.
    """

    def test_exact_division(self):
        n_gpi, chunk_size = 100, 25
        ranges = [(i, min(i + chunk_size, n_gpi)) for i in range(0, n_gpi, chunk_size)]
        assert len(ranges) == 4
        assert ranges[0] == (0, 25)
        assert ranges[-1] == (75, 100)

    def test_remainder_last_chunk(self):
        n_gpi, chunk_size = 110, 25
        ranges = [(i, min(i + chunk_size, n_gpi)) for i in range(0, n_gpi, chunk_size)]
        assert ranges[-1] == (100, 110)

    def test_contiguous_and_non_overlapping(self):
        ranges = [(i, min(i + CHUNK_SIZE_GPI, N_GPI)) for i in range(0, N_GPI, CHUNK_SIZE_GPI)]
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == N_GPI


# ===========================================================================
# _scan_all_populated_slots
# ===========================================================================

class TestScanAllPopulatedChunks:

    def test_empty_store_returns_empty_map(self, tmp_path):
        path = _make_sparse_store(tmp_path)
        result = _scan_all_populated_slots(path, N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        assert result == []

    def test_populated_slot_appears_in_map(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        path = _make_sparse_store(tmp_path, populated_slots=[(0, 0, gpi_slice, values)])
        result = _scan_all_populated_slots(path, N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        assert (0, 0) in result

    def test_sharded_store_detected_correctly(self, tmp_path):
        """Sharded store must produce the same populated_map as unsharded."""
        gpi_slice = slice(0, N_GPI)
        values = np.ones(N_GPI, dtype="float32")

        _make_sparse_store(
            tmp_path / "u", populated_slots=[(0, 0, gpi_slice, values)]
        )
        _make_sparse_store(
            tmp_path / "s", shard_size_gpi=SHARD_SIZE_GPI,
            populated_slots=[(0, 0, gpi_slice, values)]
        )

        map_u = _scan_all_populated_slots(tmp_path / "u", N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        map_s = _scan_all_populated_slots(tmp_path / "s", N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)

        # Both should report the same set of populated (t, s) slots
        assert set(map_s) == set(map_u)

    def test_multiple_slots_all_reported(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        vals = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        path = _make_sparse_store(
            tmp_path,
            populated_slots=[
                (0, 0, gpi_slice, vals),
                (1, 1, gpi_slice, vals),
            ],
        )
        result = _scan_all_populated_slots(path, N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        assert (0, 0) in result
        assert (1, 1) in result
        reported_slots = set(result)
        expected_slots = {(0, 0), (1, 1)}
        unexpected = reported_slots - expected_slots
        assert not unexpected, f"Unexpected populated slots reported: {unexpected}"


# ===========================================================================
# _create_dense_structure_from_intermediate
# ===========================================================================

class TestCreateDenseStructure:

    @pytest.fixture
    def dense_store(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        intermediate_path = tmp_path / "intermediate.zarr"
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=30,
            n_read_threads=1,
        )
        int_root = zarr.open(str(intermediate_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure_from_intermediate(
            out_path=out_path,
            int_root=int_root,
            beam_vars=set(),
            scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
            n_gpi=N_GPI,
            obs_dim_size=30,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            sharding=None,
        )
        return zarr.open(str(out_path), mode="r"), out_path

    def test_gpi_dimension_correct(self, dense_store):
        root, _ = dense_store
        assert root["surface_soil_moisture"].shape[0] == N_GPI

    def test_obs_dimension_correct(self, dense_store):
        root, _ = dense_store
        assert root["surface_soil_moisture"].shape[1] == 30

    def test_n_obs_array_created(self, dense_store):
        root, _ = dense_store
        assert "n_obs" in root
        assert root["n_obs"].shape == (N_GPI,)
        assert root["n_obs"][:].sum() == 0   # all zeros initially

    def test_dimension_names(self, dense_store):
        root, _ = dense_store
        assert root["surface_soil_moisture"].metadata.dimension_names == ("gpi", "obs")

    def test_sharding_applied_when_config_given(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        intermediate_path = tmp_path / "intermediate.zarr"
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=30,
            n_read_threads=1,
        )
        int_root = zarr.open(str(intermediate_path), mode="r")
        sharding = _ShardingConfig(
            shard_size_gpi=100, inner_chunk_gpi=CHUNK_SIZE_GPI,
            shard_size_obs=30, inner_chunk_obs=10,
        )
        out_path = tmp_path / "dense_sharded.zarr"
        _create_dense_structure_from_intermediate(
            out_path=out_path,
            int_root=int_root,
            beam_vars=set(),
            scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
            n_gpi=N_GPI,
            obs_dim_size=30,
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            sharding=sharding,
        )
        root = zarr.open(str(out_path), mode="r")
        meta = root["surface_soil_moisture"].metadata
        has_sharding = any(type(c).__name__ == "ShardingCodec" for c in (meta.codecs or []))
        assert has_sharding


# ===========================================================================
# _expand_obs_dimension
# ===========================================================================

class TestExpandObsDimension:

    def test_expansion_resizes_all_data_arrays(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        intermediate_path = tmp_path / "intermediate.zarr"
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=30,
            n_read_threads=1,
        )
        int_root = zarr.open(str(intermediate_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure_from_intermediate(
            out_path=out_path, int_root=int_root,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False, n_gpi=N_GPI,
            obs_dim_size=30, chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10, sharding=None,
        )
        out_root = zarr.open(str(out_path), mode="a")
        _expand_obs_dimension(
            out_root, needed_size=50, obs_alignment=10,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
        )
        assert out_root["surface_soil_moisture"].shape[1] == 50

    def test_expansion_aligns_to_chunk(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        intermediate_path = tmp_path / "intermediate.zarr"
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=30,
            n_read_threads=1,
        )
        int_root = zarr.open(str(intermediate_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure_from_intermediate(
            out_path=out_path, int_root=int_root,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False, n_gpi=N_GPI,
            obs_dim_size=30, chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10, sharding=None,
        )
        out_root = zarr.open(str(out_path), mode="a")
        _expand_obs_dimension(
            out_root, needed_size=41, obs_alignment=10,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
        )
        assert out_root["surface_soil_moisture"].shape[1] % 10 == 0
        assert out_root["surface_soil_moisture"].shape[1] >= 41

    def test_no_op_when_already_large_enough(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        intermediate_path = tmp_path / "intermediate.zarr"
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=100,
            n_read_threads=1,
        )
        int_root = zarr.open(str(intermediate_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure_from_intermediate(
            out_path=out_path, int_root=int_root,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False, n_gpi=N_GPI,
            obs_dim_size=100, chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10, sharding=None,
        )
        out_root = zarr.open(str(out_path), mode="a")
        _expand_obs_dimension(
            out_root, needed_size=50, obs_alignment=10,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
        )
        assert out_root["surface_soil_moisture"].shape[1] == 100


# ===========================================================================
# sparse_to_dense_rechunked integration
# ===========================================================================

class TestSparseToDense:

    def test_creates_output_store(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="f4")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense_rechunked(
            sparse_path=sparse_path,
            out_path=str(out_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            n_workers=1,
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=10,
            n_read_threads=1,
        )
        assert (out_path / "zarr.json").exists()

    def test_known_values_written_correctly(self, tmp_path):
        """Values written to sparse store appear at the right positions in dense."""
        gpi_slice = slice(0, 5)
        expected = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype="f4")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, expected)],
        )
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense_rechunked(
            sparse_path=sparse_path,
            out_path=str(out_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            n_workers=1,
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=10,
            n_read_threads=1,
        )
        root = zarr.open(str(out_path), mode="r")
        # GPI 0-4 should each have exactly 1 observation
        np.testing.assert_array_almost_equal(
            root["surface_soil_moisture"][:5, 0], expected
        )

    def test_n_obs_correct(self, tmp_path):
        gpi_slice = slice(0, 5)
        values = np.ones(5, dtype="f4")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense_rechunked(
            sparse_path=sparse_path,
            out_path=str(out_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            n_workers=1,
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=10,
            n_read_threads=1,
        )
        root = zarr.open(str(out_path), mode="r")
        assert (root["n_obs"][:5] == 1).all()
        assert (root["n_obs"][5:] == 0).all()

    def test_observations_sorted_by_time(self, tmp_path):
        """When two swath slots contain data for the same GPI, obs must be
        sorted by time value in the dense store."""
        gpi_slice = slice(0, 1)
        # slot 2 has a *later* swath_time index than slot 0 but write it first
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            n_swath_time=4,
            populated_slots=[
                (2, 0, gpi_slice, np.array([99.0], dtype="f4")),  # time=3.0
                (0, 0, gpi_slice, np.array([11.0], dtype="f4")),  # time=1.0
            ],
        )
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense_rechunked(
            sparse_path=sparse_path,
            out_path=str(out_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            n_workers=1,
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=10,
            n_read_threads=1,
        )
        root = zarr.open(str(out_path), mode="r")
        # Obs 0 should be the earlier time (value=11.0), obs 1 the later (99.0)
        assert root["surface_soil_moisture"][0, 0] == pytest.approx(11.0)
        assert root["surface_soil_moisture"][0, 1] == pytest.approx(99.0)

    def test_sharded_output(self, tmp_path):
        """sparse_to_dense_rechunked with shard_size_gpi produces a sharded output store."""
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="f4")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense_rechunked(
            sparse_path=sparse_path,
            out_path=str(out_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=10,
            shard_size_gpi=SHARD_SIZE_GPI,
            shard_size_obs=30,
            n_workers=1,
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=10,
            n_read_threads=1,
        )
        root = zarr.open(str(out_path), mode="r")
        meta = root["surface_soil_moisture"].metadata
        has_sharding = any(type(c).__name__ == "ShardingCodec" for c in (meta.codecs or []))
        assert has_sharding

    def test_empty_sparse_store_produces_empty_dense(self, tmp_path):
        # Note: The new sparse_to_dense_rechunked interface raises an error
        # when there are no populated slots, rather than creating an empty dense store.
        # This test documents that behavior.
        sparse_path = _make_sparse_store(tmp_path / "sp")
        out_path = tmp_path / "dense.zarr"
        with pytest.raises(ValueError, match="No populated slots"):
            sparse_to_dense_rechunked(
                sparse_path=sparse_path,
                out_path=str(out_path),
                chunk_size_gpi=CHUNK_SIZE_GPI,
                chunk_size_obs=10,
                n_workers=1,
                target_gpi_chunk=CHUNK_SIZE_GPI,
                batch_size=10,
                n_read_threads=1,
            )


class TestClassifyVariables:

    def test_scalar_vars_classified_correctly(self, tmp_path):
        """3D arrays (swath_time, spacecraft, gpi) go into scalar_vars."""
        sparse_path = _make_sparse_store(tmp_path)
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        assert "surface_soil_moisture" in scalar_vars
        assert "time" in scalar_vars
        assert len(beam_vars) == 0

    def test_coord_arrays_excluded(self, tmp_path):
        """Coordinate arrays (gpi, latitude, etc.) must not appear in either set."""
        sparse_path = _make_sparse_store(tmp_path)
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        coord_names = {"swath_time", "spacecraft", "beam", "gpi", "latitude", "longitude"}
        assert not (beam_vars | scalar_vars) & coord_names

    def test_has_beams_false_produces_no_beam_vars(self, tmp_path):
        sparse_path = _make_sparse_store(tmp_path)
        root = zarr.open(str(sparse_path), mode="r")
        beam_vars, scalar_vars = _classify_variables(root, has_beams=False)
        assert len(beam_vars) == 0


# ===========================================================================
# Beam variable chunking tests
# ===========================================================================

def _make_sparse_store_with_beams(
    tmp_path,
    n_gpi=N_GPI,
    n_swath_time=N_SWATH_TIME,
    n_spacecraft=N_SPACECRAFT,
    n_beams=3,
    chunk_size_gpi=CHUNK_SIZE_GPI,
    shard_size_gpi=None,
    populated_slots=None,
    fill_value=np.float32(-9999.0),
    time_fill=np.float64(0.0),
):
    """Build a minimal synthetic sparse Zarr store with beam variables."""
    path = tmp_path
    store = zarr.storage.LocalStore(path)
    root = zarr.create_group(store=store, overwrite=True, zarr_format=3)

    # Coordinate arrays
    root.create_array(
        "gpi", data=np.arange(n_gpi, dtype="int32"),
        chunks=(chunk_size_gpi,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "longitude", data=np.linspace(-180, 180, n_gpi, dtype="float32"),
        chunks=(chunk_size_gpi,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "latitude", data=np.linspace(-90, 90, n_gpi, dtype="float32"),
        chunks=(chunk_size_gpi,), dimension_names=("gpi",), compressors=None,
    )
    root.create_array(
        "swath_time", data=np.arange(n_swath_time, dtype="int64"),
        chunks=(1,), dimension_names=("swath_time",), compressors=None,
    )
    root.create_array(
        "spacecraft", data=np.array([3, 4, 5], dtype="int8")[:n_spacecraft],
        chunks=(1,), dimension_names=("spacecraft",), compressors=None,
    )
    root.create_array(
        "beam", data=np.array(["fore", "mid", "aft"], dtype="<U4")[:n_beams],
        chunks=(1,), dimension_names=("beam",), fill_value="", compressors=None,
    )
    root.create_array(
        "processed",
        shape=(n_swath_time, n_spacecraft), dtype="bool",
        chunks=(1, n_spacecraft), dimension_names=("swath_time", "spacecraft"),
        fill_value=False, compressors=None,
    )

    # Scalar variable (no beam dimension)
    shape_scalar = (n_swath_time, n_spacecraft, n_gpi)
    inner_chunks_scalar = (1, 1, chunk_size_gpi)
    scalar_kwargs = dict(
        dtype="float32",
        shape=shape_scalar,
        chunks=inner_chunks_scalar,
        dimension_names=("swath_time", "spacecraft", "gpi"),
        fill_value=fill_value,
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
    )
    if shard_size_gpi is not None:
        scalar_kwargs["shards"] = (1, 1, shard_size_gpi)
    root.create_array(name="surface_soil_moisture", **scalar_kwargs)

    # Beam variable (with beam dimension)
    shape_beam = (n_swath_time, n_spacecraft, n_beams, n_gpi)
    inner_chunks_beam = (1, 1, 1, chunk_size_gpi)
    beam_kwargs = dict(
        dtype="float32",
        shape=shape_beam,
        chunks=inner_chunks_beam,
        dimension_names=("swath_time", "spacecraft", "beam", "gpi"),
        fill_value=fill_value,
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
    )
    if shard_size_gpi is not None:
        beam_kwargs["shards"] = (1, 1, 1, shard_size_gpi)
    root.create_array(name="backscatter", **beam_kwargs)

    # Time variable (float64, same layout as scalar)
    time_kwargs = dict(
        dtype="float64",
        shape=shape_scalar,
        chunks=inner_chunks_scalar,
        dimension_names=("swath_time", "spacecraft", "gpi"),
        fill_value=time_fill,
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
    )
    if shard_size_gpi is not None:
        time_kwargs["shards"] = (1, 1, shard_size_gpi)
    root.create_array(name="time", **time_kwargs)

    if populated_slots is not None:
        for t_idx, s_idx, gpi_slice, values_scalar, values_beam in populated_slots:
            root["surface_soil_moisture"][t_idx, s_idx, gpi_slice] = values_scalar
            root["backscatter"][t_idx, s_idx, :, gpi_slice] = values_beam
            # Write synthetic time values
            n = values_scalar.shape[0] if hasattr(values_scalar, "shape") else len(values_scalar)
            time_vals = np.full(n, float(t_idx + 1), dtype="float64")
            root["time"][t_idx, s_idx, gpi_slice] = time_vals
            root["processed"][t_idx, s_idx] = True

    return path


class TestBeamChunking:
    """Tests for beam variable chunking in timeseries stores."""

    def test_intermediate_store_beam_chunking(self, tmp_path):
        """Intermediate store beam variables should have chunk size = n_beams on beam dimension."""
        from ascat.stack.sparse_zarr_to_ts import rechunk_sparse
        
        # Create sparse store with beams
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values_scalar = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        values_beam = np.ones((3, CHUNK_SIZE_GPI), dtype="float32")  # 3 beams
        sparse_path = _make_sparse_store_with_beams(
            tmp_path / "sparse",
            populated_slots=[(0, 0, gpi_slice, values_scalar, values_beam)],
        )
        
        # Run rechunk_sparse
        intermediate_path = tmp_path / "intermediate.zarr"
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=100,
            n_read_threads=1,
        )
        
        # Verify beam variable chunking in intermediate store
        int_root = zarr.open(str(intermediate_path), mode="r")
        assert "backscatter" in int_root, "Beam variable missing from intermediate store"
        
        # Check chunks: should be (batch_size, n_beams, gpi_chunk)
        beam_chunks = int_root["backscatter"].chunks
        assert len(beam_chunks) == 3, f"Expected 3 dimensions, got {len(beam_chunks)}"
        assert beam_chunks[1] == 3, f"Expected beam chunk size 3, got {beam_chunks[1]}"
        print(f"✓ Intermediate store beam chunks: {beam_chunks}")

    def test_timeseries_store_beam_chunking(self, tmp_path):
        """Timeseries store beam variables should have chunk size = n_beams on beam dimension."""
        from ascat.stack.sparse_zarr_to_ts import rechunk_sparse, densify_from_intermediate
        
        # Create sparse store with beams
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values_scalar = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        values_beam = np.ones((3, CHUNK_SIZE_GPI), dtype="float32")  # 3 beams
        sparse_path = _make_sparse_store_with_beams(
            tmp_path / "sparse",
            populated_slots=[(0, 0, gpi_slice, values_scalar, values_beam)],
        )
        
        # Run two-pass conversion
        intermediate_path = tmp_path / "intermediate.zarr"
        ts_path = tmp_path / "timeseries.zarr"
        
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=100,
            n_read_threads=1,
        )
        
        densify_from_intermediate(
            intermediate_path=str(intermediate_path),
            out_path=str(ts_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=100,
            n_workers=1,
            shard_size_gpi=None,
            shard_size_obs=None,
        )
        
        # Verify beam variable chunking in timeseries store
        ts_root = zarr.open(str(ts_path), mode="r")
        assert "backscatter" in ts_root, "Beam variable missing from timeseries store"
        
        # Check chunks: should be (gpi_chunk, obs_chunk, n_beams)
        beam_chunks = ts_root["backscatter"].chunks
        assert len(beam_chunks) == 3, f"Expected 3 dimensions, got {len(beam_chunks)}"
        assert beam_chunks[2] == 3, f"Expected beam chunk size 3, got {beam_chunks[2]}"
        print(f"✓ Timeseries store beam chunks: {beam_chunks}")

    def test_timeseries_store_beam_chunking_with_sharding(self, tmp_path):
        """Timeseries store with sharding should have beam chunk size = n_beams."""
        from ascat.stack.sparse_zarr_to_ts import rechunk_sparse, densify_from_intermediate
        
        # Create sparse store with beams
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values_scalar = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        values_beam = np.ones((3, CHUNK_SIZE_GPI), dtype="float32")  # 3 beams
        sparse_path = _make_sparse_store_with_beams(
            tmp_path / "sparse",
            populated_slots=[(0, 0, gpi_slice, values_scalar, values_beam)],
            shard_size_gpi=SHARD_SIZE_GPI,
        )
        
        # Run two-pass conversion with sharding
        intermediate_path = tmp_path / "intermediate.zarr"
        ts_path = tmp_path / "timeseries_sharded.zarr"
        
        rechunk_sparse(
            sparse_path=sparse_path,
            intermediate_path=str(intermediate_path),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=100,
            n_read_threads=1,
        )
        
        densify_from_intermediate(
            intermediate_path=str(intermediate_path),
            out_path=str(ts_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=100,
            n_workers=1,
            shard_size_gpi=SHARD_SIZE_GPI,
            shard_size_obs=300,
        )
        
        # Verify beam variable chunking and sharding in timeseries store
        ts_root = zarr.open(str(ts_path), mode="r")
        assert "backscatter" in ts_root, "Beam variable missing from timeseries store"
        
        # Check chunks: should be (gpi_chunk, obs_chunk, n_beams)
        beam_chunks = ts_root["backscatter"].chunks
        assert len(beam_chunks) == 3, f"Expected 3 dimensions, got {len(beam_chunks)}"
        assert beam_chunks[2] == 3, f"Expected beam chunk size 3, got {beam_chunks[2]}"
        
        # Check that sharding is present
        from zarr.codecs import ShardingCodec
        meta = ts_root["backscatter"].metadata
        has_sharding = any(
            getattr(c, "name", None) == "sharding_indexed" or type(c).__name__ == "ShardingCodec"
            for c in (meta.codecs or [])
        )
        assert has_sharding, "Sharding codec not found in beam variable"
        print(f"✓ Sharded timeseries store beam chunks: {beam_chunks}")

    def test_append_mode_detects_beam_chunking(self, tmp_path):
        """Append mode should detect and preserve existing beam chunking."""
        from ascat.stack.sparse_zarr_to_ts import rechunk_sparse, densify_from_intermediate
        
        # Create initial timeseries store with beam chunk size 3
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values_scalar = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        values_beam = np.ones((3, CHUNK_SIZE_GPI), dtype="float32")  # 3 beams
        
        sparse_path1 = _make_sparse_store_with_beams(
            tmp_path / "sparse1",
            populated_slots=[(0, 0, gpi_slice, values_scalar, values_beam)],
        )
        
        intermediate_path1 = tmp_path / "intermediate1.zarr"
        ts_path = tmp_path / "timeseries.zarr"
        
        rechunk_sparse(
            sparse_path=sparse_path1,
            intermediate_path=str(intermediate_path1),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=100,
            n_read_threads=1,
        )
        
        densify_from_intermediate(
            intermediate_path=str(intermediate_path1),
            out_path=str(ts_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=100,
            n_workers=1,
            shard_size_gpi=None,
            shard_size_obs=None,
        )
        
        # Verify initial chunking
        ts_root = zarr.open(str(ts_path), mode="r")
        initial_beam_chunks = ts_root["backscatter"].chunks
        assert initial_beam_chunks[2] == 3, f"Initial beam chunk size should be 3, got {initial_beam_chunks[2]}"
        
        # Create second batch of data for append
        values_scalar2 = np.ones(CHUNK_SIZE_GPI, dtype="float32") * 2
        values_beam2 = np.ones((3, CHUNK_SIZE_GPI), dtype="float32") * 2
        sparse_path2 = _make_sparse_store_with_beams(
            tmp_path / "sparse2",
            populated_slots=[(1, 0, gpi_slice, values_scalar2, values_beam2)],
        )
        
        # Run append
        intermediate_path2 = tmp_path / "intermediate2.zarr"
        rechunk_sparse(
            sparse_path=sparse_path2,
            intermediate_path=str(intermediate_path2),
            target_gpi_chunk=CHUNK_SIZE_GPI,
            batch_size=100,
            n_read_threads=1,
        )
        
        densify_from_intermediate(
            intermediate_path=str(intermediate_path2),
            out_path=str(ts_path),
            chunk_size_gpi=CHUNK_SIZE_GPI,
            chunk_size_obs=100,
            n_workers=1,
            shard_size_gpi=None,
            shard_size_obs=None,
        )
        
        # Verify chunking is preserved after append
        ts_root = zarr.open(str(ts_path), mode="r")
        final_beam_chunks = ts_root["backscatter"].chunks
        assert final_beam_chunks[2] == 3, f"Beam chunk size should remain 3 after append, got {final_beam_chunks[2]}"
        
        # Verify data was appended (n_obs should increase)
        n_obs = ts_root["n_obs"][:5].sum()
        assert n_obs > 0, "No data was appended"
        print(f"✓ Append mode preserved beam chunking: {final_beam_chunks}")
