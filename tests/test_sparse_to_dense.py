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
    _build_gpi_chunk_ranges,
    _classify_variables,
    _create_dense_structure,
    _expand_obs_dimension,
    _round_up_to_chunk,
    _scan_all_populated_chunks,
    sparse_to_dense,
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
        "spacecraft", data=np.array([3, 4], dtype="int8")[:n_spacecraft],
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
# _round_up_to_chunk
# ===========================================================================

class TestRoundUpToChunk:

    def test_already_multiple(self):
        assert _round_up_to_chunk(100, 25) == 100

    def test_rounds_up(self):
        assert _round_up_to_chunk(101, 25) == 125

    def test_zero(self):
        assert _round_up_to_chunk(0, 25) == 0


# ===========================================================================
# _build_gpi_chunk_ranges
# ===========================================================================

class TestBuildGpiChunkRanges:

    def test_exact_division(self):
        ranges = _build_gpi_chunk_ranges(100, 25)
        assert len(ranges) == 4
        assert ranges[0] == (0, 25)
        assert ranges[-1] == (75, 100)

    def test_remainder_last_chunk(self):
        ranges = _build_gpi_chunk_ranges(110, 25)
        assert ranges[-1] == (100, 110)

    def test_contiguous_and_non_overlapping(self):
        ranges = _build_gpi_chunk_ranges(N_GPI, CHUNK_SIZE_GPI)
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == N_GPI


# ===========================================================================
# _scan_all_populated_chunks
# ===========================================================================

class TestScanAllPopulatedChunks:

    def test_empty_store_returns_empty_map(self, tmp_path):
        path = _make_sparse_store(tmp_path)
        result = _scan_all_populated_chunks(path, N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        assert result == {}

    def test_populated_slot_appears_in_map(self, tmp_path):
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="float32")
        path = _make_sparse_store(tmp_path, populated_slots=[(0, 0, gpi_slice, values)])
        result = _scan_all_populated_chunks(path, N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        assert 0 in result
        assert (0, 0) in result[0]

    def test_sharded_store_detected_correctly(self, tmp_path):
        """Sharded store must produce the same populated_map as unsharded."""
        gpi_slice = slice(0, N_GPI)
        values = np.ones(N_GPI, dtype="float32")

        path_unsharded = tmp_path / "unsharded"
        path_sharded = tmp_path / "sharded"

        _make_sparse_store(
            tmp_path / "u", populated_slots=[(0, 0, gpi_slice, values)]
        )
        _make_sparse_store(
            tmp_path / "s", shard_size_gpi=SHARD_SIZE_GPI,
            populated_slots=[(0, 0, gpi_slice, values)]
        )

        map_u = _scan_all_populated_chunks(tmp_path / "u", N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        map_s = _scan_all_populated_chunks(tmp_path / "s", N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)

        # Both maps should report the same set of populated (t, s) slots for each gc
        for gc in map_u:
            assert set(map_s.get(gc, [])) == set(map_u[gc])

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
        result = _scan_all_populated_chunks(path, N_SWATH_TIME, N_SPACECRAFT, N_GPI, CHUNK_SIZE_GPI)
        assert (0, 0) in result[0]
        assert (1, 1) in result[0]


# ===========================================================================
# _create_dense_structure
# ===========================================================================

class TestCreateDenseStructure:

    @pytest.fixture
    def dense_store(self, tmp_path):
        sparse_path = _make_sparse_store(tmp_path)
        sparse_root = zarr.open(str(sparse_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure(
            out_path=out_path,
            sparse_root=sparse_root,
            beam_vars=set(),
            scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
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
        sparse_path = _make_sparse_store(tmp_path / "sp")
        sparse_root = zarr.open(str(sparse_path), mode="r")
        sharding = _ShardingConfig(
            shard_size_gpi=100, inner_chunk_gpi=CHUNK_SIZE_GPI,
            shard_size_obs=30, inner_chunk_obs=10,
        )
        out_path = tmp_path / "dense_sharded.zarr"
        _create_dense_structure(
            out_path=out_path,
            sparse_root=sparse_root,
            beam_vars=set(),
            scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False,
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
        sparse_path = _make_sparse_store(tmp_path)
        sparse_root = zarr.open(str(sparse_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure(
            out_path=out_path, sparse_root=sparse_root,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False, obs_dim_size=30, chunk_size_gpi=CHUNK_SIZE_GPI,
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
        sparse_path = _make_sparse_store(tmp_path)
        sparse_root = zarr.open(str(sparse_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure(
            out_path=out_path, sparse_root=sparse_root,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False, obs_dim_size=30, chunk_size_gpi=CHUNK_SIZE_GPI,
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
        sparse_path = _make_sparse_store(tmp_path)
        sparse_root = zarr.open(str(sparse_path), mode="r")
        out_path = tmp_path / "dense.zarr"
        _create_dense_structure(
            out_path=out_path, sparse_root=sparse_root,
            beam_vars=set(), scalar_vars={"surface_soil_moisture", "time"},
            has_beams=False, obs_dim_size=100, chunk_size_gpi=CHUNK_SIZE_GPI,
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
# sparse_to_dense integration
# ===========================================================================

class TestSparseToDense:

    def test_creates_output_store(self, tmp_path):
        sparse_path = _make_sparse_store(tmp_path / "sp")
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense(sparse_path, out_path, chunk_size_gpi=CHUNK_SIZE_GPI,
                        chunk_size_obs=10, n_workers=1)
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
        sparse_to_dense(sparse_path, out_path, chunk_size_gpi=CHUNK_SIZE_GPI,
                        chunk_size_obs=10, n_workers=1)
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
        sparse_to_dense(sparse_path, out_path, chunk_size_gpi=CHUNK_SIZE_GPI,
                        chunk_size_obs=10, n_workers=1)
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
        sparse_to_dense(sparse_path, out_path, chunk_size_gpi=CHUNK_SIZE_GPI,
                        chunk_size_obs=10, n_workers=1)
        root = zarr.open(str(out_path), mode="r")
        # Obs 0 should be the earlier time (value=11.0), obs 1 the later (99.0)
        assert root["surface_soil_moisture"][0, 0] == pytest.approx(11.0)
        assert root["surface_soil_moisture"][0, 1] == pytest.approx(99.0)

    def test_gpi_mask_skips_masked_gpis(self, tmp_path):
        gpi_slice = slice(0, 10)
        values = np.ones(10, dtype="f4")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        mask = np.zeros(N_GPI, dtype=bool)
        mask[0] = True   # mask out GPI 0 only
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense(sparse_path, out_path, chunk_size_gpi=CHUNK_SIZE_GPI,
                        chunk_size_obs=10, n_workers=1, gpi_mask=mask)
        root = zarr.open(str(out_path), mode="r")
        assert root["n_obs"][0] == 0   # masked — no data
        assert root["n_obs"][1] == 1   # not masked

    def test_gpi_mask_wrong_shape_raises(self, tmp_path):
        sparse_path = _make_sparse_store(tmp_path / "sp")
        with pytest.raises(ValueError, match="shape"):
            sparse_to_dense(
                sparse_path, tmp_path / "dense.zarr",
                gpi_mask=np.zeros(N_GPI + 1, dtype=bool),
            )

    def test_sharded_output(self, tmp_path):
        """sparse_to_dense with shard_size_gpi produces a sharded output store."""
        gpi_slice = slice(0, CHUNK_SIZE_GPI)
        values = np.ones(CHUNK_SIZE_GPI, dtype="f4")
        sparse_path = _make_sparse_store(
            tmp_path / "sp",
            populated_slots=[(0, 0, gpi_slice, values)],
        )
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense(
            sparse_path, out_path,
            chunk_size_gpi=CHUNK_SIZE_GPI, chunk_size_obs=10,
            shard_size_gpi=SHARD_SIZE_GPI, shard_size_obs=30,
            n_workers=1,
        )
        root = zarr.open(str(out_path), mode="r")
        meta = root["surface_soil_moisture"].metadata
        has_sharding = any(type(c).__name__ == "ShardingCodec" for c in (meta.codecs or []))
        assert has_sharding

    def test_empty_sparse_store_produces_empty_dense(self, tmp_path):
        sparse_path = _make_sparse_store(tmp_path / "sp")
        out_path = tmp_path / "dense.zarr"
        sparse_to_dense(sparse_path, out_path, chunk_size_gpi=CHUNK_SIZE_GPI,
                        chunk_size_obs=10, n_workers=1)
        root = zarr.open(str(out_path), mode="r")
        assert root["n_obs"][:].sum() == 0
