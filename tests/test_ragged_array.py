# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Unit tests for the CF ragged-array wrapper classes in ascat.ragged_array.
"""

import numpy as np
import pytest
import xarray as xr

from ascat.utils import dtype_to_nan
from ascat.utils import fill_value
from ascat.utils import vrange
from ascat.utils import pad_to_2d
from ascat.ragged_array import (
    verify_contiguous_ragged,
    verify_indexed_ragged,
    verify_point_array,
    PointData,
    ContiguousRaggedArray,
    IndexedRaggedArray,
    IncompleteMultidimArray,
    OrthogonalMultidimArray,
)

SAMPLE_DIM = "obs"
INSTANCE_DIM = "loc"
COUNT_VAR = "row_size"
INDEX_VAR = "locationIndex"
ROW_SIZE = np.array([2, 1, 3], dtype=np.int32)
INSTANCE_IDS = np.array([10, 20, 30], dtype=np.int64)
N_OBS = int(ROW_SIZE.sum())


def contiguous_ds():
    return xr.Dataset(
        {
            "temperature": ((SAMPLE_DIM,), np.arange(N_OBS, dtype="float32")),
            "flag": ((SAMPLE_DIM,), np.arange(N_OBS, dtype="int32")),
            COUNT_VAR: (
                (INSTANCE_DIM,), ROW_SIZE, {"sample_dimension": SAMPLE_DIM}
            ),
        },
        coords={
            INSTANCE_DIM: (INSTANCE_DIM, INSTANCE_IDS),
            "lon": ((INSTANCE_DIM,), np.array([1.0, 2.0, 3.0], dtype="float32")),
        },
    )


def indexed_ds():
    # same data as contiguous_ds, indexed form
    location_index = np.repeat(np.arange(ROW_SIZE.size), ROW_SIZE).astype("int32")
    return xr.Dataset(
        {
            "temperature": ((SAMPLE_DIM,), np.arange(N_OBS, dtype="float32")),
            "flag": ((SAMPLE_DIM,), np.arange(N_OBS, dtype="int32")),
            INDEX_VAR: (
                (SAMPLE_DIM,), location_index, {"instance_dimension": INSTANCE_DIM}
            ),
        },
        coords={
            INSTANCE_DIM: (INSTANCE_DIM, INSTANCE_IDS),
            "lon": ((INSTANCE_DIM,), np.array([1.0, 2.0, 3.0], dtype="float32")),
        },
    )


def point_ds():
    loc = np.repeat(INSTANCE_IDS, ROW_SIZE)
    return xr.Dataset(
        {"temperature": ((SAMPLE_DIM,), np.arange(N_OBS, dtype="float32"))},
        coords={
            SAMPLE_DIM: np.arange(N_OBS),
            INSTANCE_DIM: ((SAMPLE_DIM,), loc),
        },
    )


TIMES = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")


def orthogonal_contiguous_ds():
    """A contiguous ragged array where every instance shares the same 2 times
    (a complete grid -> can become a true orthogonal array)."""
    n_inst = INSTANCE_IDS.size
    return xr.Dataset(
        {
            "temperature": (
                (SAMPLE_DIM,), np.arange(n_inst * TIMES.size, dtype="float32")
            ),
            COUNT_VAR: (
                (INSTANCE_DIM,), np.full(n_inst, TIMES.size, dtype=np.int32),
                {"sample_dimension": SAMPLE_DIM},
            ),
        },
        coords={
            INSTANCE_DIM: (INSTANCE_DIM, INSTANCE_IDS),
            "time": ((SAMPLE_DIM,), np.tile(TIMES, n_inst)),
        },
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_fill_value():
    assert fill_value(np.dtype("float32")) == dtype_to_nan[np.dtype("float32")]
    assert fill_value(np.dtype("int32")) == dtype_to_nan[np.dtype("int32")]
    assert np.isnat(fill_value(np.dtype("datetime64[ns]")))


def test_vrange():
    np.testing.assert_array_equal(
        vrange(np.array([1, 3, 4, 6]), np.array([1, 5, 7, 6])),
        np.array([3, 4, 4, 5, 6]),
    )


def test_pad_to_2d_float_and_int():
    # float -> sentinel fill, no crash for int
    var = xr.DataArray(np.array([1.0, 2.0, 3.0], dtype="float32"))
    x = np.array([0, 0, 1])
    y = np.array([0, 1, 0])
    out = pad_to_2d(var, x, y, (2, 2))
    assert out[1, 1] == dtype_to_nan[np.dtype("float32")]
    ivar = xr.DataArray(np.array([1, 2, 3], dtype="int32"))
    iout = pad_to_2d(ivar, x, y, (2, 2))
    assert iout[1, 1] == dtype_to_nan[np.dtype("int32")]


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def test_verify_functions_pass():
    verify_contiguous_ragged(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    verify_indexed_ragged(indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    verify_point_array(point_ds(), SAMPLE_DIM)


def test_verify_contiguous_missing_count_var():
    ds = contiguous_ds().drop_vars(COUNT_VAR)
    with pytest.raises(RuntimeError):
        verify_contiguous_ragged(ds, COUNT_VAR, INSTANCE_DIM)


def test_verify_indexed_missing_attr():
    ds = indexed_ds()
    del ds[INDEX_VAR].attrs["instance_dimension"]
    with pytest.raises(RuntimeError):
        verify_indexed_ragged(ds, INDEX_VAR, SAMPLE_DIM)


def test_verify_point_rejects_non_point():
    with pytest.raises(RuntimeError):
        verify_point_array(contiguous_ds(), SAMPLE_DIM)


# --------------------------------------------------------------------------- #
# ContiguousRaggedArray
# --------------------------------------------------------------------------- #
def test_contiguous_basic():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    assert cra.size == 3
    assert set(cra.instance_variables) == {"temperature", "flag"}
    np.testing.assert_array_equal(cra.instance_ids, INSTANCE_IDS)


def test_contiguous_sel_instance():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    ds = cra.sel_instance(30)  # 3rd instance, row_size 3
    assert ds.sizes[SAMPLE_DIM] == 3
    np.testing.assert_array_equal(ds["temperature"].values, [3.0, 4.0, 5.0])
    # unknown id -> None
    assert cra.sel_instance(999) is None


def test_contiguous_iter():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    sizes = [ds.sizes[SAMPLE_DIM] for ds in cra]
    assert sizes == list(ROW_SIZE)


def test_contiguous_sel_instances():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    sel = cra.sel_instances(np.array([30, 10]))  # request order
    np.testing.assert_array_equal(sel["loc"].values, [30, 10])
    # instance 30 -> [3,4,5], instance 10 -> [0,1]
    np.testing.assert_array_equal(sel["temperature"].values,
                                  [3., 4., 5., 0., 1.])
    # missing ids are skipped
    sel2 = cra.sel_instances(np.array([20, 999]))
    np.testing.assert_array_equal(sel2["temperature"].values, [2.])
    assert cra.sel_instances(np.array([999])) is None


# --- large, sparse instance ids (ASCAT location_id scale) --------------------
BIG_IDS = np.array([10, 1_549_346, 3_200_000, 6_599_999], dtype=np.int64)


def big_sparse_contiguous_ds():
    rs = np.array([1, 2, 1, 3], dtype=np.int32)
    n = int(rs.sum())
    return xr.Dataset(
        {"v": ((SAMPLE_DIM,), np.arange(n, dtype="float32")),
         COUNT_VAR: ((INSTANCE_DIM,), rs, {"sample_dimension": SAMPLE_DIM})},
        coords={INSTANCE_DIM: (INSTANCE_DIM, BIG_IDS)},
    )


def big_sparse_indexed_ds():
    rs = np.array([1, 2, 1, 3], dtype=np.int32)
    n = int(rs.sum())
    li = np.repeat(np.arange(BIG_IDS.size), rs).astype("int32")
    return xr.Dataset(
        {"v": ((SAMPLE_DIM,), np.arange(n, dtype="float32")),
         INDEX_VAR: ((SAMPLE_DIM,), li, {"instance_dimension": INSTANCE_DIM})},
        coords={INSTANCE_DIM: (INSTANCE_DIM, BIG_IDS)},
    )


def test_contiguous_large_sparse_ids_selection():
    cra = ContiguousRaggedArray(big_sparse_contiguous_ds(), COUNT_VAR,
                                INSTANCE_DIM)
    np.testing.assert_array_equal(
        cra.sel_instance(1_549_346)["v"].values, [1., 2.])
    np.testing.assert_array_equal(
        cra.sel_instance(6_599_999)["v"].values, [4., 5., 6.])
    assert cra.sel_instance(12_345) is None
    sel = cra.sel_instances(np.array([6_599_999, 10]))
    np.testing.assert_array_equal(sel["loc"].values, [6_599_999, 10])
    np.testing.assert_array_equal(sel["v"].values, [4., 5., 6., 0.])


def test_indexed_large_sparse_ids_selection():
    ira = IndexedRaggedArray(big_sparse_indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    np.testing.assert_array_equal(
        ira.sel_instance(1_549_346)["v"].values, [1., 2.])
    sel = ira.sel_instances(np.array([6_599_999, 10]))
    np.testing.assert_array_equal(sel.instance_ids, [6_599_999, 10])


def test_instance_id_vs_positional_index_modes():
    # MODE 1: the instance dimension carries real location_ids
    cra_id = ContiguousRaggedArray(big_sparse_contiguous_ds(), COUNT_VAR,
                                   INSTANCE_DIM)
    np.testing.assert_array_equal(
        cra_id.sel_instance(6_599_999)["v"].values, [4., 5., 6.])

    # MODE 2: the instance dimension is just a positional index 0..X-1
    rs = np.array([2, 1, 3], dtype=np.int32)
    n = int(rs.sum())
    ds = xr.Dataset(
        {"v": ((SAMPLE_DIM,), np.arange(n, dtype="float32")),
         COUNT_VAR: ((INSTANCE_DIM,), rs, {"sample_dimension": SAMPLE_DIM})},
        coords={INSTANCE_DIM: (INSTANCE_DIM, np.arange(3))},
    )
    cra_idx = ContiguousRaggedArray(ds, COUNT_VAR, INSTANCE_DIM)
    np.testing.assert_array_equal(cra_idx.sel_instance(2)["v"].values,
                                  [3., 4., 5.])
    # lookup memory scales with instance count in both modes;
    # the 0..X-1 case takes the O(1) identity fast path
    assert cra_id._lookup.size == cra_id.size
    assert not cra_id._lookup._identity
    assert cra_idx._lookup.size == cra_idx.size
    assert cra_idx._lookup._identity


def test_multidim_large_sparse_ids_construct():
    # constructing orthogonal/incomplete with large sparse ids must not blow up
    inc = ContiguousRaggedArray(big_sparse_contiguous_ds(), COUNT_VAR,
                                INSTANCE_DIM).to_incomplete()
    np.testing.assert_array_equal(inc.sel_instance(6_599_999)["v"].values[:3],
                                  [4., 5., 6.])


# --------------------------------------------------------------------------- #
# round trips
# --------------------------------------------------------------------------- #
def test_contiguous_to_indexed_index_var_distinct():
    # regression: index var must not collapse onto the sample dimension
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    ira = cra.to_indexed()
    assert ira.index_var == INDEX_VAR
    assert ira.index_var != ira.sample_dim


def test_contiguous_indexed_round_trip():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    back = cra.to_indexed().to_contiguous()
    assert back.count_var == COUNT_VAR
    np.testing.assert_array_equal(back.ds[COUNT_VAR].values, ROW_SIZE)
    np.testing.assert_array_equal(
        back.ds["temperature"].values, np.arange(N_OBS, dtype="float32")
    )


def test_contiguous_to_incomplete_fill():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    om = cra.to_incomplete()
    assert isinstance(om, IncompleteMultidimArray)
    # shape is (n_instances, longest series) = (3, 3)
    assert om.ds["temperature"].shape == (3, 3)
    # padded cells use the project sentinel (regression for the -2**31 bug)
    vals = om.ds["temperature"].values
    assert vals[1, 1] == dtype_to_nan[np.dtype("float32")]
    # real values preserved
    assert vals[0, 0] == 0.0 and vals[2, 2] == 5.0


# --------------------------------------------------------------------------- #
# IndexedRaggedArray
# --------------------------------------------------------------------------- #
def test_indexed_basic():
    ira = IndexedRaggedArray(indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    assert ira.size == 3
    np.testing.assert_array_equal(ira.instance_ids, INSTANCE_IDS)


def test_indexed_to_contiguous_round_trip():
    ira = IndexedRaggedArray(indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    cra = ira.to_contiguous()
    assert cra.count_var == COUNT_VAR
    np.testing.assert_array_equal(cra.ds[COUNT_VAR].values, ROW_SIZE)


def test_indexed_save_and_load(tmp_path):
    ira = IndexedRaggedArray(indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    fn = tmp_path / "indexed.nc"
    ira.save(str(fn))
    loaded = IndexedRaggedArray.from_file(str(fn), INDEX_VAR, SAMPLE_DIM)
    np.testing.assert_array_equal(loaded.instance_ids, INSTANCE_IDS)


def test_indexed_save_unknown_suffix(tmp_path):
    ira = IndexedRaggedArray(indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    with pytest.raises(ValueError):
        ira.save(str(tmp_path / "bad.txt"))


# --------------------------------------------------------------------------- #
# PointData
# --------------------------------------------------------------------------- #
def test_point_to_indexed_and_contiguous():
    pd = PointData(point_ds(), SAMPLE_DIM)
    ira = pd.to_indexed(index_var=INDEX_VAR, instance_dim=INSTANCE_DIM)
    assert isinstance(ira, IndexedRaggedArray)
    np.testing.assert_array_equal(np.sort(ira.instance_ids), INSTANCE_IDS)

    cra = PointData(point_ds(), SAMPLE_DIM).to_contiguous(
        count_var=COUNT_VAR, instance_dim=INSTANCE_DIM
    )
    assert isinstance(cra, ContiguousRaggedArray)
    assert int(cra.ds[COUNT_VAR].sum()) == N_OBS


# --------------------------------------------------------------------------- #
# ragged / orthomulti -> point (full conversion matrix)
# --------------------------------------------------------------------------- #
def test_contiguous_to_point_data():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    pd = cra.to_point_data()
    assert isinstance(pd, PointData)
    assert pd.ds.sizes[SAMPLE_DIM] == N_OBS
    # instance-level lon broadcast to every observation
    np.testing.assert_array_equal(
        pd.ds["lon"].values, np.repeat([1.0, 2.0, 3.0], ROW_SIZE)
    )


def test_indexed_to_point_data():
    ira = IndexedRaggedArray(indexed_ds(), INDEX_VAR, SAMPLE_DIM)
    pd = ira.to_point_data()
    assert isinstance(pd, PointData)
    assert pd.ds.sizes[SAMPLE_DIM] == N_OBS


def test_incomplete_to_contiguous_round_trip():
    # contiguous -> incomplete -> contiguous must recover row_size and data
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    back = cra.to_incomplete().to_contiguous()
    assert isinstance(back, ContiguousRaggedArray)
    np.testing.assert_array_equal(back.ds[COUNT_VAR].values, ROW_SIZE)
    np.testing.assert_array_equal(
        back.ds["temperature"].values, np.arange(N_OBS, dtype="float32")
    )


def test_incomplete_to_indexed_and_point():
    cra = ContiguousRaggedArray(contiguous_ds(), COUNT_VAR, INSTANCE_DIM)
    om = cra.to_incomplete()
    assert isinstance(om.to_indexed(), IndexedRaggedArray)
    pd = om.to_point_data()
    assert isinstance(pd, PointData)
    assert pd.ds.sizes[SAMPLE_DIM] == N_OBS


def test_point_to_incomplete():
    pd = PointData(point_ds(), SAMPLE_DIM)
    om = pd.to_incomplete(count_var=COUNT_VAR, instance_dim=INSTANCE_DIM)
    assert isinstance(om, IncompleteMultidimArray)
    # 3 instances, longest series has 3 samples
    assert om.ds["temperature"].shape == (3, 3)


# --------------------------------------------------------------------------- #
# orthogonal multidimensional array (CF 9.3.1)
# --------------------------------------------------------------------------- #
def test_contiguous_to_orthogonal_shared_axis():
    cra = ContiguousRaggedArray(orthogonal_contiguous_ds(), COUNT_VAR,
                                INSTANCE_DIM)
    om = cra.to_orthogonal("time")
    assert isinstance(om, OrthogonalMultidimArray)
    # complete (3 instances x 2 shared times), no padding
    assert om.ds["temperature"].shape == (3, 2)
    # a shared 1-D time coordinate
    assert om.ds["time"].dims == ("time",)
    np.testing.assert_array_equal(om.ds["time"].values, TIMES)
    # no fill values present
    assert not (om.ds["temperature"].values
                == dtype_to_nan[np.dtype("float32")]).any()


def test_orthogonal_round_trip():
    orig = orthogonal_contiguous_ds()
    cra = ContiguousRaggedArray(orig, COUNT_VAR, INSTANCE_DIM)
    back = cra.to_orthogonal("time").to_contiguous(sample_dim=SAMPLE_DIM)
    np.testing.assert_array_equal(
        back.ds[COUNT_VAR].values, orig[COUNT_VAR].values
    )
    np.testing.assert_array_equal(
        np.sort(back.ds["temperature"].values),
        np.sort(orig["temperature"].values),
    )


def test_orthogonal_strict_raises_on_incomplete_grid():
    # instances with different time axes cannot form an orthogonal array
    cds = xr.Dataset(
        {
            "temperature": ((SAMPLE_DIM,), np.arange(5, dtype="float32")),
            COUNT_VAR: (
                (INSTANCE_DIM,), np.array([2, 1, 2], dtype=np.int32),
                {"sample_dimension": SAMPLE_DIM},
            ),
        },
        coords={
            INSTANCE_DIM: (INSTANCE_DIM, INSTANCE_IDS),
            "time": ((SAMPLE_DIM,), np.array(
                ["2020-01-01", "2020-01-02", "2020-01-01",
                 "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")),
        },
    )
    cra = ContiguousRaggedArray(cds, COUNT_VAR, INSTANCE_DIM)
    with pytest.raises(ValueError):
        cra.to_orthogonal("time", strict=True)


def multidim_contiguous_ds():
    """A contiguous ragged array with a 2-D data variable (extra 'beam' dim)
    and instance coordinates, like real ASCAT cell data."""
    return xr.Dataset(
        {
            "sm": (("beam", SAMPLE_DIM),
                   np.arange(3 * N_OBS, dtype="float32").reshape(3, N_OBS)),
            "backscatter": ((SAMPLE_DIM,), np.arange(N_OBS, dtype="float32")),
            COUNT_VAR: ((INSTANCE_DIM,), ROW_SIZE,
                        {"sample_dimension": SAMPLE_DIM}),
        },
        coords={
            INSTANCE_DIM: (INSTANCE_DIM, INSTANCE_IDS),
            "lon": ((INSTANCE_DIM,), np.array([1., 2., 3.], dtype="float32")),
            "lat": ((INSTANCE_DIM,), np.array([4., 5., 6.], dtype="float32")),
        },
    )


def test_incomplete_preserves_multidim_and_coords():
    # regression: extra dims (beam) and instance coords must survive the
    # round trip through the incomplete multidimensional array
    orig = multidim_contiguous_ds()
    cra = ContiguousRaggedArray(orig, COUNT_VAR, INSTANCE_DIM)
    inc = cra.to_incomplete()
    assert "sm" in inc.ds and inc.ds["sm"].dims == ("beam", INSTANCE_DIM,
                                                     SAMPLE_DIM)
    assert "lon" in inc.ds.coords and "lat" in inc.ds.coords

    back = inc.to_contiguous()
    np.testing.assert_array_equal(back.ds[COUNT_VAR].values, ROW_SIZE)
    assert back.ds["sm"].dims == ("beam", SAMPLE_DIM)
    np.testing.assert_array_equal(back.ds["sm"].values, orig["sm"].values)
    np.testing.assert_array_equal(back.ds["lon"].values, orig["lon"].values)


# --------------------------------------------------------------------------- #
# the conversion functions can be used directly on a raw dataset
# --------------------------------------------------------------------------- #
def test_conversions_usable_without_classes():
    from ascat import cf_conversions as cc

    cds = contiguous_ds()
    idx = cc.contiguous_to_indexed(
        cds, SAMPLE_DIM, INSTANCE_DIM, COUNT_VAR, INDEX_VAR
    )
    assert INDEX_VAR in idx and COUNT_VAR not in idx
    back = cc.indexed_to_contiguous(
        idx, SAMPLE_DIM, INSTANCE_DIM, COUNT_VAR, INDEX_VAR
    )
    np.testing.assert_array_equal(back[COUNT_VAR].values, ROW_SIZE)

    # incomplete round trip
    inc = cc.contiguous_to_incomplete(cds, SAMPLE_DIM, INSTANCE_DIM, COUNT_VAR)
    assert inc["temperature"].shape == (3, 3)
    cont = cc.incomplete_to_contiguous(
        inc, INSTANCE_DIM, SAMPLE_DIM, count_var=COUNT_VAR, sample_dim=SAMPLE_DIM
    )
    np.testing.assert_array_equal(cont[COUNT_VAR].values, ROW_SIZE)

    # orthogonal round trip
    ods = orthogonal_contiguous_ds()
    orth = cc.contiguous_to_orthogonal(
        ods, SAMPLE_DIM, INSTANCE_DIM, COUNT_VAR, "time"
    )
    assert orth["temperature"].shape == (3, 2)
    ocont = cc.orthogonal_to_contiguous(
        orth, INSTANCE_DIM, "time", element_coord="time",
        count_var=COUNT_VAR, sample_dim=SAMPLE_DIM
    )
    np.testing.assert_array_equal(
        ocont[COUNT_VAR].values, ods[COUNT_VAR].values
    )
