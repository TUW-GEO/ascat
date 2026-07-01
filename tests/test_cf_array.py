# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Unit tests for the CF discrete sampling geometry helpers in ascat.cf_array.

These lock in the behaviour of the array-type detection, the class dispatch and
the conversions between point / indexed ragged / contiguous ragged / orthomulti
timeseries representations, so the module can be refactored safely.
"""

import numpy as np
import pytest
import xarray as xr

from ascat.read_native import generate_test_data as gtd
from ascat.cf_array import (
    cf_array_type,
    cf_array_class,
    check_orthomulti_ts,
    contiguous_to_indexed,
    indexed_to_contiguous,
    RaggedArray,
    TimeseriesPointArray,
    OrthoMultiTimeseriesArray,
)


def contiguous():
    return gtd.contiguous_ragged_ds_2588.copy(deep=True)


def indexed():
    return gtd.indexed_ragged_ds_2588.copy(deep=True)


def point():
    return gtd.swath_ds.copy(deep=True)


def make_orthomulti():
    """A minimal orthomulti timeseries dataset (locations x time)."""
    return xr.Dataset(
        {
            "location_id": (
                "locations",
                np.array([1, 2, 3], dtype=np.int64),
                {"cf_role": "timeseries_id"},
            ),
            "sm": (
                ("locations", "time"),
                np.arange(9, dtype=np.float32).reshape(3, 3),
            ),
        },
        coords={
            "time": np.array(
                ["2020-01-01", "2020-01-02", "2020-01-03"],
                dtype="datetime64[ns]",
            )
        },
    )


# --------------------------------------------------------------------------- #
# type detection / dispatch
# --------------------------------------------------------------------------- #
def test_cf_array_type_detection():
    assert cf_array_type(contiguous()) == "contiguous"
    assert cf_array_type(indexed()) == "indexed"
    assert cf_array_type(point()) == "point"
    assert cf_array_type(make_orthomulti()) == "orthomulti_ts"


def test_cf_array_type_unknown_raises():
    ds = xr.Dataset({"a": ("x", np.arange(3))})
    with pytest.raises(ValueError):
        cf_array_type(ds)


def test_cf_array_class_dispatch():
    assert isinstance(cf_array_class(point(), "point"), TimeseriesPointArray)
    assert isinstance(cf_array_class(indexed(), "indexed"), RaggedArray)
    assert isinstance(cf_array_class(contiguous(), "contiguous"), RaggedArray)
    assert isinstance(
        cf_array_class(make_orthomulti(), "orthomulti_ts"),
        OrthoMultiTimeseriesArray,
    )


def test_cf_array_class_unknown_raises():
    with pytest.raises(ValueError):
        cf_array_class(point(), "not_a_real_type")


def test_check_orthomulti_ts():
    assert check_orthomulti_ts(make_orthomulti())
    assert not check_orthomulti_ts(point())


# --------------------------------------------------------------------------- #
# ragged round trips
# --------------------------------------------------------------------------- #
def test_contiguous_to_indexed_and_back_identical():
    orig = contiguous()
    idx = RaggedArray(orig.copy(deep=True)).to_indexed_ragged()
    assert cf_array_type(idx) == "indexed"
    back = RaggedArray(idx).to_contiguous_ragged()
    assert cf_array_type(back) == "contiguous"
    xr.testing.assert_identical(back, orig)


def test_indexed_to_contiguous_and_back_identical():
    orig = indexed()
    cont = RaggedArray(orig.copy(deep=True)).to_contiguous_ragged()
    assert cf_array_type(cont) == "contiguous"
    back = RaggedArray(cont).to_indexed_ragged()
    assert cf_array_type(back) == "indexed"
    xr.testing.assert_identical(back, orig)


def test_module_level_conversions_round_trip():
    cont = contiguous()
    idx = contiguous_to_indexed(
        cont.copy(deep=True), "time", "locations", "row_size", "locationIndex"
    )
    assert "locationIndex" in idx
    assert "row_size" not in idx
    back = indexed_to_contiguous(
        idx, "time", "locations", "row_size", "locationIndex"
    )
    assert "row_size" in back
    assert "locationIndex" not in back


def test_timeseries_id_detection():
    assert RaggedArray(contiguous()).timeseries_id == "location_id"


def test_conversions_do_not_mutate_input():
    # point -> indexed / contiguous must not modify the caller's dataset
    pt = point()
    before = pt.copy(deep=True)
    TimeseriesPointArray(pt).to_indexed_ragged(timeseries_id="location_id")
    xr.testing.assert_identical(pt, before)
    TimeseriesPointArray(pt).to_contiguous_ragged(timeseries_id="location_id")
    xr.testing.assert_identical(pt, before)

    # ragged conversions must not modify their input either
    cont = contiguous()
    cont_before = cont.copy(deep=True)
    RaggedArray(cont).to_indexed_ragged()
    xr.testing.assert_identical(cont, cont_before)


# --------------------------------------------------------------------------- #
# ragged <-> point
# --------------------------------------------------------------------------- #
def test_ragged_to_point():
    ds = RaggedArray(contiguous()).to_point_array()
    assert ds.attrs["featureType"] == "point"
    # one sample per time step
    assert ds.sizes["time"] == contiguous().sizes["time"]


def test_point_to_indexed_round_trips_values():
    pt = point()
    tpa = TimeseriesPointArray(pt.copy(deep=True))
    idx = tpa.to_indexed_ragged(timeseries_id="location_id")
    assert cf_array_type(idx) == "indexed"
    back = RaggedArray(idx).to_point_array()
    # every location's soil moisture is preserved
    for lid in np.unique(pt["location_id"].values):
        np.testing.assert_array_equal(
            np.sort(
                back["surface_soil_moisture"].values[
                    back["location_id"].values == lid
                ]
            ),
            np.sort(
                pt["surface_soil_moisture"].values[
                    pt["location_id"].values == lid
                ]
            ),
        )


def test_point_to_contiguous_round_trips_values():
    pt = point()
    cont = TimeseriesPointArray(pt.copy(deep=True)).to_contiguous_ragged(
        timeseries_id="location_id"
    )
    assert cf_array_type(cont) == "contiguous"
    back = RaggedArray(cont).to_point_array()
    assert int(back.sizes["obs"]) == int(pt.sizes["obs"])


# --------------------------------------------------------------------------- #
# explicit error on unsupported conversion (regression for the silent-None bug)
# --------------------------------------------------------------------------- #
def test_unsupported_conversion_raises():
    ra = RaggedArray(contiguous())
    ra._ra_type = "bogus"
    with pytest.raises(ValueError):
        ra.to_point_array()
    with pytest.raises(ValueError):
        ra.to_indexed_ragged()
    with pytest.raises(ValueError):
        ra.to_contiguous_ragged()


# --------------------------------------------------------------------------- #
# orthomulti
# --------------------------------------------------------------------------- #
def test_orthomulti_resolves_dims():
    arr = OrthoMultiTimeseriesArray(make_orthomulti())
    assert arr.array_type == "orthomulti_ts"
    assert arr._instance_dimension == "locations"
    # regression: the sample dimension must be resolved (was left as None)
    assert arr._sample_dimension == "time"


def test_orthomulti_set_sample_dimension():
    arr = OrthoMultiTimeseriesArray(make_orthomulti())
    ds = arr.set_sample_dimension("t")
    assert "t" in ds.dims
    assert "time" not in ds.dims


def test_orthomulti_sel_instances():
    arr = OrthoMultiTimeseriesArray(make_orthomulti())
    sel = arr.sel_instances(instance_vals=[2])
    np.testing.assert_array_equal(sel["location_id"].values, [2])


def test_orthomulti_to_point():
    om = make_orthomulti()
    pt = OrthoMultiTimeseriesArray(om).to_point_array()
    assert cf_array_type(pt) == "point"
    # dense: one observation per (location, time) pair
    assert pt.sizes["obs"] == om.sizes["locations"] * om.sizes["time"]
    np.testing.assert_array_equal(
        np.sort(pt["sm"].values), np.sort(om["sm"].values.ravel())
    )


def test_orthomulti_to_indexed_and_contiguous():
    om = make_orthomulti()
    idx = OrthoMultiTimeseriesArray(om).to_indexed_ragged()
    assert cf_array_type(idx) == "indexed"
    cont = OrthoMultiTimeseriesArray(om).to_contiguous_ragged()
    assert cf_array_type(cont) == "contiguous"


def test_orthomulti_point_round_trip():
    # orthomulti -> point -> orthomulti recovers the original values
    om = make_orthomulti()
    pt = OrthoMultiTimeseriesArray(om).to_point_array()
    back = TimeseriesPointArray(pt).to_orthomulti(timeseries_id="location_id")
    np.testing.assert_array_equal(
        back["sm"].sel(location_id=om["location_id"].values).transpose(
            "location_id", "time"
        ).values,
        om["sm"].values,
    )


def test_from_dataset_factory():
    from ascat.cf_array import CFDiscreteGeom
    assert isinstance(
        CFDiscreteGeom.from_dataset(point()), TimeseriesPointArray
    )
    assert isinstance(
        CFDiscreteGeom.from_dataset(contiguous()), RaggedArray
    )
    assert isinstance(
        CFDiscreteGeom.from_dataset(make_orthomulti()),
        OrthoMultiTimeseriesArray,
    )
