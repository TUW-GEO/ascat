# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test EPS-SG SCA Level 1b reader.

No SCA test data is shipped with the package, so the files used here are
written on the fly. They follow the structure of the real products: the
variables live in groups, integers are scaled with a "scale_factor" and gaps
are marked with "missing_value" rather than "_FillValue".
"""

from datetime import datetime, timedelta

import numpy as np
import numpy.testing as nptest
import pytest
import netCDF4
import xarray as xr

from ascat.eumetsat.sca import flags
from ascat.eumetsat.sca.level1 import (ScaL1bFile, ScaL1bFileList,
                                       ScaL1bSzfFile, ScaL1bSzrFile,
                                       get_product_type, read_grid,
                                       read_quality, szf_beams, szr_beams)

# netCDF4 1.7.4 trips a NumPy 2.5 deprecation inside its own __setitem__ when
# the test files are written; it says nothing about the reader under test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning")

N_TIME, N_RANGE, N_POINTS, N_BEAMS = 4, 5, 7, 5
EPOCH = "seconds since 2020-01-01 00:00:00.000"


def _scaled(group, name, values, dtype, scale, missing=None, **attrs):
    """Store values the way the SCA products do, as scaled integers."""
    var = group.createVariable(name, dtype, attrs.pop("dims"))
    var.set_auto_maskandscale(False)
    var[:] = np.asarray(values / scale, dtype=dtype)
    var.scale_factor = scale
    var.add_offset = 0.0
    if missing is not None:
        var.missing_value = np.array(missing, dtype=dtype)
    for k, v in attrs.items():
        setattr(var, k, v)


def _status(fid, sensing_start):
    status = fid.createGroup("status")
    inst = status.createGroup("instrument")
    inst.createDimension("mode_items", 1)
    mode = inst.createVariable("instrument_mode", str, ("mode_items",))
    mode[0] = "OPER"
    fid.spacecraft = "SGB1"
    fid.instrument = "SCA"
    fid.product_level = "1B"
    fid.orbit_start = np.uint32(12)
    fid.orbit_end = np.uint32(12)
    fid.sensing_start_time_utc = sensing_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    fid.sensing_end_time_utc = (
        sensing_start + timedelta(seconds=57)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


N_ALONG, N_ACROSS = 3, 6


def _quality(fid, n_beams, extra=False):
    """The file level summary flags, as the "quality" group of the products."""
    q = fid.createGroup("quality")
    q.createDimension("number_beams", n_beams)
    q.createDimension("number_quality_values", 3)
    q.createVariable("flag_summary", "u4", ())[...] = np.uint32(7)
    q.createVariable("flag_generic", "u4", ("number_beams",))[:] = np.full(
        n_beams, 7, dtype="u4")
    q.createVariable("flag_quality", "u4",
                     ("number_beams", "number_quality_values"))[:] = np.tile(
                         np.array([1, 2, 3], dtype="u4"), (n_beams, 1))
    if extra:
        q.createVariable("pentuplet_quality", "u4",
                         ("number_quality_values",))[:] = np.array(
                             [0, 0, N_POINTS], dtype="u4")
        q.createVariable("triplet_quality", "u4",
                         ("number_quality_values",))[:] = np.array(
                             [0, 0, N_POINTS], dtype="u4")
        td = q.createVariable("tdiff", "f8", ())
        td.units = "seconds"
        td[...] = 3450.0
        tf = q.createVariable("tfore", "f8", ())
        tf.units = "seconds"
        tf[...] = 0.0


def write_szf(path, sensing_start=datetime(2026, 1, 2, 3, 4, 5), flag_generic=0):
    """Write a minimal SCA SZF file with all twelve beam groups."""
    with netCDF4.Dataset(path, "w") as fid:
        fid.product_name = path.name
        fid.type = "SZF"
        _status(fid, sensing_start)
        data = fid.createGroup("data")
        for i, group_name in enumerate(szf_beams.values()):
            g = data.createGroup(group_name)
            g.createDimension("time", N_TIME)
            g.createDimension("range", N_RANGE)
            t = g.createVariable("time", "f8", ("time",))
            t.units = EPOCH
            base = (sensing_start - datetime(2020, 1, 1)).total_seconds()
            t[:] = base + np.arange(N_TIME)
            shape = (N_TIME, N_RANGE)
            _scaled(g, "backscatter", np.full(shape, -15.0 - i), "i4", 1e-7,
                    -2147483648, dims=("time", "range"))
            _scaled(g, "longitude", np.linspace(-10, 10, N_TIME * N_RANGE
                                                ).reshape(shape), "i4", 1e-6,
                    -2147483648, dims=("time", "range"), units="degrees_east")
            _scaled(g, "latitude", np.linspace(40, 60, N_TIME * N_RANGE
                                               ).reshape(shape), "i4", 1e-6,
                    -2147483648, dims=("time", "range"), units="degrees_north")
            _scaled(g, "incidence_angle", np.full(shape, 35.0), "i2", 0.01,
                    -32768, dims=("time", "range"), units="degrees")
            _scaled(g, "azimuth_angle", np.full(shape, 180.0), "u2", 0.01,
                    65535, dims=("time", "range"))
            _scaled(g, "lcr", np.full(shape, 0.5), "u2", 1e-4, 65535,
                    dims=("time", "range"))
            fg = g.createVariable("flag_generic", "u4", ("time", "range"))
            fg[:] = np.full(shape, flag_generic, dtype="u4")
            fq = g.createVariable("flag_quality", "u1", ("time", "range"))
            fq[:] = np.ones(shape, dtype="u1")
            fp = g.createVariable("flag_pass", "u1", ("time",))
            fp[:] = np.zeros(N_TIME, dtype="u1")
            fs = g.createVariable("flag_surface", "u1", ("time", "range"))
            fs[:] = np.ones(shape, dtype="u1")
        grid = data.createGroup("grid")
        grid.createDimension("points_along_track", N_ALONG)
        grid.createDimension("points_across_track", N_ACROSS)
        gt = grid.createVariable("time", "f8", ("points_along_track",))
        gt.units = EPOCH
        gt[:] = ((sensing_start - datetime(2020, 1, 1)).total_seconds()
                 + np.arange(N_ALONG))
        gdims = ("points_along_track", "points_across_track")
        gshape = (N_ALONG, N_ACROSS)
        for side, base in (("left", 45.0), ("right", 50.0)):
            _scaled(grid, f"latitude_{side}", np.full(gshape, base), "i4",
                    1e-6, -2147483648, dims=gdims, units="degrees_north")
            _scaled(grid, f"longitude_{side}", np.full(gshape, 5.0), "i4",
                    1e-6, -2147483648, dims=gdims, units="degrees_east")
        _quality(fid, len(szf_beams))
    return path


def write_szr(path, sensing_start=datetime(2026, 1, 2, 3, 4, 5), flag_generic=0):
    """Write a minimal SCA SZR file."""
    with netCDF4.Dataset(path, "w") as fid:
        fid.product_name = path.name
        fid.type = "SZR"
        _status(fid, sensing_start)
        g = fid.createGroup("data")
        g.createDimension("number_points", N_POINTS)
        g.createDimension("number_beams", N_BEAMS)
        t = g.createVariable("time", "f8", ("number_points",))
        t.units = EPOCH
        t[:] = ((sensing_start - datetime(2020, 1, 1)).total_seconds()
                + np.arange(N_POINTS))
        pt, bm = ("number_points",), ("number_points", "number_beams")
        shape = (N_POINTS, N_BEAMS)
        _scaled(g, "backscatter", np.full(shape, -12.0), "i4", 1e-7,
                -2147483648, dims=bm, units="dB")
        _scaled(g, "longitude", np.linspace(-5, 5, N_POINTS), "i4", 1e-6,
                -2147483648, dims=pt, units="degrees_east")
        _scaled(g, "latitude", np.linspace(45, 55, N_POINTS), "i4", 1e-6,
                -2147483648, dims=pt, units="degrees_north")
        _scaled(g, "incidence_angle", np.full(shape, 40.0), "i2", 0.01,
                -32768, dims=bm, units="degrees")
        _scaled(g, "azimuth_angle", np.full(shape, 90.0), "u2", 0.01, 65535,
                dims=bm)
        _scaled(g, "lcr", np.full(shape, 0.25), "u2", 1e-4, 65535, dims=bm)
        _scaled(g, "kp", np.full(shape, 0.05), "u2", 1e-4, 65535, dims=bm)
        _scaled(g, "corrected_cross_pol", np.full(N_POINTS, -25.0), "i4",
                1e-7, -2147483648, dims=pt, units="dB")
        _scaled(g, "faraday_rotation_angle", np.full(N_POINTS, 1.5), "i2",
                0.01, -32768, dims=pt, units="degrees")
        g.createVariable("line_index", "u4", pt)[:] = np.arange(N_POINTS)
        g.createVariable("node_index", "i2", pt)[:] = np.arange(N_POINTS) - 3
        g.createVariable("flag_generic", "u4", bm)[:] = np.full(shape, flag_generic, dtype="u4")
        g.createVariable("flag_quality", "u1", bm)[:] = np.ones(shape, dtype="u1")
        g.createVariable("flag_surface", "u1", bm)[:] = np.zeros(shape, dtype="u1")
        g.createVariable("flag_pass", "u1", bm)[:] = np.ones(shape, dtype="u1")
        _quality(fid, 2 * N_BEAMS, extra=True)
    return path


def _name(product, start):
    return ("W_XX-EUMETSAT-Darmstadt,SAT,SGB1-SCA-1B-{}_C_EUMT_20190101000030"
            "_G_O_{}_{}_O_N___.nc").format(
                product, start.strftime("%Y%m%d%H%M%S"),
                (start + timedelta(seconds=57)).strftime("%Y%m%d%H%M%S"))


@pytest.fixture
def szf_file(tmp_path):
    start = datetime(2026, 1, 2, 3, 4, 5)
    return write_szf(tmp_path / _name("SZF", start), start)


@pytest.fixture
def szr_file(tmp_path):
    start = datetime(2026, 1, 2, 3, 4, 5)
    return write_szr(tmp_path / _name("SZR", start), start)


class TestScaL1bFile:

    def test_dispatch(self, szf_file, szr_file):
        assert isinstance(ScaL1bFile(szf_file), ScaL1bSzfFile)
        assert isinstance(ScaL1bFile(szr_file), ScaL1bSzrFile)

    def test_product_type_from_name(self, szf_file, szr_file):
        assert get_product_type(szf_file) == "SZF"
        assert get_product_type(szr_file) == "SZR"

    def test_unknown_product_type(self, szf_file):
        with pytest.raises(RuntimeError):
            ScaL1bFile(szf_file, product_type="SZX")

    def test_szf_beams(self, szf_file):
        data, metadata = ScaL1bFile(szf_file).read()
        assert list(data) == list(szf_beams)
        assert data["lf-vv"].shape[0] == N_TIME * N_RANGE

    def test_metadata(self, szf_file):
        _, metadata = ScaL1bFile(szf_file).read()
        assert metadata["product_type"] == "SZF"
        assert metadata["spacecraft"] == "SGB1"
        assert metadata["sat_id"] == 6
        assert metadata["sat_name"] == "B1"
        assert metadata["sensor"] == "SCA"
        assert metadata["instrument_mode"] == "OPER"
        assert metadata["sensing_start_time"] == datetime(2026, 1, 2, 3, 4, 5)

    def test_szf_scaling_applied(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read()
        beam = data["lf-vv"]
        nptest.assert_allclose(beam["inc"], 35.0, atol=1e-4)
        nptest.assert_allclose(beam["azi"], 180.0, atol=1e-4)
        nptest.assert_allclose(beam["f_land"], 0.5, atol=1e-4)
        nptest.assert_allclose(beam["lat"].min(), 40.0, atol=1e-4)
        nptest.assert_allclose(beam["lat"].max(), 60.0, atol=1e-4)

    def test_szf_time_repeated_over_range(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read()
        time = data["lf-vv"]["time"]
        assert time.size == N_TIME * N_RANGE
        assert np.unique(time).size == N_TIME
        assert time[0] == np.datetime64("2026-01-02T03:04:05.000")

    def test_swath_indicator_from_beam_name(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read()
        for beam in szf_beams:
            expected = 1 if beam.startswith("r") else 0
            assert np.all(data[beam]["swath_indicator"] == expected)

    def test_non_generic_keeps_original_names(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read(generic=False)
        names = data["lf-vv"].dtype.names
        assert {"backscatter", "longitude", "latitude", "incidence_angle",
                "lcr", "flag_quality"} <= set(names)
        assert not {"sig", "lon", "lat", "inc", "f_land", "f_usable"} & set(names)

    def test_generic_template(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read()
        assert {"time", "lon", "lat", "sig", "inc", "azi", "f_land",
                "f_usable", "as_des_pass", "swath_indicator",
                "sat_id"} <= set(data["lf-vv"].dtype.names)

    def test_to_xarray(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read(to_xarray=True)
        ds = data["lf-vv"]
        assert isinstance(ds, xr.Dataset)
        assert ds.sizes["obs"] == N_TIME * N_RANGE
        assert set(ds.coords) == {"lon", "lat", "time"}

    def test_szr_shape(self, szr_file):
        data, metadata = ScaL1bFile(szr_file).read()
        assert metadata["product_type"] == "SZR"
        assert data.shape[0] == N_POINTS
        assert data["sig"].shape == (N_POINTS, N_BEAMS)
        assert len(szr_beams) == N_BEAMS

    def test_szr_to_xarray(self, szr_file):
        data, _ = ScaL1bFile(szr_file).read(to_xarray=True)
        assert data.sizes == {"obs": N_POINTS, "beam": N_BEAMS}

    def test_merge(self, tmp_path):
        starts = [datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 1, 2, 3, 5, 5)]
        files = [write_szf(tmp_path / _name("SZF", s), s) for s in starts]
        data, metadata = ScaL1bFile(files).read()
        assert data["lf-vv"].shape[0] == 2 * N_TIME * N_RANGE
        assert len(metadata) == 2
        assert data["lf-vv"].dtype.names is not None

    def test_toi(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read(
            toi=(datetime(2026, 1, 2, 3, 4, 4), datetime(2026, 1, 2, 3, 4, 7)))
        assert data["lf-vv"].shape[0] == 2 * N_RANGE

    def test_roi(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read(roi=(40.0, -10.0, 50.0, 10.0))
        beam = data["lf-vv"]
        assert beam.shape[0] > 0
        assert beam["lat"].max() <= 50.0


class TestScaFlags:

    def test_summary_is_highest_category(self):
        fg = np.array([0, 1 << 5, 1 << 6, (1 << 5) | (1 << 6)], dtype=np.uint32)
        nptest.assert_array_equal(flags.set_flags(fg), [0, 1, 2, 2])

    def test_rfi_red(self):
        fg = np.array([1 << 7], dtype=np.uint32)
        assert flags.set_flags(fg)[0] == flags.RED
        assert flags.set_flags(fg, rfi_red=False)[0] == flags.NOMINAL

    def test_rfi_does_not_mask_other_red_flags(self):
        fg = np.array([(1 << 7) | (1 << 6)], dtype=np.uint32)
        assert flags.set_flags(fg, rfi_red=False)[0] == flags.RED

    def test_classification_bits_are_nominal(self):
        for name in ("land", "water", "asc", "desc", "pof"):
            bit = flags.flag_bits[name][0]
            fg = np.array([1 << bit], dtype=np.uint32)
            assert flags.set_flags(fg)[0] == flags.NOMINAL

    def test_ignore(self):
        fg = np.array([1 << flags.flag_bits["vnoise"][0]], dtype=np.uint32)
        assert flags.set_flags(fg)[0] == flags.RED
        assert flags.set_flags(fg, ignore=("vnoise",))[0] == flags.NOMINAL

    def test_unknown_flag_rejected(self):
        with pytest.raises(KeyError):
            flags.set_flags(np.array([0], dtype=np.uint32), ignore=("nope",))

    def test_masks_disjoint(self):
        amber, red = flags.category_masks(flags.flag_bits)
        assert amber & red == 0

    def test_shape_and_dtype_preserved(self):
        fg = np.zeros((3, 5), dtype=np.uint32)
        out = flags.set_flags(fg)
        assert out.shape == (3, 5)
        assert out.dtype == np.int8


class TestScaFlagKwargs:

    def test_no_kwarg_means_no_user_flag(self, szf_file):
        data, _ = ScaL1bFile(szf_file).read()
        assert "f_usable_user" not in data["lf-vv"].dtype.names

    def test_kwarg_adds_user_flag_and_keeps_product_flag(self, tmp_path):
        start = datetime(2026, 1, 2, 3, 4, 5)
        path = write_szf(tmp_path / _name("SZF", start), start,
                         flag_generic=1 << 6)
        data, _ = ScaL1bFile(path).read(flag_kwargs={"rfi_red": False})
        beam = data["lf-vv"]
        assert "f_usable_user" in beam.dtype.names
        # the product flag is written as 1 in the test file and must survive
        assert np.all(beam["f_usable"] == 1)
        assert np.all(beam["f_usable_user"] == flags.RED)

    def test_szr_user_flag_keeps_beam_dimension(self, szr_file):
        data, _ = ScaL1bFile(szr_file).read(flag_kwargs={"rfi_red": False})
        assert data["f_usable_user"].shape == (N_POINTS, N_BEAMS)


class TestScaL1bFileList:

    def test_search_and_read_period(self, tmp_path):
        starts = [datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 1, 2, 3, 5, 5),
                  datetime(2026, 1, 2, 3, 6, 5)]
        for s in starts:
            write_szf(tmp_path / _name("SZF", s), s)
        fl = ScaL1bFileList(tmp_path, sat="SGB1", product="szf")
        kw = {"date_field_fmt": "%Y%m%d%H%M%S"}
        found = fl.search_period(datetime(2026, 1, 2), datetime(2026, 1, 3), **kw)
        assert len(found) == 3

        data, metadata = fl.read_period(
            datetime(2026, 1, 2, 3, 5, 0), datetime(2026, 1, 2, 3, 6, 0),
            dt_delta=timedelta(hours=1), dt_buffer=timedelta(hours=1),
            end_inclusive=False, **kw)
        assert data["lf-vv"].shape[0] > 0
        assert data["lf-vv"]["time"].min() >= np.datetime64("2026-01-02T03:05:00")

    def test_sat_vocabulary(self, tmp_path):
        for name in ("b1", "B1", "SGB1", "METOP-SG B1"):
            assert ScaL1bFileList(tmp_path, sat=name).sat == "SGB1"


class TestScaQuality:

    def test_quality_in_metadata(self, szf_file):
        _, metadata = ScaL1bFile(szf_file).read()
        assert metadata["quality_flag_summary"] == 7
        assert metadata["quality_flag_generic"].shape == (len(szf_beams),)
        assert metadata["quality_flag_quality"].shape == (len(szf_beams), 3)

    def test_szr_carries_the_extra_summaries(self, szr_file):
        _, metadata = ScaL1bFile(szr_file).read()
        nptest.assert_array_equal(metadata["quality_pentuplet_quality"],
                                  [0, 0, N_POINTS])
        nptest.assert_array_equal(metadata["quality_triplet_quality"],
                                  [0, 0, N_POINTS])
        assert metadata["quality_tdiff"] == 3450.0
        assert metadata["quality_tfore"] == 0.0

    def test_read_quality_without_reading_measurements(self, szf_file):
        quality = read_quality(szf_file)
        assert set(quality) == {"quality_flag_summary", "quality_flag_generic",
                                "quality_flag_quality"}

    def test_read_quality_matches_a_full_read(self, szr_file):
        quality = read_quality(szr_file)
        _, metadata = ScaL1bFile(szr_file).read()
        for key, value in quality.items():
            nptest.assert_array_equal(value, metadata[key])


class TestScaGrid:

    def test_read_grid(self, szf_file):
        grid = read_grid(szf_file)
        assert set(grid) == {"time", "latitude_left", "longitude_left",
                             "latitude_right", "longitude_right"}
        assert grid["latitude_left"].shape == (N_ALONG, N_ACROSS)
        assert grid["time"].shape == (N_ALONG,)
        nptest.assert_allclose(grid["latitude_left"], 45.0, atol=1e-4)
        nptest.assert_allclose(grid["latitude_right"], 50.0, atol=1e-4)

    def test_read_grid_to_xarray(self, szf_file):
        grid = read_grid(szf_file, to_xarray=True)
        assert grid.sizes == {"along_track": N_ALONG, "across_track": N_ACROSS}

    def test_szr_has_no_grid(self, szr_file):
        with pytest.raises(KeyError, match="no grid"):
            read_grid(szr_file)


class TestScaAttrs:

    def test_dataset_can_be_written(self, szf_file, tmp_path):
        data, _ = ScaL1bFile(szf_file).read(to_xarray=True)
        data["lf-vv"].to_netcdf(tmp_path / "out.nc")

    def test_unserialisable_metadata_left_out_of_attrs(self, szr_file):
        data, metadata = ScaL1bFile(szr_file).read(to_xarray=True)
        # the per beam counts are two-dimensional, netCDF cannot store them
        assert "quality_flag_quality" in metadata
        assert "quality_flag_quality" not in data.attrs
        assert "quality_flag_generic" in data.attrs
