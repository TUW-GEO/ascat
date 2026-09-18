# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test plotting a full resolution backscatter file.

Reading and preparing the measurements is tested without a plot, as matplotlib
and eomaps are not installed with the package. The plot itself is tested where
they are, see the "plot" dependency group.
"""

from datetime import datetime

import numpy as np
import pytest

from ascat.plot.plot import default_beams
from ascat.plot.plot import read_szf
from ascat.plot.plot import thin
from ascat.plot.plot import to_dataframe

from get_path import get_testdata_path
from test_sca import _name, write_szf

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning")

ASCAT_SZF = ("eumetsat/ASCAT_generic_reader_data/eps_nat/ASCA_SZF_1B_M01"
             "_20180611041800Z_20180611055959Z_N_O_20180611050637Z.nat")


@pytest.fixture
def ascat_file():
    return get_testdata_path() / ASCAT_SZF


@pytest.fixture
def sca_file(tmp_path):
    start = datetime(2026, 1, 2, 3, 4, 5)
    return write_szf(tmp_path / _name("SZF", start), start)


class TestReadSzf:

    def test_ascat(self, ascat_file):
        data, metadata = read_szf(ascat_file)
        assert set(default_beams) <= set(data)
        assert metadata["product_type"] == "SZF"

    def test_sca(self, sca_file):
        data, metadata = read_szf(sca_file)
        assert set(default_beams) <= set(data)
        assert metadata["product_type"] == "SZF"

    def test_neither_instrument(self, tmp_path):
        path = tmp_path / "something_else.nc"
        path.touch()
        with pytest.raises(ValueError, match="neither an ASCAT"):
            read_szf(path)

    def test_the_same_fields_for_both(self, ascat_file, sca_file):
        """One plot routine works for both because the fields are shared."""
        ascat, _ = read_szf(ascat_file)
        sca, _ = read_szf(sca_file)
        shared = set(ascat["lf-vv"].dtype.names) & set(sca["lf-vv"].dtype.names)
        assert {"lon", "lat", "sig", "inc", "azi", "time"} <= shared


class TestToDataFrame:

    def test_beams_below_each_other(self, sca_file):
        data, _ = read_szf(sca_file)
        frame = to_dataframe(data, ["lf-vv", "lm-vv"])
        assert len(frame) == data["lf-vv"].shape[0] + data["lm-vv"].shape[0]
        assert set(frame["beam"]) == {"lf-vv", "lm-vv"}

    def test_all_beams_by_default(self, sca_file):
        data, _ = read_szf(sca_file)
        assert set(to_dataframe(data)["beam"]) == set(data)

    def test_unknown_beam(self, sca_file):
        data, _ = read_szf(sca_file)
        with pytest.raises(KeyError, match="No beam"):
            to_dataframe(data, ["nope"])

    def test_masked_values_become_nan(self, sca_file):
        data, _ = read_szf(sca_file)
        beam = data["lf-vv"]
        beam["sig"][:5] = np.ma.masked
        frame = to_dataframe({"lf-vv": beam}, ["lf-vv"])
        assert frame["sig"][:5].isna().all()


class TestThin:

    def test_few_enough_is_left_alone(self, sca_file):
        data, _ = read_szf(sca_file)
        frame = to_dataframe(data, ["lf-vv"])
        assert thin(frame, len(frame) + 1) is frame

    def test_no_limit(self, sca_file):
        data, _ = read_szf(sca_file)
        frame = to_dataframe(data, ["lf-vv"])
        assert thin(frame, 0) is frame
        assert thin(frame, None) is frame

    def test_every_nth(self, sca_file):
        data, _ = read_szf(sca_file)
        frame = to_dataframe(data, ["lf-vv"])
        thinned = thin(frame, len(frame) // 4)
        assert len(thinned) <= len(frame) // 4 + 1
        assert thinned.iloc[0].equals(frame.iloc[0])


class TestPlot:
    """The plot itself, where matplotlib and eomaps are installed."""

    @pytest.mark.parametrize("parameter", ["sig", "inc"])
    def test_plot_writes_a_map(self, sca_file, tmp_path, parameter):
        pytest.importorskip("eomaps")
        import matplotlib
        matplotlib.use("Agg")
        from ascat.plot.plot import plot_szf

        m = plot_szf(sca_file, parameter=parameter, size=3)
        out = tmp_path / f"{parameter}.png"
        m.f.savefig(out)
        assert out.stat().st_size > 0

    def test_unknown_parameter(self, sca_file):
        pytest.importorskip("eomaps")
        import matplotlib
        matplotlib.use("Agg")
        from ascat.plot.plot import plot_szf

        with pytest.raises(KeyError, match="No field"):
            plot_szf(sca_file, parameter="nope")
