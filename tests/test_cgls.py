# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Tests for reading CGLOPS SWI data.
"""

import pandas as pd
import numpy as np
import pytest

from get_path import get_testdata_path
from ascat.cgls import SWI_TS


def test_swi_ts_reader():
    """
    Test SWI time series reader.
    """
    data_path = get_testdata_path() / "cglops" / "swi_ts"

    rd = SWI_TS(data_path)
    data = rd.read(3002621, mask_frozen=False)
    data_sorted = data.sort_index()

    assert np.all(data_sorted.index == data.index)
    # just check if enough data is there
    reference_index = pd.date_range('20070101T12:00:00', '20161231T12:00:00')
    assert len(data) == len(reference_index)
    assert np.all(data_sorted.index == reference_index)

    lon, lat = rd.grid.gpi2lonlat(3002621)
    data = rd.read(lon, lat, mask_frozen=False)
    data_sorted = data.sort_index()

    assert np.all(data_sorted.index == data.index)
    # just check if enough data is there
    reference_index = pd.date_range('20070101T12:00:00', '20161231T12:00:00')
    assert len(data) == len(reference_index)
    assert np.all(data_sorted.index == reference_index)


def test_swi_ts_reader_no_data_in_folder():
    """
    Test SWI time series reader when no data is in folder.
    """
    data_path = get_testdata_path() / "cglops" / "swi_ts_non_existing"

    with pytest.raises(IOError):
        SWI_TS(data_path)


def test_swi_ts_qflag_reading():
    """
    Test SWI time series quality flag reader.
    """
    data_path = get_testdata_path() / "cglops" / "swi_ts"
    rd = SWI_TS(data_path, parameters=['SWI_001', 'QFLAG_001', 'SSF'])
    data = rd.read(3002621, mask_frozen=True)
    # check if QFLAG is correctly read. It should have as many NaN values as
    # SWI
    assert len(data[data.loc[:, 'QFLAG_001'] != np.nan]) > 0
    assert (len(data[data.loc[:, 'QFLAG_001'] == np.nan]) == len(
        data[data.loc[:, 'SWI_001'] == np.nan]))


if __name__ == "__main__":
    test_swi_ts_reader()
    test_swi_ts_qflag_reading()
    test_swi_ts_reader_no_data_in_folder()
