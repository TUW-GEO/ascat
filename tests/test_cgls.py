# Copyright (c) 2021, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Tests for reading CGLOPS SWI data.
"""

from ascat.cgls import SWI_TS
import os
import pandas as pd
import numpy as np
import pytest


def test_swi_ts_reader():
    """
    Test SWI time series reader.
    """
    data_path = os.path.join(
        os.path.dirname(__file__), 'ascat_test_data', 'cglops', 'swi_ts')

    rd = SWI_TS(data_path)
    data = rd.read_ts(3002621, mask_frozen=False)
    data_sorted = data.sort_index()

    assert np.all(data_sorted.index == data.index)
    # just check if enough data is there
    reference_index = pd.date_range('20070101T12:00:00', '20161231T12:00:00')
    assert len(data) == len(reference_index)
    assert np.all(data_sorted.index == reference_index)

    lon, lat = rd.grid.gpi2lonlat(3002621)
    data = rd.read_ts(lon, lat, mask_frozen=False)
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
    data_path = os.path.join(
        os.path.dirname(__file__), 'ascat_test_data', 'cglops', 'swi_ts_non_existing')

    with pytest.raises(IOError):
        SWI_TS(data_path)


def test_swi_ts_qflag_reading():
    """
    Test SWI time series quality flag reader.
    """
    data_path = os.path.join(
        os.path.dirname(__file__), 'ascat_test_data', 'cglops', 'swi_ts')
    rd = SWI_TS(data_path, parameters=['SWI_001', 'QFLAG_001', 'SSF'])
    data = rd.read_ts(3002621, mask_frozen=True)
    # check if QFLAG is correctly read. It should have as many NaN values as
    # SWI
    assert len(data[data.loc[:, 'QFLAG_001'] != np.nan]) > 0
    assert (len(data[data.loc[:, 'QFLAG_001'] == np.nan]) ==
            len(data[data.loc[:, 'SWI_001'] == np.nan]))


if __name__ == "__main__":
    test_swi_ts_reader()
    test_swi_ts_qflag_reading()
    test_swi_ts_reader_no_data_in_folder()
