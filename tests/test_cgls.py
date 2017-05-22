# Copyright (c) 2017, Vienna University of Technology (TU Wien),
# Department of Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# All information contained herein is, and remains the property of Vienna
# University of Technology (TU Wien), Department of Geodesy and Geoinformation
# (GEO). The intellectual and technical concepts contained herein are
# proprietary to Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO). Dissemination of this information or
# reproduction of this material is forbidden unless prior written permission
# is obtained from Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).

'''
Tests for reading CGLOPS SWI data.
'''

from ascat.cgls import SWI_TS
import os
import pandas as pd
import numpy as np


def test_swi_ts_reader():

    data_path = os.path.join(
        os.path.dirname(__file__), 'test-data', 'sat', 'cglops', 'swi_ts')
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


def test_swi_ts_qflag_reading():
    data_path = os.path.join(
        os.path.dirname(__file__), 'test-data', 'sat', 'cglops', 'swi_ts')
    rd = SWI_TS(data_path, parameters=['SWI_001', 'QFLAG_001', 'SSF'])
    data = rd.read_ts(3002621, mask_frozen=True)
    # check if QFLAG is correctly read. It should have as many NaN values as
    # SWI
    assert len(data[data.loc[:, 'QFLAG_001'] != np.nan]) > 0
    assert (len(data[data.loc[:, 'QFLAG_001'] == np.nan]) ==
            len(data[data.loc[:, 'SWI_001'] == np.nan]))

if __name__ == "__main__":
    test_swi_ts_reader()
