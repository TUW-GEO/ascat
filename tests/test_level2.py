# Copyright (c) 2017,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Tests for general level 2 readers.
'''

import datetime
import numpy as np
import numpy.testing as nptest
import os
import pytest
import unittest
import sys
from ascat.level2 import AscatL2SsmBufr
from ascat.level2 import AscatL2SsmBufrChunked
from ascat.level2 import AscatL2SsmBufrFile
from ascat.level2 import AscatL2SsmNcFile


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL2SsmBufr_ioclass_kws(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h07')
        self.reader = AscatL2SsmBufr(data_path, msg_name_lookup={65: 'ssm',
                                                                 74: 'ssm mean'})

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2010, 5, 1), datetime.datetime(2010, 5, 1, 12))
        timestamps_should = [datetime.datetime(2010, 5, 1, 8, 33, 1)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2010, 5, 1, 8, 33, 1))

        ssm_should = np.array([51.2, 65.6, 46.2, 56.9, 61.4, 61.5, 58.1, 47.1, 72.7, 13.8, 60.9, 52.1,
                               78.5, 57.8, 56.2, 79.8, 67.7, 53.8, 86.5, 29.4, 50.6, 88.8, 56.9, 68.9,
                               52.4, 64.4, 81.5, 50.5, 84., 79.6, 47.4, 79.5, 46.9, 60.7, 81.3, 52.9,
                               84.5, 25.5, 79.2, 93.3, 52.6, 93.9, 74.4, 91.4, 76.2, 92.5, 80., 88.3,
                               79.1, 97.2, 56.8])

        lats_should = np.array([70.21162, 69.32506, 69.77325, 68.98149, 69.12295, 65.20364, 67.89625,
                                67.79844, 67.69112, 67.57446, 67.44865, 67.23221, 66.97207, 66.7103,
                                66.34695, 65.90996, 62.72462, 61.95761, 61.52935, 61.09884, 60.54359,
                                65.60223, 65.33588, 65.03098, 64.58972, 61.46131, 60.62553, 59.52057,
                                64.27395, 63.80293, 60.6569, 59.72684, 58.74838, 63.42774])

        ssm_mean_should = np.array([0.342,  0.397,  0.402,  0.365,  0.349,  0.354,  0.37,  0.36,
                                    0.445,  0.211,  0.394,  0.354,  0.501,  0.361,  0.366,  0.45,
                                    0.545,  0.329,  0.506,  0.229,  0.404,  0.591,  0.348,  0.433,
                                    0.29,  0.508,  0.643,  0.343,  0.519,  0.61,  0.414,  0.594,
                                    0.399,  0.512,  0.681,  0.457,  0.622,  0.396,  0.572,  0.7,
                                    0.302,  0.722,  0.493,  0.747,  0.521,  0.72,  0.578,  0.718,
                                    0.536,  0.704,  0.466]) * 100

        nptest.assert_allclose(lats[25:-1:30], lats_should, atol=1e-5)
        nptest.assert_allclose(data['ssm'][
                               15:-1:20], ssm_should, atol=0.01)
        nptest.assert_allclose(data['ssm mean'][15:-1:20],
                               ssm_mean_should,
                               atol=0.01)


class Test_AscatL2SsmNcFile(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'nc')
        fname = os.path.join(
            data_path,
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170220041500_53652_eps_o_125_ssm_l2.nc')
        self.reader = AscatL2SsmNcFile(fname)

    def tearDown(self):
        self.reader = None

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read()

        ssm_should = np.array([2.96000004, 0., 0., 0., 0., 0., 0., 0., 0.,
                               1.82999992, 3.32999992, 4.78999996, 4.31999969,
                               2.53999996, 0., 3.83999991, 5.76999998, 1.5,
                               2.44000006, 4.11999989, 2.25999999, 2.65999985,
                               5.5999999, 5.53999996, 4.85999966])

        lats_should = np.array([62.60224, 62.67133, 62.74015, 62.80871, 62.877,
                                62.94502, 63.01276, 63.08024, 63.14743,
                                63.21435, 63.28098, 63.34734, 63.41341,
                                63.47919, 63.54468, 63.60988, 63.67479,
                                63.7394, 63.80372, 63.86773, 63.93144,
                                63.99485, 64.05795, 64.12075, 64.18323])

        ssm_mean_should = np.array([21.26000023, 21.27999878, 21.38999939,
                                    22.43000031, 23.36999893, 24.51000023,
                                    26.01000023, 27.04999924, 26.94999886,
                                    26.63999939, 27.09999847, 27.56999969,
                                    27.43000031, 26.64999962, 26.53999901,
                                    27.48999977, 28.20999908, 28.38999939,
                                    28.79999924, 29.21999931, 30.01000023,
                                    30.97999954, 31.27999878, 31.8599987,
                                    32.05999756])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][
                               :25], ssm_should, atol=1e-5)
        nptest.assert_allclose(data['mean_soil_moisture'][:25],
                               ssm_mean_should,
                               atol=1e-5)


class Test_AscatL2SsmNcFile_vsAscatL2SsmBufrFile(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125')
        fname_nc = os.path.join(
            data_path, 'nc',
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170220041500_53652_eps_o_125_ssm_l2.nc')
        self.reader_nc = AscatL2SsmNcFile(fname_nc)

        fname_bufr = os.path.join(
            data_path, 'bufr', 'M02-ASCA-ASCSMR02-NA-5.0-20170220041500.000000000Z-20170220055656-1207110.bfr')
        self.reader_bufr = AscatL2SsmBufrFile(fname_bufr)

    def tearDown(self):
        self.reader_nc = None
        self.reader_bufr = None

    def test_image_reading(self):
        data_nc, meta, timestamp, lons_nc, lats_nc, time_var_nc = self.reader_nc.read()
        data_bufr, meta, timestamp, lons_bufr, lats_bufr, time_var_bufr = self.reader_bufr.read()

        nptest.assert_allclose(lats_nc, lats_bufr, atol=1e-4)

        nc_bufr_matching = {
            'slope40':                   'Slope At 40 Deg Incidence Angle',
            'sigma40_error':             'Estimated Error In Sigma0 At 40 Deg Incidence Angle',
            'utc_line_nodes':            None,
            'jd':                        'jd',
            'wet_backscatter':           'Wet Backscatter',
            'swath_indicator':           None,
            'frozen_soil_probability':   'Frozen Land Surface Fraction',
            'wetland_flag':              'Inundation And Wetland Fraction',
            # The processing flag definition between BUFR and netCDF is slightly different
            # 'proc_flag1':                'Soil Moisture Processing Flag',
            'proc_flag2':                None,
            'abs_line_number':           None,
            'sat_track_azi':             None,
            'sigma40':                   'Backscatter',
            'soil_moisture':             'Surface Soil Moisture (Ms)',
            'soil_moisture_error':       'Estimated Error In Surface Soil Moisture',
            'rainfall_flag':             'Rain Fall Detection',
            'soil_moisture_sensitivity': 'Soil Moisture Sensitivity',
            'corr_flags':                'Soil Moisture Correction Flag',
            'dry_backscatter':           'Dry Backscatter',
            'aggregated_quality_flag':   None,
            'mean_soil_moisture':        'Mean Surface Soil Moisture',
            'as_des_pass':               None,
            'slope40_error':             'Estimated Error In Slope At 40 Deg Incidence Angle',
            'topography_flag':           'Topographic Complexity',
            'snow_cover_probability':    'Snow Cover'}

        # 'Direction Of Motion Of Moving Observing Platform']
        # BUFR files contain less accurate data so we only compare to one 0.1
        # accuracy.
        for nc_name in nc_bufr_matching:
            bufr_name = nc_bufr_matching[nc_name]
            if bufr_name is None:
                continue

            # flags and probabilities do not have the same NaN value so we mask
            # the invalid values for comparison
            if nc_name in ['snow_cover_probability',
                           'rainfall_flag',
                           'topography_flag']:
                valid = np.where(data_nc[nc_name] != 255)
                data_nc[nc_name] = data_nc[nc_name][valid]
                data_bufr[bufr_name] = data_bufr[bufr_name][valid]

            nptest.assert_allclose(data_nc[nc_name],
                                   data_bufr[bufr_name], atol=0.1)


def test_AscatL2SsmBufrChunked():

    data_path = os.path.join(
        os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'PDU')
    day_search_str = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPB+ASCAT_C_EUMP_%Y%m%d*_125_ssm_l2.bin'
    file_search_str = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPB+ASCAT_C_EUMP_{datetime}*_125_ssm_l2.bin'
    datetime_format = '%Y%m%d%H%M%S'
    filename_datetime_format = (63, 77, '%Y%m%d%H%M%S')
    reader = AscatL2SsmBufrChunked(data_path, month_path_str='',
                                   day_search_str=day_search_str,
                                   file_search_str=file_search_str,
                                   datetime_format=datetime_format,
                                   filename_datetime_format=filename_datetime_format)

    intervals = reader.tstamps_for_daterange(datetime.datetime(2017, 2, 20, 5),
                                             datetime.datetime(2017, 2, 20, 6))
    data = reader.read(intervals[0])
    assert len(data.metadata.keys()) == 3
    assert data.data['jd'].shape == (23145,)
    assert data.lon.shape == (23145,)
    assert data.lat.shape == (23145,)
