# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
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
Tests for level 2 readers.
"""

from datetime import datetime
import numpy as np
import numpy.testing as nptest
import os
import pytest
import unittest
import sys

from ascat.read_native.bufr import AscatL2SsmBufr
from ascat.read_native.bufr import AscatL2SsmBufrChunked
from ascat.read_native.bufr import AscatL2SsmBufrFile
from ascat.read_native.nc import AscatL2SsmNcFile
import ascat.level2 as level2

float32_nan = np.finfo(np.float32).min


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL2SsmBufr_ioclass_kws(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), 'ascat_test_data',
                                 'hsaf', 'h07')
        self.reader = AscatL2SsmBufr(data_path,
                                     msg_name_lookup={65: 'ssm',
                                                      74: 'ssm mean'})

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime(2010, 5, 1), datetime(2010, 5, 1, 12))
        timestamps_should = [datetime(2010, 5, 1, 8, 33, 1)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime(2010, 5, 1, 8, 33, 1))

        ssm_should = np.array([51.2, 65.6, 46.2, 56.9, 61.4, 61.5, 58.1, 47.1,
                               72.7, 13.8, 60.9, 52.1, 78.5, 57.8, 56.2, 79.8,
                               67.7, 53.8, 86.5, 29.4, 50.6, 88.8, 56.9, 68.9,
                               52.4, 64.4, 81.5, 50.5, 84., 79.6, 47.4, 79.5,
                               46.9, 60.7, 81.3, 52.9, 84.5, 25.5, 79.2, 93.3,
                               52.6, 93.9, 74.4, 91.4, 76.2, 92.5, 80., 88.3,
                               79.1, 97.2, 56.8])

        lats_should = np.array([70.21162, 69.32506, 69.77325, 68.98149,
                                69.12295, 65.20364, 67.89625, 67.79844,
                                67.69112, 67.57446, 67.44865, 67.23221,
                                66.97207, 66.7103, 66.34695, 65.90996,
                                62.72462, 61.95761, 61.52935, 61.09884,
                                60.54359, 65.60223, 65.33588, 65.03098,
                                64.58972, 61.46131, 60.62553, 59.52057,
                                64.27395, 63.80293, 60.6569, 59.72684,
                                58.74838, 63.42774])

        ssm_mean_should = np.array([0.342, 0.397, 0.402, 0.365, 0.349,
                                    0.354, 0.37, 0.36, 0.445, 0.211,
                                    0.394, 0.354, 0.501, 0.361, 0.366,
                                    0.45, 0.545, 0.329, 0.506, 0.229,
                                    0.404, 0.591, 0.348, 0.433, 0.29,
                                    0.508, 0.643, 0.343, 0.519, 0.61,
                                    0.414, 0.594, 0.399, 0.512, 0.681,
                                    0.457, 0.622, 0.396, 0.572, 0.7,
                                    0.302, 0.722, 0.493, 0.747, 0.521,
                                    0.72, 0.578, 0.718, 0.536, 0.704,
                                    0.466]) * 100

        nptest.assert_allclose(lats[25:-1:30], lats_should, atol=1e-5)
        nptest.assert_allclose(data['ssm'][
                               15:-1:20], ssm_should, atol=0.01)
        nptest.assert_allclose(data['ssm mean'][15:-1:20],
                               ssm_mean_should,
                               atol=0.01)


class Test_AscatL2SsmBufrFile(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_L2_SM_125', 'bufr', 'Metop_B')
        fname = os.path.join(
            data_path,
            'M01-ASCA-ASCSMR02-NA-5.0-20170220050900.000000000Z-20170220055833-1207110.bfr')
        self.reader = AscatL2SsmBufrFile(fname)

    def tearDown(self):
        self.reader = None

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read()

        ssm_should = np.array(
            [29.2, 30.2, 35.7, 38.6, 37.5, 37.6, 40.5, 44.5, 40.7,
             39.7, 41.5, 38.8, 34.5, 36.8, 39.4, 41.2, 42.4, 42.9,
             39.3, 30.5, 26.7, 26.5, 26.7, 23.9, 26.2])

        lats_should = np.array(
            [64.74398, 64.81854, 64.89284, 64.96688, 65.04066, 65.11416,
             65.18739, 65.26036, 65.33304, 65.40545, 65.47758, 65.54942,
             65.62099, 65.69226, 65.76324, 65.83393, 65.90432, 65.97442,
             66.04422, 66.11371, 66.1829, 66.25177, 66.32034, 66.38859,
             66.45653])

        ssm_mean_should = np.array(
            [36.7, 35.4, 33.4, 32.5, 32.5, 32., 31.2, 29.4, 28.7,
             27.6, 25.8, 25.4, 25.5, 25.3, 24.4, 23.4, 22.3, 21.3,
             20.4, 20.4, 19.9, 19.7, 20.3, 21.5, 22.9])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=1e-5)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=1e-5)

    def test_image_reading_masked(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            ssm_masked=True)

        ssm_should = np.array(
            [15.6, 10.8, 15.3, 15.9, 19.8, 27., 27.8, 26.8, 28.6,
             35.6, 36., 32.3, 27.6, 31.2, 36.8, 13.4, 18.7, 23.1,
             24.5, 22.1, 17.1, 17.9, 17.8, 21.1, 23.])

        lats_should = np.array(
            [54.27036, 54.3167, 54.36279, 54.40862, 54.45419, 54.49951,
             54.54456, 54.58936, 54.6339, 54.67818, 54.72219, 54.76594,
             54.80943, 54.85265, 54.89561, 56.95692, 56.98178, 57.00631,
             57.03053, 57.05442, 57.07799, 57.10123, 57.12415, 57.14675,
             57.16902])

        ssm_mean_should = np.array(
            [24.4, 22.2, 21., 19.6, 18.7, 19.1, 19.1, 19.9, 19.9,
             20.1, 20.9, 21., 19.6, 17.3, 16.9, 25.6, 25.6, 24.9,
             23.6, 22.9, 22.4, 23.2, 24.1, 24.5, 26.1])

        nptest.assert_allclose(lats[10000:10025], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)']
                               [10000:10025], ssm_should,
                               atol=1e-5)
        nptest.assert_allclose(data['Mean Surface Soil Moisture']
                               [10000:10025], ssm_mean_should,
                               atol=1e-5)


class Test_AscatL2SsmNcFile(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_L2_SM_125', 'nc', 'Metop_A')
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

    def test_image_reading_masked(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            ssm_masked=True)

        ssm_should = np.array(
            [33.39999771118164, 27.06999969482422,
             20.649999618530273, 18.28999900817871,
             24.229999542236328, 24.939998626708984,
             23.639999389648438, 20.3799991607666,
             14.15999984741211, 10.059999465942383,
             9.539999961853027, 9.019999504089355,
             9.420000076293945, 12.279999732971191,
             21.529998779296875, 33.880001068115234,
             39.57999801635742, 35.34000015258789,
             38.88999938964844, 44.459999084472656,
             46.66999816894531, 40.12999725341797,
             38.39999771118164, 43.959999084472656,
             33.43000030517578])

        lats_should = np.array(
            [65.11197384, 65.17437784, 65.23645384, 65.29819884, 65.35961083,
             65.42068783, 65.48142683, 65.54182483, 65.60187983, 65.66158983,
             65.72095083, 65.77996183, 65.83861883, 68.62952883, 68.66132883,
             68.69261383, 68.72337983, 68.75362483, 68.78334683, 68.81254383,
             68.84121383, 68.86935383, 68.89696283, 68.92403783, 68.95057683])

        ssm_mean_should = np.array([26.85999870300293, 25.90999984741211,
                                    25.670000076293945, 25.81999969482422,
                                    24.65999984741211, 22.6299991607666,
                                    20.389999389648438, 18.94999885559082,
                                    17.68000030517578, 16.28999900817871,
                                    15.130000114440918, 14.739999771118164,
                                    15.5,
                                    26.51999855041504, 31.529998779296875,
                                    36.09000015258789, 40.36000061035156,
                                    42.61000061035156, 45.529998779296875,
                                    47.939998626708984, 47.45000076293945,
                                    44.689998626708984, 41.12999725341797,
                                    37.59000015258789, 33.09000015258789])

        nptest.assert_allclose(lats[50000:50025], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][50000:50025], ssm_should,
                               atol=1e-5)
        nptest.assert_allclose(data['mean_soil_moisture'][50000:50025],
                               ssm_mean_should,
                               atol=1e-5)


class Test_AscatL2SsmNcFile_vsAscatL2SsmBufrFile(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_L2_SM_125')
        fname_nc = os.path.join(
            data_path, 'nc', 'Metop_A',
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170220041500_53652_eps_o_125_ssm_l2.nc')
        self.reader_nc = AscatL2SsmNcFile(fname_nc)

        fname_bufr = os.path.join(
            data_path, 'bufr', 'Metop_A',
            'M02-ASCA-ASCSMR02-NA-5.0-20170220041500.000000000Z-20170220055656-1207110.bfr')
        self.reader_bufr = AscatL2SsmBufrFile(fname_bufr)

    def tearDown(self):
        self.reader_nc = None
        self.reader_bufr = None

    def test_image_reading(self):
        data_nc, meta, timestamp, lons_nc, lats_nc, time_var_nc = self.reader_nc.read()
        data_bufr, meta, timestamp, lons_bufr, lats_bufr, time_var_bufr = self.reader_bufr.read()

        nptest.assert_allclose(lats_nc, lats_bufr, atol=1e-4)

        nc_bufr_matching = {
            'slope40': 'Slope At 40 Deg Incidence Angle',
            'sigma40_error': 'Estimated Error In Sigma0 At 40 Deg Incidence Angle',
            'utc_line_nodes': None,
            'jd': 'jd',
            'wet_backscatter': 'Wet Backscatter',
            'swath_indicator': None,
            'frozen_soil_probability': 'Frozen Land Surface Fraction',
            'wetland_flag': 'Inundation And Wetland Fraction',
            # The processing flag definition between BUFR and netCDF is slightly different
            # 'proc_flag1':                'Soil Moisture Processing Flag',
            'proc_flag2': None,
            'abs_line_number': None,
            'sat_track_azi': None,
            'sigma40': 'Backscatter',
            'soil_moisture': 'Surface Soil Moisture (Ms)',
            'soil_moisture_error': 'Estimated Error In Surface Soil Moisture',
            'rainfall_flag': 'Rain Fall Detection',
            'soil_moisture_sensitivity': 'Soil Moisture Sensitivity',
            'corr_flags': 'Soil Moisture Correction Flag',
            'dry_backscatter': 'Dry Backscatter',
            'aggregated_quality_flag': None,
            'mean_soil_moisture': 'Mean Surface Soil Moisture',
            'as_des_pass': None,
            'slope40_error': 'Estimated Error In Slope At 40 Deg Incidence Angle',
            'topography_flag': 'Topographic Complexity',
            'snow_cover_probability': 'Snow Cover'}

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
                           'topography_flag',
                           'frozen_soil_probability',
                           'wetland_flag',
                           'snow_cover_probability']:
                valid = np.where(data_nc[nc_name] != 255)
                data_nc[nc_name] = data_nc[nc_name][valid]
                data_bufr[bufr_name] = data_bufr[bufr_name][valid]

            nptest.assert_allclose(data_nc[nc_name],
                                   data_bufr[bufr_name], atol=0.1)


def test_AscatL2SsmBufrChunked():
    data_path = os.path.join(
        os.path.dirname(
            __file__), 'ascat_test_data', 'eumetsat', 'ASCAT_L2_SM_125',
        'PDU', 'Metop_B')
    day_search_str = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPB+ASCAT_C_EUMP_%Y%m%d*_125_ssm_l2.bin'
    file_search_str = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPB+ASCAT_C_EUMP_{datetime}*_125_ssm_l2.bin'
    datetime_format = '%Y%m%d%H%M%S'
    filename_datetime_format = (63, 77, '%Y%m%d%H%M%S')
    reader = AscatL2SsmBufrChunked(data_path, month_path_str='',
                                   day_search_str=day_search_str,
                                   file_search_str=file_search_str,
                                   datetime_format=datetime_format,
                                   filename_datetime_format=filename_datetime_format)

    intervals = reader.tstamps_for_daterange(datetime(2017, 2, 20, 5),
                                             datetime(2017, 2, 20, 6))
    data = reader.read(intervals[0])
    assert len(data.metadata.keys()) == 3
    assert data.data['jd'].shape == (23616,)
    assert data.lon.shape == (23616,)
    assert data.lat.shape == (23616,)


class Test_AscatL2Image(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_generic_reader_data')
        name_b = os.path.join(data_path, 'bufr',
                              'M01-ASCA-ASCSMO02-NA-5.0-20180612035700.000000000Z-20180612044530-1281300.bfr')
        name_e = os.path.join(data_path, 'eps_nat',
                              'ASCA_SMO_02_M01_20180612035700Z_20180612053856Z_N_O_20180612044530Z.nat')
        name_n = os.path.join(data_path, 'nc',
                              'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPB+ASCAT_C_EUMP_20180612035700_29742_eps_o_250_ssm_l2.nc')
        self.image_bufr = level2.AscatL2Image(name_b)
        self.image_eps = level2.AscatL2Image(name_e)
        self.image_nc = level2.AscatL2Image(name_n)

    def tearDown(self):
        self.image_nc = None
        self.image_bufr = None
        self.image_eps = None

    def test_image_reading_all_formats(self):
        self.reader_bufr = self.image_bufr.read()
        self.reader_eps = self.image_eps.read()
        self.reader_nc = self.image_nc.read()

        nptest.assert_allclose(self.reader_bufr.lat, self.reader_eps.lat,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_eps.lat, self.reader_nc.lat,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_nc.lat, self.reader_bufr.lat,
                               atol=1e-4)

        nptest.assert_allclose(self.reader_bufr.lon, self.reader_eps.lon,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_eps.lon, self.reader_nc.lon,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_nc.lon, self.reader_bufr.lon,
                               atol=1e-4)

        matching = ['jd', 'sat_id', 'abs_line_nr', 'abs_orbit_nr', 'node_num',
                    'line_num', 'as_des_pass', 'swath', 'azif', 'azim', 'azia',
                    'incf', 'incm', 'inca', 'sigf', 'sigm', 'siga', 'sm',
                    'sm_noise', 'sm_sensitivity', 'sig40', 'sig40_noise',
                    'slope40', 'slope40_noise', 'dry_backscatter',
                    'wet_backscatter', 'mean_surf_sm', 'correction_flag',
                    'processing_flag', 'aggregated_quality_flag',
                    'snow_cover_probability', 'frozen_soil_probability',
                    'innudation_or_wetland', 'topographical_complexity']

        # lists with no data fields
        bufr_none = ['abs_line_nr', 'abs_orbit_nr', 'aggregated_quality_flag']
        nc_none = ['azif', 'azim', 'azia', 'incf', 'incm', 'inca',
                   'sigf', 'sigm', 'siga', 'processing_flag']

        # BUFR files contain less accurate data so we only compare to one 0.1
        # accuracy.
        for field in matching:

            # difference between the files should not be the case
            if field == 'sig40':
                mask = self.reader_nc.data[field] == float32_nan
                self.reader_bufr.data[field][mask] = float32_nan
                self.reader_eps.data[field][mask] = float32_nan

            # difference between the files should not be the case
            if field in ['snow_cover_probability', 'frozen_soil_probability',
                         'innudation_or_wetland', 'topographical_complexity']:
                mask = self.reader_eps.data[field] == float32_nan
                self.reader_nc.data[field][mask] = float32_nan
                self.reader_eps.data[field][mask] = float32_nan

            if field not in bufr_none:
                nptest.assert_allclose(self.reader_bufr.data[field],
                                       self.reader_eps.data[field], atol=0.1)

            if field not in nc_none:
                nptest.assert_allclose(self.reader_eps.data[field],
                                       self.reader_nc.data[field], atol=0.1)

            if field not in bufr_none and field not in nc_none:
                nptest.assert_allclose(self.reader_nc.data[field],
                                       self.reader_bufr.data[field], atol=0.1)

    def test_image_reading_eps(self):
        self.reader = self.image_eps.read()

        sm_should = np.array(
            [69.11, 74.23, 74.12, 75.95, 76.23, 80.74, 83.45, 84.94, 84.28,
             86.33, 86.19, 86.31, 87.64, 87.92, 90.65, 90.52, 89.71, 89.33,
             91.41, 91.89, 94.51, 70.43, 67.75, 60.54, 69.43])

        lat_should = np.array(
            [64.06651, 64.21156, 64.355545, 64.49845, 64.64026, 64.78095,
             64.9205, 65.05891, 65.19613, 65.33216, 65.46697, 65.600555,
             65.73289, 65.86394, 65.9937, 66.12214, 66.249245, 66.374985,
             66.499344, 66.62231, 66.743835, 69.63313, 69.698105, 69.760895,
             69.821495])

        lon_should = np.array(
            [121.95572, 121.564156, 121.16849, 120.76867, 120.36467,
             119.95644, 119.54396, 119.12719, 118.70608, 118.2806,
             117.85073, 117.41643, 116.97765, 116.53439, 116.08661,
             115.63427, 115.17735, 114.715836, 114.24969, 113.77889,
             113.30343, 96.66666, 96.049965, 95.42956, 94.80551])

        mean_surf_sm_should = np.array(
            [77.97, 77.57, 79.2, 78.38, 77.85, 79.81, 80.72, 81.23, 82.43,
             82.11, 81.93, 82.55, 83.41, 81.84, 81.43, 81.28, 80.37, 79.6,
             79.43, 78.02, 77.49, 42.42, 41.69, 42.99, 47.51])

        jd_should = np.array(
            [2458281.66458332, 2458281.66458332, 2458281.66458332,
             2458281.66458332, 2458281.66458332, 2458281.66458332,
             2458281.66458332, 2458281.66462674, 2458281.66462674,
             2458281.66462674])

        nptest.assert_allclose(self.reader.lat[:25], lat_should, atol=1e-5)
        nptest.assert_allclose(self.reader.lon[:25], lon_should, atol=1e-5)
        nptest.assert_allclose(self.reader.data['sm'][:25],
                               sm_should, atol=1e-5)
        nptest.assert_allclose(self.reader.data['mean_surf_sm'][:25],
                               mean_surf_sm_should, atol=1e-5)
        nptest.assert_allclose(self.reader.data['jd'][35:45],
                               jd_should, atol=1e-5)


class Test_AscatL2Bufr(unittest.TestCase):

    def setUp(self):
        self.data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_generic_reader_data', 'bufr')

        self.image_bufr = level2.AscatL2Bufr(self.data_path, eo_portal=True)

    def tearDown(self):
        self.image_bufr = None

    def test_image_reading(self):
        data, meta, timestamp, lon, lat, time_var = self.image_bufr.read(
            datetime(2018, 6, 12, 3, 57))

        assert lon.shape == (68544,)
        assert lat.shape == (68544,)

    def test_get_orbit_start_date(self):
        filename = os.path.join(self.data_path,
                                'M01-ASCA-ASCSMO02-NA-5.0-20180612035700.000000000Z-20180612044530-1281300.bfr')
        orbit_start = self.image_bufr._get_orbit_start_date(filename)
        orbit_start_should = datetime(2018, 6, 12, 3, 57)

        assert orbit_start == orbit_start_should

    def test_tstamp_for_daterange(self):
        tstamps = self.image_bufr.tstamps_for_daterange(datetime(2018, 6, 12),
                                                        datetime(2018, 6, 13))
        tstamps_should = [datetime(2018, 6, 12, 3, 57)]

        assert tstamps == tstamps_should


class Test_AscatL2Eps(unittest.TestCase):

    def setUp(self):
        self.data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_generic_reader_data', 'eps_nat')

        self.image_eps = level2.AscatL2Eps(self.data_path, eo_portal=True)

    def tearDown(self):
        self.image_eps = None

    def test_image_reading(self):
        data, meta, timestamp, lon, lat, time_var = self.image_eps.read(
            datetime(2018, 6, 12, 3, 57))

        assert lon.shape == (68544,)
        assert lat.shape == (68544,)

    def test_get_orbit_start_date(self):
        filename = os.path.join(self.data_path,
                                'ASCA_SMO_02_M01_20180612035700Z_20180612053856Z_N_O_20180612044530Z.nat')
        orbit_start = self.image_eps._get_orbit_start_date(filename)
        orbit_start_should = datetime(2018, 6, 12, 3, 57)

        assert orbit_start == orbit_start_should

    def test_tstamp_for_daterange(self):
        tstamps = self.image_eps.tstamps_for_daterange(datetime(2018, 6, 12),
                                                       datetime(2018, 6, 13))
        tstamps_should = [datetime(2018, 6, 12, 3, 57)]

        assert tstamps == tstamps_should


class Test_AscatL2Nc(unittest.TestCase):

    def setUp(self):
        self.data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_generic_reader_data', 'nc')

        self.image_nc = level2.AscatL2Nc(self.data_path, eo_portal=True)

    def tearDown(self):
        self.image_nc = None

    def test_image_reading(self):
        data, meta, timestamp, lon, lat, time_var = self.image_nc.read(
            datetime(2018, 6, 12, 3, 57))

        assert lon.shape == (68544,)
        assert lat.shape == (68544,)

    def test_get_orbit_start_date(self):
        filename = os.path.join(self.data_path,
                                'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPB+ASCAT_C_EUMP_20180612035700_29742_eps_o_250_ssm_l2.nc')
        orbit_start = self.image_nc._get_orbit_start_date(filename)
        orbit_start_should = datetime(2018, 6, 12, 3, 57)

        assert orbit_start == orbit_start_should

    def test_tstamp_for_daterange(self):
        tstamps = self.image_nc.tstamps_for_daterange(datetime(2018, 6, 12),
                                                      datetime(2018, 6, 13))
        tstamps_should = [datetime(2018, 6, 12, 3, 57)]

        assert tstamps == tstamps_should
