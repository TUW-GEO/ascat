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
Test ASCAT Level 2 reader.
"""

import os
import sys
import pytest
import unittest
from datetime import datetime

import numpy as np
import numpy.testing as nptest

from ascat.read_native.bufr import AscatL2BufrFile
from ascat.read_native.nc import AscatL2NcFile
from ascat.eumetsat.level2 import AscatL2File
from ascat.eumetsat.level2 import AscatL2NcFileList
from ascat.eumetsat.level2 import AscatL2BufrFileList
from ascat.eumetsat.level2 import AscatL2EpsFileList

eps_float_nan = -2147483648.
bufr_float_nan = 1.7e+38
uint8_nan = np.iinfo(np.uint8).max
uint16_nan = np.iinfo(np.uint16).max
float32_nan = -999999.


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL2BufrFile(unittest.TestCase):

    def setUp(self):
        """
        Setup test files.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_L2_SM_125', 'bufr', 'Metop_B')

        fname = os.path.join(
            data_path, 'M01-ASCA-ASCSMR02-NA-5.0-20170220050900.000000000Z-20170220055833-1207110.bfr')

        self.reader = AscatL2BufrFile(fname)

    def test_read(self):
        """
        Test read.
        """
        data = self.reader.read()

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

        nptest.assert_allclose(data['lat'][:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][:25],
                               ssm_should, atol=1e-5)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should, atol=1e-5)


class Test_AscatL2NcFile(unittest.TestCase):

    def setUp(self):
        """
        Setup test files.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_L2_SM_125', 'nc', 'Metop_A')
        fname = os.path.join(
            data_path,
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170220041500_53652_eps_o_125_ssm_l2.nc')
        self.reader = AscatL2NcFile(fname)

    def test_read(self):
        """
        Test read.
        """
        data = self.reader.read()

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

        nptest.assert_allclose(data['latitude'][:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][:25],
                               ssm_should, atol=1e-5)
        nptest.assert_allclose(data['mean_soil_moisture'][:25],
                               ssm_mean_should, atol=1e-5)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL2NcFile_AscatL2BufrFile(unittest.TestCase):

    def setUp(self):
        """
        Setup test files.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_L2_SM_125')
        fname_nc = os.path.join(
            data_path, 'nc', 'Metop_A',
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170220041500_53652_eps_o_125_ssm_l2.nc')
        self.reader_nc = AscatL2NcFile(fname_nc)

        fname_bufr = os.path.join(
            data_path, 'bufr', 'Metop_A',
            'M02-ASCA-ASCSMR02-NA-5.0-20170220041500.000000000Z-20170220055656-1207110.bfr')
        self.reader_bufr = AscatL2BufrFile(fname_bufr)

    def test_read(self):
        """
        Test read.
        """
        data_nc = self.reader_nc.read()
        data_bufr = self.reader_bufr.read()

        nptest.assert_allclose(data_nc['latitude'], data_bufr['lat'],
                               atol=1e-4)

        nc_bufr_matching = {
            'slope40': 'Slope At 40 Deg Incidence Angle',
            'sigma40_error': 'Estimated Error In Sigma0 At 40 Deg Incidence Angle',
            'utc_line_nodes': None,
            'wet_backscatter': 'Wet Backscatter',
            'swath_indicator': None,
            'frozen_soil_probability': 'Frozen Land Surface Fraction',
            'wetland_flag': 'Inundation And Wetland Fraction',
            # The processing flag definition between BUFR and netCDF is slightly different
            # 'proc_flag1': 'Soil Moisture Processing Flag',
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

        # BUFR contains less accurate data so we only compare to 0.1
        for nc_name in nc_bufr_matching:

            bufr_name = nc_bufr_matching[nc_name]
            if bufr_name is None:
                continue

            if nc_name in ['mean_soil_moisture']:
                valid = ((data_nc[nc_name] != uint16_nan) &
                         (data_bufr[bufr_name] != bufr_float_nan))
            elif nc_name in ['snow_cover_probability', 'rainfall_flag',
                             'topography_flag', 'frozen_soil_probability',
                             'wetland_flag', 'snow_cover_probability']:
                valid = ((data_nc[nc_name] != uint8_nan) &
                         (data_bufr[bufr_name] != bufr_float_nan))
            else:
                valid = ((data_nc[nc_name] != eps_float_nan) &
                         (data_bufr[bufr_name] != bufr_float_nan))

            nptest.assert_allclose(data_nc[nc_name][valid],
                                   data_bufr[bufr_name][valid], atol=0.1)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL2File(unittest.TestCase):

    def setUp(self):
        """
        Setup test files.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_generic_reader_data')

        name_b = os.path.join(
            data_path, 'bufr',
            'M01-ASCA-ASCSMO02-NA-5.0-20180612035700.000000000Z-20180612044530-1281300.bfr')
        name_e = os.path.join(
            data_path, 'eps_nat',
            'ASCA_SMO_02_M01_20180612035700Z_20180612053856Z_N_O_20180612044530Z.nat')
        name_n = os.path.join(
            data_path, 'nc',
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPB+ASCAT_C_EUMP_20180612035700_29742_eps_o_250_ssm_l2.nc')

        self.bufr = AscatL2File(name_b)
        self.eps = AscatL2File(name_e)
        self.nc = AscatL2File(name_n)

    def test_read_all_formats(self):
        """
        Test read.
        """
        bufr_ds = self.bufr.read()
        eps_ds = self.eps.read()
        nc_ds = self.nc.read()

        for coord in ['lon', 'lat']:
            nptest.assert_allclose(bufr_ds[coord], eps_ds[coord], atol=1e-4)
            nptest.assert_allclose(eps_ds[coord], nc_ds[coord], atol=1e-4)
            nptest.assert_allclose(nc_ds[coord], bufr_ds[coord], atol=1e-4)

        matching = ['sm', 'sm_noise', 'sm_mean', 'sig40', 'sig40_noise',
                    'slope40', 'slope40_noise', 'dry_sig40', 'wet_sig40',
                    'azi', 'sig', 'inc', 'sm_sens', 'snow_prob', 'frozen_prob',
                    'wetland', 'topo', 'sat_id', 'proc_flag', 'agg_flag',
                    'corr_flag', 'line_num', 'node_num', 'sat_id',
                    'swath_indicator']

        # rounding issues in sat_track_azi leads to different as_des_pass
        # 'as_des_pass', 'sat_track_azi'

        # lists with no data fields
        nc_none = ['azi', 'inc', 'sig', 'corr_flag', 'proc_flag']

        # BUFR contain less accurate data so we only compare to 0.1
        for field in matching:

            # difference between the files should not be the case
            if field == 'sig40':
                mask = nc_ds[field] == float32_nan
                bufr_ds[field][mask] = float32_nan
                eps_ds[field][mask] = float32_nan

                nptest.assert_allclose(bufr_ds[field], eps_ds[field], atol=0.1)

            if field not in nc_none:
                nptest.assert_allclose(eps_ds[field], nc_ds[field], atol=0.1)
                nptest.assert_allclose(nc_ds[field], bufr_ds[field], atol=0.1)

    def test_eps(self):
        """
        Test read EPS.
        """
        eps_ds = self.eps.read()

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

        sm_mean_should = np.array(
            [77.97, 77.57, 79.2, 78.38, 77.85, 79.81, 80.72, 81.23, 82.43,
             82.11, 81.93, 82.55, 83.41, 81.84, 81.43, 81.28, 80.37, 79.6,
             79.43, 78.02, 77.49, 42.42, 41.69, 42.99, 47.51])

        t_should = np.array(
            ['2018-06-12T03:56:59.999', '2018-06-12T03:56:59.999',
             '2018-06-12T03:56:59.999', '2018-06-12T03:56:59.999',
             '2018-06-12T03:56:59.999', '2018-06-12T03:56:59.999',
             '2018-06-12T03:56:59.999', '2018-06-12T03:57:03.750',
             '2018-06-12T03:57:03.750', '2018-06-12T03:57:03.750'],
            dtype='datetime64[ms]')

        nptest.assert_allclose(eps_ds['lat'][:25], lat_should, atol=1e-5)
        nptest.assert_allclose(eps_ds['lon'][:25], lon_should, atol=1e-5)
        nptest.assert_allclose(eps_ds['sm'][:25], sm_should, atol=1e-5)
        nptest.assert_allclose(eps_ds['sm_mean'][:25],
                               sm_mean_should, atol=1e-5)
        nptest.assert_equal(eps_ds['time'][35:45], t_should)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL2FileList(unittest.TestCase):

    """
    Test read AscatL2FileList in various formats.
    """

    def setUp(self):
        """
        Setup test data.
        """
        root_path = os.path.join(os.path.dirname(__file__), 'ascat_test_data',
                                 'eumetsat', 'ASCAT_generic_reader_data')

        self.bufr_smo = AscatL2BufrFileList(
            os.path.join(root_path, 'bufr'), sat='B', res='SMO')
        self.nc_smo = AscatL2NcFileList(
            os.path.join(root_path, 'nc'), sat='B', res='SMO')
        self.eps_smo = AscatL2EpsFileList(
            os.path.join(root_path, 'eps_nat'), sat='B', res='SMO')

    def test_smo_read_date(self):
        """
        Test read date for SMO formats.
        """
        dt = datetime(2018, 6, 12, 3, 57, 0)

        bufr_data = self.bufr_smo.read(dt)
        nc_data = self.nc_smo.read(dt)
        eps_data = self.eps_smo.read(dt)

        for coord in ['lon', 'lat']:
            nptest.assert_allclose(
                bufr_data[coord], eps_data[coord], atol=1e-4)
            nptest.assert_allclose(
                eps_data[coord], nc_data[coord], atol=1e-4)
            nptest.assert_allclose(
                nc_data[coord], bufr_data[coord], atol=1e-4)

    def test_smo_read_period(self):
        """
        Test read period for SMO formats.
        """
        dt_start = datetime(2018, 6, 12, 4, 0, 0)
        dt_end = datetime(2018, 6, 12, 4, 13, 0)

        bufr_data = self.bufr_smo.read_period(dt_start, dt_end)
        nc_data = self.nc_smo.read_period(dt_start, dt_end)
        eps_data = self.eps_smo.read_period(dt_start, dt_end)

        for coord in ['lon', 'lat']:
            nptest.assert_allclose(
                bufr_data[coord], eps_data[coord], atol=1e-4)
            nptest.assert_allclose(
                eps_data[coord], nc_data[coord], atol=1e-4)
            nptest.assert_allclose(
                nc_data[coord], bufr_data[coord], atol=1e-4)


if __name__ == '__main__':
    unittest.main()
