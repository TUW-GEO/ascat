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
Test ASCAT Level 1 reader.
"""

import os
import sys
import pytest
import unittest
from datetime import datetime

import numpy as np
import numpy.testing as nptest

from ascat.eumetsat.level1 import AscatL1bFile
from ascat.eumetsat.level1 import AscatL1bNcFileList
from ascat.eumetsat.level1 import AscatL1bEpsFileList
from ascat.eumetsat.level1 import AscatL1bBufrFileList
from ascat.eumetsat.level1 import AscatL1bHdf5FileList

float32_nan = -999999.


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL1bFile(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'eumetsat',
            'ASCAT_generic_reader_data')

        name_b = os.path.join(
            data_path, 'bufr',
            'M02-ASCA-ASCSZR1B0200-NA-9.1-20100609013900.000000000Z-20130824233100-1280350.bfr')
        name_e = os.path.join(
            data_path, 'eps_nat',
            'ASCA_SZR_1B_M02_20100609013900Z_20100609032058Z_R_O_20130824233100Z.nat')
        name_n = os.path.join(
            data_path, 'nc',
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20100609013900_18872_eps_o_125_l1.nc')

        name_e11 = os.path.join(
            data_path, 'eps_nat',
            'ASCA_SZR_1B_M02_20071212071500Z_20071212085659Z_R_O_20081225063118Z.nat')

        name_e_szf = os.path.join(
            data_path, 'eps_nat',
            'ASCA_SZF_1B_M01_20180611041800Z_20180611055959Z_N_O_20180611050637Z.nat')

        name_h = os.path.join(
            data_path, 'hdf5',
            'ASCA_SZF_1B_M01_20180611041800Z_20180611055959Z_N_O_20180611050637Z.h5')

        self.bufr = AscatL1bFile(name_b)
        self.nc = AscatL1bFile(name_n)

        self.eps = AscatL1bFile(name_e)
        self.eps_fmv11 = AscatL1bFile(name_e11)

        self.eps_szf = AscatL1bFile(name_e_szf)
        self.h5_szf = AscatL1bFile(name_h)

    def test_read_szx(self):
        """
        Test SZX data in all data formats (BUFR, EPS Native, NetCDF).
        """
        self.bufr_ds = self.bufr.read()
        self.eps_ds = self.eps.read()
        self.nc_ds = self.nc.read()

        for f in ['lat', 'lon']:
            nptest.assert_allclose(self.bufr_ds[f], self.eps_ds[f], atol=1e-2)
            nptest.assert_allclose(self.eps_ds[f], self.nc_ds[f], atol=1e-2)
            nptest.assert_allclose(self.nc_ds[f], self.bufr_ds[f], atol=1e-2)

        matching = ['sig', 'inc', 'azi', 'kp', 'kp_quality',
                    'swath_indicator', 'f_usable', 'f_land', 'sat_id',
                    'line_num', 'node_num']

        # BUFR contain less accurate data so we only compare 0.1 accuracy.
        for field in matching:

            if field == 'sig':
                nan_mask = (self.nc_ds[field] == float32_nan)
                self.eps_ds[field][nan_mask] = float32_nan
                self.bufr_ds[field][nan_mask] = float32_nan

                valid = ((self.eps_ds[field] > -25))

                nptest.assert_allclose(self.bufr_ds[field][valid],
                                       self.eps_ds[field][valid], atol=0.1)
            else:
                nptest.assert_allclose(self.bufr_ds[field],
                                       self.eps_ds[field], atol=0.1)

            nptest.assert_allclose(self.nc_ds[field],
                                   self.bufr_ds[field], atol=0.1)

            nptest.assert_allclose(self.eps_ds[field],
                                   self.nc_ds[field], atol=0.1)

    def test_read_szf(self):
        """
        Test read SZF formats.
        """
        self.eps_ds = self.eps_szf.read()
        self.h5_ds = self.h5_szf.read()

        for antenna in ['lf', 'lm', 'la', 'rf', 'rm', 'ra']:
            for coord in ['lon', 'lat']:
                nptest.assert_allclose(self.eps_ds[antenna][coord],
                                       self.h5_ds[antenna][coord], atol=1e-4)

            matching = ['sig', 'inc', 'azi', 'sat_id', 'as_des_pass',
                        'beam_number', 'swath_indicator']

            for field in matching:
                nptest.assert_allclose(self.eps_ds[antenna][coord],
                                       self.h5_ds[antenna][coord], atol=0.1)

    def test_szx_eps(self):
        """
        Test read SZX EPS.
        """
        self.reader = self.eps.read(generic=True)

        lat_should = np.array(
            [68.91681, 69.005196, 69.09337, 69.18132, 69.26905, 69.35655,
             69.443825, 69.53087, 69.61768, 69.704254, 69.79059, 69.87668,
             69.962524, 70.04812, 70.13346, 70.218544, 70.303375, 70.38794,
             70.47224, 70.55626, 70.640015, 70.723495, 70.806694, 70.88961,
             70.97224])

        lon_should = np.array(
            [168.80144, 168.60977, 168.41656, 168.22179, 168.02544, 167.82748,
             167.62794, 167.42676, 167.22394, 167.01947, 166.81332, 166.60548,
             166.39592, 166.18465, 165.97163, 165.75685, 165.5403, 165.32195,
             165.10178, 164.87979, 164.65594, 164.43024, 164.20264, 163.97314,
             163.74171])

        sig_should = np.array(
            [-13.510671, -13.421737, -13.872492, -14.351357, -14.395881,
             -14.382635, -14.860762, -16.108913, -17.354418, -18.86383,
             -18.793966, -18.631758, -18.46626, -18.71435, -19.150038,
             -19.315845, -19.79865, -19.845669, -19.892258, -20.138796,
             -20.151554, -20.154343, -20.165552, -20.013523, -19.238102])

        kp_should = np.array(
            [0.0307, 0.032, 0.051, 0.0696, 0.0703, 0.0584, 0.045, 0.0464,
             0.0615, 0.0477, 0.0323, 0.04, 0.0346, 0.0369, 0.0378, 0.0397,
             0.0341, 0.0399, 0.0418, 0.0408, 0.0421, 0.0347, 0.0424, 0.0451,
             0.0523])

        t_should = np.array(
            ['2010-06-09T01:39:00.000', '2010-06-09T01:39:00.000',
             '2010-06-09T01:39:00.000', '2010-06-09T01:39:00.000',
             '2010-06-09T01:39:00.000', '2010-06-09T01:39:00.000',
             '2010-06-09T01:39:00.000', '2010-06-09T01:39:01.874',
             '2010-06-09T01:39:01.874', '2010-06-09T01:39:01.874'],
            dtype='datetime64[ms]')

        nptest.assert_allclose(self.reader['lat'][:25], lat_should, atol=1e-5)
        nptest.assert_allclose(self.reader['lon'][:25], lon_should, atol=1e-5)
        nptest.assert_allclose(
            self.reader['sig'][:25, 0], sig_should, atol=1e-5)
        nptest.assert_allclose(self.reader['kp'][:25, 0], kp_should, atol=1e-5)
        nptest.assert_equal(self.reader['time'][75:85], t_should)

    def test_szx_eps_fmv11(self):
        """
        Test read SZX EPS format version 11.
        """
        self.reader = self.eps_fmv11.read()

        lat_should = np.array(
            [61.849445, 61.916786, 61.983864, 62.050674, 62.11722, 62.183495,
             62.2495, 62.315228, 62.380684, 62.44586, 62.51076, 62.57538,
             62.63971, 62.70376, 62.76752, 62.830994, 62.894173, 62.95706,
             63.019653, 63.081947, 63.143944, 63.205635, 63.267025, 63.32811,
             63.388885])

        lon_should = np.array(
            [69.18133, 68.991295, 68.80043, 68.60872, 68.41617, 68.22277,
             68.028534, 67.833435, 67.63749, 67.44069, 67.243034, 67.04452,
             66.84513, 66.64489, 66.44378, 66.2418, 66.03894, 65.83522,
             65.630615, 65.42514, 65.21878, 65.01154, 64.80342, 64.59441,
             64.38452])

        sig_should = np.array(
            [-15.008125, -14.547356, -15.067405, -15.340037, -15.381483,
             -15.085848, -14.620477, -14.200545, -13.873865, -13.29581,
             -12.962119, -12.909232, -12.990307, -13.076723, -13.039384,
             -13.010556, -13.238036, -13.045113, -12.981088, -13.003889,
             -14.009461, -14.633162, -14.706434, -14.042056, -13.7074])

        kp_should = np.array(
            [0.052, 0.0417, 0.0462, 0.0264, 0.0308, 0.0296, 0.0363, 0.0348,
             0.0377, 0.036, 0.0329, 0.0258, 0.0296, 0.0245, 0.0275, 0.0309,
             0.035, 0.0325, 0.0288, 0.0292, 0.0431, 0.0363, 0.0435, 0.0282,
             0.0309])

        t_should = np.array(
            ['2007-12-12T07:15:00.000', '2007-12-12T07:15:00.000',
             '2007-12-12T07:15:00.000', '2007-12-12T07:15:00.000',
             '2007-12-12T07:15:00.000', '2007-12-12T07:15:00.000',
             '2007-12-12T07:15:00.000', '2007-12-12T07:15:01.876',
             '2007-12-12T07:15:01.876', '2007-12-12T07:15:01.876'],
            dtype='datetime64[ms]')

        nptest.assert_allclose(self.reader['lat'][:25], lat_should, atol=1e-5)
        nptest.assert_allclose(self.reader['lon'][:25], lon_should, atol=1e-5)
        nptest.assert_allclose(
            self.reader['sig'][:25, 0], sig_should, atol=1e-5)
        nptest.assert_allclose(self.reader['kp'][:25, 0], kp_should, atol=1e-5)
        nptest.assert_equal(self.reader['time'][75:85], t_should)

    def test_szf_eps(self):
        """
        Test read SZF EPS format.
        """
        self.reader = self.eps_szf.read()

        lat_should = np.array(
            [64.45502, 64.42318, 64.39127, 64.35929, 64.32724, 64.29512,
             64.262924, 64.23065, 64.19831, 64.16589, 64.1334, 64.10083,
             64.06819, 64.03547, 64.00268, 63.969807, 63.936855, 63.903828,
             63.870724, 63.837536, 63.804276, 63.77093, 63.737507, 63.704002,
             63.67042])

        lon_should = np.array(
            [103.29956, 103.32185, 103.34413, 103.36641, 103.38869,
             103.41095, 103.43322, 103.45548, 103.47774, 103.49999,
             103.52224, 103.54449, 103.566734, 103.588974, 103.61121,
             103.63345, 103.655685, 103.67792, 103.70014, 103.722374,
             103.7446, 103.76682, 103.78904, 103.811264, 103.83348])

        sig_should = np.array(
            [-9.713457, -8.768949, -9.294478, -7.449275, -8.939872,
             -7.893198, -8.570546, -8.934691, -7.851117, -7.782818,
             -8.33993, -7.539894, -7.833797, -8.465893, -8.244121,
             -7.59996, -8.976448, -9.36595, -10.800382, -8.289896,
             -9.127579, -9.410345, -7.238986, -8.335969, -7.897769])

        t_should = np.array(
            ['2018-06-11T04:18:00.630', '2018-06-11T04:18:00.630',
             '2018-06-11T04:18:01.479', '2018-06-11T04:18:01.479',
             '2018-06-11T04:18:01.479', '2018-06-11T04:18:01.479',
             '2018-06-11T04:18:01.479', '2018-06-11T04:18:01.479',
             '2018-06-11T04:18:01.479', '2018-06-11T04:18:01.479'],
            dtype='datetime64[ms]')

        nptest.assert_allclose(self.reader['lf']['lat'][:25],
                               lat_should, atol=1e-5)
        nptest.assert_allclose(self.reader['lf']['lon'][:25],
                               lon_should, atol=1e-5)
        nptest.assert_allclose(self.reader['lf']['sig'][:25],
                               sig_should, atol=1e-5)
        nptest.assert_equal(self.reader['lf']['time'][190:200], t_should)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatL1bFileList(unittest.TestCase):

    """
    Test read AscatL1bFileList in various formats.
    """

    def setUp(self):
        """
        Setup test data.
        """
        root_path = os.path.join(os.path.dirname(__file__), 'ascat_test_data',
                                 'eumetsat', 'ASCAT_generic_reader_data')

        self.bufr_szr = AscatL1bBufrFileList(
            os.path.join(root_path, 'bufr'), sat='A')
        self.nc_szr = AscatL1bNcFileList(os.path.join(root_path, 'nc'))
        self.eps_szr = AscatL1bEpsFileList(os.path.join(root_path, 'eps_nat'))

        self.eps_szf = AscatL1bEpsFileList(
            os.path.join(root_path, 'eps_nat'), sat='B', res='SZF')
        self.hdf5_szf = AscatL1bHdf5FileList(
            os.path.join(root_path, 'hdf5'), sat='B')

    def test_szr_read_date(self):
        """
        Test read date for SZR formats.
        """
        dt = datetime(2010, 6, 9, 1, 39, 0)

        bufr_data = self.bufr_szr.read(dt)
        nc_data = self.nc_szr.read(dt)
        eps_data = self.eps_szr.read(dt)

        for f in ['lat', 'lon']:
            nptest.assert_allclose(bufr_data[f], eps_data[f], atol=1e-2)
            nptest.assert_allclose(eps_data[f], nc_data[f], atol=1e-2)
            nptest.assert_allclose(nc_data[f], bufr_data[f], atol=1e-2)

        matching = ['sig', 'inc', 'azi', 'kp', 'kp_quality',
                    'swath_indicator', 'f_usable', 'f_land', 'sat_id',
                    'line_num', 'node_num']

        # BUFR contain less accurate data so we only compare 0.1 accuracy.
        for field in matching:
            if field == 'sig':
                nan_mask = (nc_data[field] == float32_nan)
                eps_data[field][nan_mask] = float32_nan
                bufr_data[field][nan_mask] = float32_nan
                valid = ((eps_data[field] > -25))

                nptest.assert_allclose(bufr_data[field][valid],
                                       eps_data[field][valid], atol=0.1)
            else:
                nptest.assert_allclose(bufr_data[field],
                                       eps_data[field], atol=0.1)

            nptest.assert_allclose(nc_data[field], bufr_data[field], atol=0.1)
            nptest.assert_allclose(eps_data[field], nc_data[field], atol=0.1)

    def test_szr_read_period(self):
        """
        Test read period for SZR formats.
        """
        dt_start = datetime(2010, 6, 9, 1, 39, 0)
        dt_end = datetime(2010, 6, 9, 2, 0, 0)

        bufr_data = self.bufr_szr.read_period(dt_start, dt_end)
        nc_data = self.nc_szr.read_period(dt_start, dt_end)
        eps_data = self.eps_szr.read_period(dt_start, dt_end)

        for f in ['lat', 'lon']:
            nptest.assert_allclose(bufr_data[f], eps_data[f], atol=1e-2)
            nptest.assert_allclose(eps_data[f], nc_data[f], atol=1e-2)
            nptest.assert_allclose(nc_data[f], bufr_data[f], atol=1e-2)

        matching = ['sig', 'inc', 'azi', 'kp', 'kp_quality',
                    'swath_indicator', 'f_usable', 'f_land', 'sat_id',
                    'line_num', 'node_num']

        # BUFR contain less accurate data so we only compare 0.1 accuracy.
        for field in matching:
            if field == 'sig':
                nan_mask = (nc_data[field] == float32_nan)
                eps_data[field][nan_mask] = float32_nan
                bufr_data[field][nan_mask] = float32_nan
                valid = ((eps_data[field] > -25))

                nptest.assert_allclose(bufr_data[field][valid],
                                       eps_data[field][valid], atol=0.1)
            else:
                nptest.assert_allclose(bufr_data[field],
                                       eps_data[field], atol=0.1)

            nptest.assert_allclose(nc_data[field], bufr_data[field], atol=0.1)
            nptest.assert_allclose(eps_data[field], nc_data[field], atol=0.1)

    def test_szf_read_date(self):
        """
        Test read date for SZF formats.
        """
        dt = datetime(2018, 6, 11, 4, 18, 0)

        eps_data = self.eps_szf.read(dt)
        hdf5_data = self.hdf5_szf.read(dt)

        for antenna in ['lf', 'lm', 'la', 'rf', 'rm', 'ra']:
            for coord in ['lon', 'lat']:
                nptest.assert_allclose(eps_data[antenna][coord],
                                       hdf5_data[antenna][coord], atol=1e-4)

            matching = ['sig', 'inc', 'azi', 'sat_id', 'as_des_pass',
                        'land_frac', 'f_usable', 'f_land', 'beam_number',
                        'swath_indicator']

            for field in matching:
                nptest.assert_allclose(eps_data[antenna][coord],
                                       hdf5_data[antenna][coord], atol=0.1)

    def test_szf_read_period(self):
        """
        Test read period for SZF formats.
        """
        dt_start = datetime(2018, 6, 11, 4, 18, 0)
        dt_end = datetime(2018, 6, 11, 4, 19, 0)

        eps_data = self.eps_szf.read_period(dt_start, dt_end)
        hdf5_data = self.hdf5_szf.read_period(dt_start, dt_end)

        for antenna in ['lf', 'lm', 'la', 'rf', 'rm', 'ra']:
            for coord in ['lon', 'lat']:
                nptest.assert_allclose(eps_data[antenna][coord],
                                       hdf5_data[antenna][coord], atol=1e-4)

            matching = ['sig', 'inc', 'azi', 'sat_id', 'as_des_pass',
                        'land_frac', 'f_usable', 'f_land', 'beam_number',
                        'swath_indicator']

            for field in matching:
                nptest.assert_allclose(eps_data[antenna][coord],
                                       hdf5_data[antenna][coord], atol=0.1)


if __name__ == '__main__':
    unittest.main()
