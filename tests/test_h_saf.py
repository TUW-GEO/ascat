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

import os
import sys
import pytest
import unittest
from datetime import datetime

import numpy as np
import numpy.testing as nptest

from ascat.h_saf import H08Bufr
from ascat.h_saf import H08BufrFileList
from ascat.h_saf import H14GribFileList
from ascat.h_saf import AscatNrtBufrFileList
from ascat.h_saf import AscatSsmDataRecord


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H08(unittest.TestCase):

    """
    Test H08 reading.
    """

    def setUp(self):
        """
        Setup test data.
        """
        self.root_path = os.path.join(
            os.path.dirname(__file__), 'ascat_test_data', 'hsaf', 'h08')

    def test_read(self):
        """
        Test read file.
        """
        filename = os.path.join(self.root_path, 'h08_201005_buf',
                                'h08_20100501_083301_metopa_18322_ZAMG.buf')

        h08 = H08Bufr(filename)
        data = h08.read()

        assert sorted(data.keys()) == sorted(['lon', 'lat', 'ssm', 'corr_flag',
                                              'ssm_noise', 'proc_flag'])
        for var_name, var in data.items():
            assert var.shape == (3120, 7680)

    def test_read_files(self):
        """
        Test read files.
        """
        h08_files = H08BufrFileList(self.root_path)

        dt = datetime(2010, 5, 1, 8, 33, 1)
        data = h08_files.read(dt)

        assert sorted(data.keys()) == sorted(['lon', 'lat', 'ssm', 'corr_flag',
                                              'ssm_noise', 'proc_flag'])
        for var_name, var in data.items():
            assert var.shape == (3120, 7680)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_AscatNrtSsm(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.root_path = os.path.join(
            os.path.dirname(__file__),  'ascat_test_data', 'hsaf')

    def test_h16_read(self):
        """
        Test read file.
        """
        dt = datetime(2017, 2, 20, 11, 15, 0)
        h16 = AscatNrtBufrFileList(os.path.join(self.root_path, 'h16'), 'h16')
        data = h16.read(dt)

        sm_should = np.array([0., 3.6, 7.8, 8.2, 12.3, 14.7, 21.6, 26.7,
                              30.6, 32.2, 43., 50.5, 46.3, 47.6, 58.])

        lats_should = np.array([-28.25222, -28.21579, -28.1789, -28.14155,
                                -28.10374, -28.06547, -28.02674, -27.98755,
                                -27.94791, -27.90782, -27.86727, -27.82627,
                                -27.78482, -27.74292, -27.70058, ])

        sm_mean_should = np.array([32.8, 35.1, 37.8, 39.2, 39., 38., 36.2,
                                   39.6, 43.8, 45.6, 46.1, 47., 43.7, 46.,
                                   46.7])

        nptest.assert_allclose(data['lat'][798:813], lats_should, atol=1e-5)
        nptest.assert_allclose(data['sm'][798:813], sm_should, atol=0.01)
        nptest.assert_allclose(data['sm_mean'][798:813], sm_mean_should,
                               atol=0.01)

    def test_h101_read(self):
        """
        Test read file.
        """
        dt = datetime(2017, 2, 20, 10, 42, 0)
        h101 = AscatNrtBufrFileList(
            os.path.join(self.root_path, 'h101'), 'h101')
        data = h101.read(dt)

        sm_should = np.array([26.1, 31.5, 49.5, 64.3, 80.3, 87.9, 24.5, 18.5,
                              20., 12.9, 6.3, 3.1, 3.5, 5.7, 2., 0., 4.6, 8.,
                              9.6, 11.8])

        lats_should = np.array([56.26292, 56.29267, 56.32112, 56.34827,
                                56.37413, 56.39868, 51.98611, 52.08869,
                                52.19038, 52.29117, 52.39104, 52.48999,
                                52.58802, 52.68512, 52.78127, 52.87648,
                                52.97073, 53.06402, 53.15634, 53.24768])

        sm_mean_should = np.array([22.8, 27., 31.8, 38.3, 44.3, 52.4, 24.8,
                                   27.8, 24.7, 23.9, 22.8, 25., 25.6, 26.1,
                                   25.9, 26.9, 28.5, 27.8, 24.3, 22.7])

        nptest.assert_allclose(data['lat'][1800:1820], lats_should, atol=1e-5)
        nptest.assert_allclose(data['sm'][1800:1820], sm_should, atol=0.01)
        nptest.assert_allclose(data['sm_mean'][1800:1820], sm_mean_should,
                               atol=0.01)

    def test_h102_read(self):
        """
        Test read file.
        """
        dt = datetime(2017, 2, 20, 10, 42, 0)
        h102 = AscatNrtBufrFileList(
            os.path.join(self.root_path, 'h102'), 'h102')
        data = h102.read(dt)

        sm_should = np.array([45.8, 43.5, 41.7, 42.7, 38.6, 31.1, 23.4, 21.7,
                              23.6, 26.8, 30.5, 30.1, 32.7, 34.7, 35.8, 38.3,
                              46.2, 53.7, 46.2])

        lats_should = np.array([43.16844, 43.21142, 43.25423, 43.29686,
                                43.33931, 43.38158, 43.42367, 43.46558,
                                43.50731, 43.54887, 43.59024, 43.63143,
                                43.67243, 43.71326, 43.7539, 43.79436,
                                43.83463, 43.87472, 43.91463])

        sm_mean_should = np.array([50.8, 44.5, 36.3, 29.9, 28.7, 29.1, 30.,
                                   30.1, 30.8, 33., 34.9, 35.7, 35.2, 34.2,
                                   32.9, 32.9, 34.5, 32.7, 30.6])

        nptest.assert_allclose(data['lat'][0:19], lats_should, atol=1e-5)
        nptest.assert_allclose(data['sm'][0:19], sm_should, atol=0.01)
        nptest.assert_allclose(data['sm_mean'][0:19], sm_mean_should,
                               atol=0.01)

    def test_h103_read(self):
        """
        Test read file.
        """
        dt = datetime(2017, 2, 20, 10, 30, 0)
        h103 = AscatNrtBufrFileList(
            os.path.join(self.root_path, 'h103'), 'h103')
        data = h103.read(dt)

        sm_should = np.array([20.4, 15.2, 4.2, 0., 0.7, 6.8, 14.2, 21.9,
                              23.7, 17.4, 14.8, 19.8, 15.4, 17.1, 28.9,
                              28.5, 23.9, 18.1, 25.4])

        lats_should = np.array([8.82896, 8.85636, 8.88373, 8.91107, 8.93837,
                                8.96564, 8.99288, 9.02009, 9.04726, 9.0744,
                                9.1015, 9.12858, 9.15561, 9.18262, 9.20959,
                                9.23653, 9.26343, 9.2903, 9.31713])

        sm_mean_should = np.array([14., 14., 13.8, 13.5, 13.3, 13.1, 12.8,
                                   12.6, 12.6, 12., 12.1, 12.6, 13.2, 14.1,
                                   13.9, 12.4, 10.9, 10.3, 10.7])

        nptest.assert_allclose(data['lat'][0:19], lats_should, atol=1e-5)
        nptest.assert_allclose(data['sm'][0:19], sm_should, atol=0.01)
        nptest.assert_allclose(data['sm_mean'][0:19], sm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H14(unittest.TestCase):

    def setUp(self):
        """
        Setup test data.
        """
        self.root_path = os.path.join(
            os.path.dirname(__file__),  'ascat_test_data', 'hsaf', 'h14')

    def test_read(self):
        """
        Test read file.
        """
        dt = datetime(2014, 5, 15, 0)
        h14 = H14GribFileList(self.root_path)
        data = h14.read(dt)

        assert sorted(data.keys()) == sorted(['lat', 'lon',
                                              'SM_layer1_0-7cm',
                                              'SM_layer2_7-28cm',
                                              'SM_layer3_28-100cm',
                                              'SM_layer4_100-289cm'])

        for var in data:
            assert data[var].shape == (800, 1600)


class Test_AscatSsmDataRecord(unittest.TestCase):

    def setUp(self):

        path = os.path.dirname(__file__)

        self.gpi = 3066159
        self.cdr_path = os.path.join(path, 'ascat_test_data', 'hsaf')
        self.grid_path = os.path.join(path, 'ascat_test_data', 'hsaf', 'grid')
        self.static_layer_path = os.path.join(path, 'ascat_test_data', 'hsaf',
                                              'static_layer')

    def test_read_h25(self):
        """
        Test read H25.
        """
        self.h25 = AscatSsmDataRecord(
            os.path.join(self.cdr_path, 'h25'), self.grid_path,
            static_layer_path=self.static_layer_path)

        data = self.h25.read(self.gpi, absolute_sm=True)
        assert data.attrs['gpi'] == self.gpi

        np.testing.assert_approx_equal(
            data.attrs['lon'], 19.03533, significant=4)
        np.testing.assert_approx_equal(
            data.attrs['lat'], 70.05438, significant=4)

        assert len(data) == 7737
        assert data.iloc[15].name.to_pydatetime() == datetime(
            2007, 1, 7, 10, 49, 9, 4)
        assert data.iloc[15]['sm'] == 22
        assert data.iloc[15]['ssf'] == 1
        assert data.iloc[15]['sm_noise'] == 6
        assert data.iloc[15]['frozen_prob'] == 0
        assert data.iloc[15]['snow_prob'] == 127
        assert data.iloc[15]['orbit_dir'].decode('utf-8') == 'D'
        assert data.iloc[15]['proc_flag'] == 0

        np.testing.assert_equal(
            data.iloc[15]['abs_sm_gldas'], np.nan)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_gldas'], np.nan)

        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_hwsd'], 0.1078, significant=6)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_hwsd'], 0.0294, significant=6)

        assert data.attrs['topo_complex'] == 9
        assert data.attrs['wetland_frac'] == 41

        np.testing.assert_approx_equal(
            data.attrs['porosity_gldas'], np.nan, significant=5)
        np.testing.assert_approx_equal(
            data.attrs['porosity_hwsd'], 0.49000001, significant=5)

    def test_read_h108(self):
        """
        Test read H108.
        """
        self.h108 = AscatSsmDataRecord(
            os.path.join(self.cdr_path, 'h108'), self.grid_path,
            static_layer_path=self.static_layer_path)

        data = self.h108.read(self.gpi, absolute_sm=True)
        assert data.attrs['gpi'] == self.gpi

        np.testing.assert_approx_equal(data.attrs['lon'], 19.03533,
                                       significant=4)
        np.testing.assert_approx_equal(data.attrs['lat'], 70.05438,
                                       significant=4)

        assert len(data) == 8222
        assert data.iloc[15].name.to_pydatetime() == datetime(
            2007, 1, 7, 10, 49, 9, 4)
        assert data.iloc[15]['sm'] == 22
        assert data.iloc[15]['ssf'] == 2
        assert data.iloc[15]['sm_noise'] == 6
        assert data.iloc[15]['frozen_prob'] == 0
        assert data.iloc[15]['snow_prob'] == 127
        assert data.iloc[15]['orbit_dir'].decode('utf-8') == 'D'
        assert data.iloc[15]['proc_flag'] == 0

        np.testing.assert_equal(
            data.iloc[15]['abs_sm_gldas'], np.nan)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_gldas'], np.nan)

        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_hwsd'], 0.1078, significant=6)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_hwsd'], 0.0294, significant=6)

        assert data.attrs['topo_complex'] == 9
        assert data.attrs['wetland_frac'] == 41

        np.testing.assert_equal(data.attrs['porosity_gldas'], np.nan)
        np.testing.assert_approx_equal(data.attrs['porosity_hwsd'],
                                       0.49000001, significant=5)

    def test_read_h109(self):
        """
        Test read H109.
        """
        self.h109 = AscatSsmDataRecord(
            os.path.join(self.cdr_path, 'h109'), self.grid_path,
            static_layer_path=self.static_layer_path)

        data = self.h109.read(self.gpi, absolute_sm=True)
        assert data.attrs['gpi'] == self.gpi

        np.testing.assert_approx_equal(data.attrs['lon'], 19.03533,
                                       significant=4)
        np.testing.assert_approx_equal(data.attrs['lat'], 70.05438,
                                       significant=4)

        assert len(data) == 11736
        assert data.iloc[15].name.to_pydatetime() == \
            datetime(2007, 1, 7, 10, 49, 9, 379200)
        assert data.iloc[15]['sm'] == 27
        assert data.iloc[15]['ssf'] == 1
        assert data.iloc[15]['sm_noise'] == 5
        assert data.iloc[15]['frozen_prob'] == 0
        assert data.iloc[15]['snow_prob'] == 127
        assert data.iloc[15]['dir'] == 1
        assert data.iloc[15]['proc_flag'] == 0
        assert data.iloc[15]['corr_flag'] == 16
        assert data.iloc[15]['sat_id'] == 3

        np.testing.assert_equal(
            data.iloc[15]['abs_sm_gldas'], np.nan)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_gldas'], np.nan)

        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_hwsd'], 0.1323, significant=6)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_hwsd'], 0.0245, significant=6)

        assert data.attrs['topo_complex'] == 9
        assert data.attrs['wetland_frac'] == 41

        np.testing.assert_equal(data.attrs['porosity_gldas'], np.nan)
        np.testing.assert_approx_equal(data.attrs['porosity_hwsd'],
                                       0.49000001, significant=5)

    def test_read_h110(self):
        """
        Test read H110.
        """
        self.h110 = AscatSsmDataRecord(
            os.path.join(self.cdr_path, 'h110'), self.grid_path,
            static_layer_path=self.static_layer_path)

        data = self.h110.read(self.gpi, absolute_sm=True)
        assert data.attrs['gpi'] == self.gpi

        np.testing.assert_approx_equal(data.attrs['lon'], 19.03533,
                                       significant=4)
        np.testing.assert_approx_equal(data.attrs['lat'], 70.05438,
                                       significant=4)

        assert len(data) == 1148
        assert data.iloc[15].name.to_pydatetime() == datetime(
            2016, 1, 3, 19, 34, 28, 99200)
        assert data.iloc[15]['sm'] == 48
        assert data.iloc[15]['ssf'] == 1
        assert data.iloc[15]['sm_noise'] == 5
        assert data.iloc[15]['frozen_prob'] == 0
        assert data.iloc[15]['snow_prob'] == 127
        assert data.iloc[15]['dir'] == 0
        assert data.iloc[15]['proc_flag'] == 0
        assert data.iloc[15]['corr_flag'] == 0
        assert data.iloc[15]['sat_id'] == 4

        np.testing.assert_equal(
            data.iloc[15]['abs_sm_gldas'], np.nan)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_gldas'], np.nan)

        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_hwsd'], 0.2352, significant=6)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_hwsd'], 0.0245, significant=6)

        assert data.attrs['topo_complex'] == 9
        assert data.attrs['wetland_frac'] == 41

        np.testing.assert_equal(data.attrs['porosity_gldas'], np.nan)
        np.testing.assert_approx_equal(data.attrs['porosity_hwsd'],
                                       0.49000001, significant=5)

    def test_read_h111(self):
        """
        Test read H111.
        """
        self.h111 = AscatSsmDataRecord(
            os.path.join(self.cdr_path, 'h111'), self.grid_path,
            static_layer_path=self.static_layer_path)

        data = self.h111.read(self.gpi, absolute_sm=True)
        assert data.attrs['gpi'] == self.gpi

        np.testing.assert_approx_equal(data.attrs['lon'], 19.03533,
                                       significant=4)
        np.testing.assert_approx_equal(data.attrs['lat'], 70.05438,
                                       significant=4)

        assert len(data) == 13715
        assert data.iloc[15].name.to_pydatetime() == datetime(
            2007, 1, 7, 10, 49, 9, 379200)
        assert data.iloc[15]['sm'] == 28
        assert data.iloc[15]['ssf'] == 1
        assert data.iloc[15]['sm_noise'] == 5
        assert data.iloc[15]['frozen_prob'] == 0
        assert data.iloc[15]['snow_prob'] == 127
        assert data.iloc[15]['dir'] == 1
        assert data.iloc[15]['proc_flag'] == 0
        assert data.iloc[15]['corr_flag'] == 4
        assert data.iloc[15]['sat_id'] == 3

        np.testing.assert_equal(data.iloc[15]['abs_sm_gldas'], np.nan)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_gldas'], np.nan)

        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_hwsd'], 0.1372, significant=6)
        np.testing.assert_approx_equal(
            data.iloc[15]['abs_sm_noise_hwsd'], 0.0245, significant=6)

        assert data.attrs['topo_complex'] == 9
        assert data.attrs['wetland_frac'] == 41

        np.testing.assert_equal(data.attrs['porosity_gldas'], np.nan)
        np.testing.assert_approx_equal(data.attrs['porosity_hwsd'],
                                       0.49000001, significant=5)

    def test_read_2points_cell_switch(self):
        """
        Test reading of two points in two different cells.
        This did not work in the past when the static layer class
        was closed too soon.
        """
        self.h111 = AscatSsmDataRecord(
            os.path.join(self.cdr_path, 'h111'), self.grid_path,
            static_layer_path=self.static_layer_path)

        gpi = 3066159
        data = self.h111.read(gpi, absolute_sm=True)
        assert data.attrs['gpi'] == gpi

        gpi = 2577735
        data = self.h111.read(gpi, absolute_sm=True)
        assert data.attrs['gpi'] == gpi


if __name__ == "__main__":
    unittest.main()
