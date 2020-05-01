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


import os
import unittest
from datetime import datetime

import numpy as np

import ascat


class TestAscatNcV55R12(unittest.TestCase):

    def setUp(self):

        path = os.path.dirname(__file__)

        cdr_path = os.path.join(path, 'ascat_test_data', 'tuw', 'ascat', 'ssm',
                                '55R12')

        grid_path = os.path.join(path, 'ascat_test_data', 'hsaf', 'grid')

        ioclass_kws = {'loc_dim_name': 'gp', 'loc_ids_name': 'gpi'}

        self.ascat_reader = ascat.AscatSsmCdr(cdr_path, grid_path,
                                              ioclass_kws=ioclass_kws)

    def test_read(self):

        gpi = 2329253
        result = self.ascat_reader.read(gpi)

        assert result.gpi == gpi
        np.testing.assert_approx_equal(result.longitude, 14.28413,
                                       significant=4)
        np.testing.assert_approx_equal(result.latitude, 45.698074,
                                       significant=4)

        assert len(result.data) == 2292
        assert result.data.iloc[15].name == datetime(2007, 1, 15, 19, 34, 41,
                                                     771032)
        assert result.data.iloc[15]['sm'] == 52
        assert result.data.iloc[15]['ssf'] == 1
        assert result.data.iloc[15]['sm_noise'] == 7
        assert result.data.iloc[15]['orbit_dir'].decode('utf-8') == 'A'
        assert result.data.iloc[15]['proc_flag'] == 0

    def test_neighbor_search(self):

        gpi, distance = self.ascat_reader.grid.find_nearest_gpi(3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


class TestAscatNcV55R21(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)

        cdr_path = os.path.join(path, 'ascat_test_data', 'tuw', 'ascat', 'ssm',
                                '55R21')

        grid_path = os.path.join(path, 'ascat_test_data', 'hsaf', 'grid')

        static_layer_path = os.path.join(path, 'ascat_test_data', 'hsaf',
                                         'static_layer')

        self.ascat_reader = ascat.AscatSsmCdr(
            cdr_path, grid_path, static_layer_path=static_layer_path)

    def tearDown(self):
        self.ascat_reader.close()

    def test_read(self):

        gpi = 2329253
        result = self.ascat_reader.read(gpi, absolute_sm=True)
        assert result.gpi == gpi
        np.testing.assert_approx_equal(
            result.longitude, 14.28413, significant=4)
        np.testing.assert_approx_equal(
            result.latitude, 45.698074, significant=4)

        assert len(result.data) == 2457
        assert result.data.iloc[15].name == datetime(
            2007, 1, 15, 19, 34, 41, 5)
        assert result.data.iloc[15]['sm'] == 55
        assert result.data.iloc[15]['ssf'] == 1
        assert result.data.iloc[15]['sm_noise'] == 7
        assert result.data.iloc[15]['frozen_prob'] == 29
        assert result.data.iloc[15]['snow_prob'] == 0
        assert result.data.iloc[15]['orbit_dir'].decode('utf-8') == 'A'
        assert result.data.iloc[15]['proc_flag'] == 0

        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_gldas'], 0.2969999, significant=6)

        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_noise_gldas'], 0.03779999,
            significant=6)

        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_hwsd'], 0.2364999, significant=6)
        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_noise_hwsd'], 0.030100, significant=6)

        assert result.topo_complex == 14
        assert result.wetland_frac == 0

        np.testing.assert_approx_equal(
            result.porosity_gldas, 0.539999, significant=5)
        np.testing.assert_approx_equal(
            result.porosity_hwsd, 0.4299994, significant=5)

    def test_neighbor_search(self):

        gpi, distance = self.ascat_reader.grid.find_nearest_gpi(3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


class TestAscatNcV55R22(unittest.TestCase):

    def setUp(self):

        path = os.path.dirname(__file__)

        cdr_path = os.path.join(path, 'ascat_test_data', 'tuw', 'ascat', 'ssm',
                                '55R22')

        grid_path = os.path.join(path, 'ascat_test_data', 'hsaf', 'grid')

        static_layer_path = os.path.join(path, 'ascat_test_data', 'hsaf',
                                         'static_layer')

        self.ascat_reader = ascat.AscatSsmCdr(
            cdr_path, grid_path, static_layer_path=static_layer_path)

    def tearDown(self):
        self.ascat_reader.close()

    def test_read(self):

        gpi = 2329253
        result = self.ascat_reader.read(gpi, absolute_sm=True)
        assert result.gpi == gpi
        np.testing.assert_approx_equal(result.longitude, 14.28413,
                                       significant=4)
        np.testing.assert_approx_equal(result.latitude, 45.698074,
                                       significant=4)

        ref_list = ['orbit_dir', 'proc_flag', 'sm', 'sm_noise', 'ssf',
                    'snow_prob', 'frozen_prob', 'abs_sm_gldas',
                    'abs_sm_noise_gldas', 'abs_sm_hwsd', 'abs_sm_noise_hwsd']

        assert sorted(list(result.data.columns)) == sorted(ref_list)

        assert len(result.data) == 2642
        assert result.data.iloc[15].name == datetime(
            2007, 1, 15, 19, 34, 41, 5)
        assert result.data.iloc[15]['sm'] == 55
        assert result.data.iloc[15]['ssf'] == 1
        assert result.data.iloc[15]['sm_noise'] == 7
        assert result.data.iloc[15]['frozen_prob'] == 29
        assert result.data.iloc[15]['snow_prob'] == 0
        assert result.data.iloc[15]['orbit_dir'].decode('utf-8') == 'A'
        assert result.data.iloc[15]['proc_flag'] == 0

        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_gldas'], 0.2969999, significant=6)
        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_noise_gldas'], 0.03779999,
            significant=6)

        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_hwsd'], 0.2364999, significant=6)
        np.testing.assert_approx_equal(
            result.data.iloc[15]['abs_sm_noise_hwsd'], 0.0301000, significant=6)

        assert result.topo_complex == 14
        assert result.wetland_frac == 0

        np.testing.assert_approx_equal(
            result.porosity_gldas, 0.539999, significant=5)
        np.testing.assert_approx_equal(
            result.porosity_hwsd, 0.4299994, significant=5)

    def test_neighbor_search(self):

        gpi, distance = self.ascat_reader.grid.find_nearest_gpi(3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


if __name__ == '__main__':
    unittest.main()
