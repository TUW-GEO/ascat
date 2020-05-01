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

from ascat.tuw import AscatVodTs, Ascat_SSM


class TestAscat(unittest.TestCase):

    def setUp(self):
        self.ascat_folder = os.path.join(os.path.dirname(__file__),
                                         'ascat_test_data', 'tuw', 'ascat', 'ssm')

        self.ascat_adv_folder = os.path.join(os.path.dirname(__file__),
                                             'ascat_test_data', 'tuw',
                                             'advisory_flags')

        self.ascat_grid_folder = os.path.join(os.path.dirname(__file__),
                                              'ascat_test_data', 'tuw',
                                              'grid')

        # init the ASCAT_SSM reader with the paths
        self.ascat_SSM_reader = Ascat_SSM(
            self.ascat_folder, self.ascat_grid_folder,
            advisory_flags_path=self.ascat_adv_folder)

    def test_read_ssm(self):

        gpi = 2329253
        result = self.ascat_SSM_reader.read_ssm(gpi)
        assert result.gpi == gpi
        assert result.longitude == 14.28413
        assert result.latitude == 45.698074
        assert sorted(list(result.data.columns)) == sorted([
            'ERR', 'SSF', 'SSM', 'frozen_prob', 'snow_prob'])
        assert len(result.data) == 2058
        assert result.data.iloc[15].name == datetime(2007, 1, 15, 19)
        assert result.data.iloc[15]['ERR'] == 7
        assert result.data.iloc[15]['SSF'] == 1
        assert result.data.iloc[15]['SSM'] == 53
        assert result.data.iloc[15]['frozen_prob'] == 29
        assert result.data.iloc[15]['snow_prob'] == 0

    def test_neighbor_search(self):

        self.ascat_SSM_reader._load_grid_info()
        gpi, distance = self.ascat_SSM_reader.grid.find_nearest_gpi(
            3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


class TestAscatVodTs(unittest.TestCase):

    def setUp(self):
        self.ascat_folder = os.path.join(os.path.dirname(__file__),
                                         'ascat_test_data', 'tuw', 'ascat', 'vod')

        self.ascat_grid_folder = os.path.join(os.path.dirname(__file__),
                                              'ascat_test_data', 'hsaf', 'grid')

        # init the ASCAT_SSM reader with the paths
        self.ascat_VOD_reader = AscatVodTs(self.ascat_folder,
                                           self.ascat_grid_folder)

    def test_read_vod(self):
        gpi = 2199945
        data = self.ascat_VOD_reader.read(gpi)
        lon, lat = self.ascat_VOD_reader.grid.gpi2lonlat(gpi)
        np.testing.assert_approx_equal(lon, 9.1312, significant=4)
        np.testing.assert_approx_equal(lat, 42.5481, significant=4)

        assert list(data.columns) == ['vod']
        assert len(data) == 4018
        assert data.iloc[15].name == datetime(2007, 1, 16, 12, 0, 0)
        assert data.iloc[15]['vod'] == np.float32(0.62470651)

    def test_neighbor_search(self):
        gpi, distance = self.ascat_VOD_reader.grid.find_nearest_gpi(
            3.25, 46.13)
        assert gpi == 2346869
        np.testing.assert_approx_equal(distance, 2267.42, significant=2)


if __name__ == '__main__':
    unittest.main()
