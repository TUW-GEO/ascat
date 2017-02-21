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
Test data downloaded from EUMETSAT UMARF
'''

import unittest
import datetime
import numpy as np
import numpy.testing as nptest
import os
import sys
import pytest

from ascat.eumetsat import AscatAL2Ssm125, AscatBL2Ssm125
from ascat.eumetsat import AscatAL2Ssm125PDU, AscatBL2Ssm125PDU
from ascat.eumetsat import AscatAL2Ssm250, AscatBL2Ssm250
from ascat.eumetsat import AscatAL2Ssm250PDU, AscatBL2Ssm250PDU
from ascat.eumetsat import AscatAL2Ssm125Nc, AscatBL2Ssm125Nc
from ascat.eumetsat import AscatAL2Ssm250Nc, AscatBL2Ssm250Nc


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_A_L2_SSM_125_BUFR(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'bufr')
        self.reader = AscatAL2Ssm125(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 4, 15),
                             datetime.datetime(2017, 2, 20, 5, 57)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 4, 15))

        ssm_should = np.array([3., 0., 0., 0., 0., 0., 0., 0., 0., 1.8, 3.3,
                               4.8, 4.3, 2.5, 0., 3.8, 5.8, 1.5, 2.4, 4.1, 2.3,
                               2.7, 5.6, 5.5, 4.9])

        lats_should = np.array([62.60224, 62.67133, 62.74015, 62.80871, 62.877,
                                62.94502, 63.01276, 63.08024, 63.14743,
                                63.21435, 63.28098, 63.34734, 63.41341,
                                63.47919, 63.54468, 63.60988, 63.67479,
                                63.7394, 63.80372, 63.86773, 63.93144,
                                63.99485, 64.05795, 64.12075, 64.18323])

        ssm_mean_should = np.array([21.3, 21.3, 21.4, 22.4, 23.4, 24.5, 26.,
                                    27.1, 27., 26.6, 27.1, 27.6, 27.4, 26.7,
                                    26.5, 27.5, 28.2, 28.4, 28.8, 29.2, 30.,
                                    31., 31.3, 31.9, 32.1])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_B_L2_SSM_125_BUFR(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'bufr')
        self.reader = AscatBL2Ssm125(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 5, 9)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 5, 9))

        ssm_should = np.array([29.2, 30.2, 35.7, 38.6, 37.5, 37.6, 40.5, 44.5,
                               40.7, 39.7, 41.5, 38.8, 34.5, 36.8, 39.4, 41.2,
                               42.4, 42.9, 39.3, 30.5, 26.7, 26.5, 26.7, 23.9,
                               26.2])

        lats_should = np.array([64.74398, 64.81854, 64.89284, 64.96688,
                                65.04066, 65.11416, 65.18739, 65.26036,
                                65.33304, 65.40545, 65.47758, 65.54942,
                                65.62099, 65.69226, 65.76324, 65.83393,
                                65.90432, 65.97442, 66.04422, 66.11371,
                                66.1829, 66.25177, 66.32034, 66.38859,
                                66.45653])

        ssm_mean_should = np.array([36.7, 35.4, 33.4, 32.5, 32.5, 32., 31.2,
                                    29.4, 28.7, 27.6, 25.8, 25.4, 25.5, 25.3,
                                    24.4, 23.4, 22.3, 21.3, 20.4, 20.4, 19.9,
                                    19.7, 20.3, 21.5, 22.9])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_A_L2_SSM_125_BUFR_PDU(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'PDU')
        self.reader = AscatAL2Ssm125PDU(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 4, 15),
                             datetime.datetime(2017, 2, 20, 4, 18),
                             datetime.datetime(2017, 2, 20, 4, 21)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 4, 15))

        ssm_should = np.array([3., 0., 0., 0., 0., 0., 0., 0., 0., 1.8, 3.3,
                               4.8, 4.3, 2.5, 0., 3.8, 5.8, 1.5, 2.4, 4.1, 2.3,
                               2.7, 5.6, 5.5, 4.9])

        lats_should = np.array([62.60224, 62.67133, 62.74015, 62.80871, 62.877,
                                62.94502, 63.01276, 63.08024, 63.14743,
                                63.21435, 63.28098, 63.34734, 63.41341,
                                63.47919, 63.54468, 63.60988, 63.67479,
                                63.7394, 63.80372, 63.86773, 63.93144,
                                63.99485, 64.05795, 64.12075, 64.18323])

        ssm_mean_should = np.array([21.3, 21.3, 21.4, 22.4, 23.4, 24.5, 26.,
                                    27.1, 27., 26.6, 27.1, 27.6, 27.4, 26.7,
                                    26.5, 27.5, 28.2, 28.4, 28.8, 29.2, 30.,
                                    31., 31.3, 31.9, 32.1])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_B_L2_SSM_125_BUFR_PDU(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'PDU')
        self.reader = AscatBL2Ssm125PDU(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 5, 9),
                             datetime.datetime(2017, 2, 20, 5, 12),
                             datetime.datetime(2017, 2, 20, 5, 15)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 5, 9))

        ssm_should = np.array([29.2, 30.2, 35.7, 38.6, 37.5, 37.6, 40.5, 44.5,
                               40.7, 39.7, 41.5, 38.8, 34.5, 36.8, 39.4, 41.2,
                               42.4, 42.9, 39.3, 30.5, 26.7, 26.5, 26.7, 23.9,
                               26.2])

        lats_should = np.array([64.74398, 64.81854, 64.89284, 64.96688,
                                65.04066, 65.11416, 65.18739, 65.26036,
                                65.33304, 65.40545, 65.47758, 65.54942,
                                65.62099, 65.69226, 65.76324, 65.83393,
                                65.90432, 65.97442, 66.04422, 66.11371,
                                66.1829, 66.25177, 66.32034, 66.38859,
                                66.45653])

        ssm_mean_should = np.array([36.7, 35.4, 33.4, 32.5, 32.5, 32., 31.2,
                                    29.4, 28.7, 27.6, 25.8, 25.4, 25.5, 25.3,
                                    24.4, 23.4, 22.3, 21.3, 20.4, 20.4, 19.9,
                                    19.7, 20.3, 21.5, 22.9])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_A_L2_SSM_250_BUFR(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_250', 'bufr')
        self.reader = AscatAL2Ssm250(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 4, 15),
                             datetime.datetime(2017, 2, 20, 5, 57)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 4, 15))

        ssm_should = np.array([1.8, 0., 0., 0., 0., 4.6, 2.8, 4., 4.1, 4.2,
                               4.7, 5.4, 7.1, 7.1, 8.2, 9.2, 14.5, 15.4, 14.3,
                               17.7, 25.5, 36.9, 37.8, 39.4, 24.1])

        lats_should = np.array([62.60224, 62.74015, 62.877, 63.01276,
                                63.14743, 63.28098, 63.41341, 63.54468,
                                63.67479, 63.80372, 63.93144, 64.05795,
                                64.18323, 64.30725, 64.42999, 64.55145,
                                64.6716, 64.79042, 64.9079, 65.02401,
                                65.13873, 67.85438, 67.91597, 67.97556,
                                68.03314])

        ssm_mean_should = np.array([21.3, 21.4, 23.4, 26., 27., 27.1, 27.4,
                                    26.5, 28.2, 28.8, 30., 31.3, 32.1, 30.6,
                                    27.8, 28.9, 29.5, 32.1, 33.8, 32.9, 28.9,
                                    41.1, 40.8, 34.4, 31.])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_B_L2_SSM_250_BUFR(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_250', 'bufr')
        self.reader = AscatBL2Ssm250(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 5, 9)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 5, 9))

        ssm_should = np.array([28.8, 31., 35.8, 38.7, 39.3, 38.9, 39.6, 40.7,
                               40.9, 35.5, 28.7, 25.2, 25.8, 27.2, 26.3, 29.1,
                               30., 27.1, 25.5, 23.9, 25.7, 44.9, 38.7, 36.7,
                               40.6])

        lats_should = np.array([64.74398, 64.89284, 65.04066, 65.18739,
                                65.33304, 65.47758, 65.62099, 65.76324,
                                65.90432, 66.04422, 66.1829, 66.32034,
                                66.45653, 66.59144, 66.72505, 66.85734,
                                66.98829, 67.11787, 67.24605, 67.37283,
                                67.49816, 70.48423, 70.55154, 70.61658,
                                70.67934])

        ssm_mean_should = np.array([36.7, 33.4, 32.5, 31.2, 28.7, 25.8, 25.5,
                                    24.4, 22.3, 20.4, 19.9, 20.3, 22.9, 23.7,
                                    23.5, 22.2, 22.2, 22.4, 25.3, 27.8, 27.7,
                                    30.7, 30.7, 31.6, 33.6])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_A_L2_SSM_250_BUFR_PDU(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_250', 'PDU')
        self.reader = AscatAL2Ssm250PDU(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 4, 15),
                             datetime.datetime(2017, 2, 20, 4, 18),
                             datetime.datetime(2017, 2, 20, 4, 21)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 4, 15))

        ssm_should = np.array([1.8, 0., 0., 0., 0., 4.6, 2.8, 4., 4.1, 4.2,
                               4.7, 5.4, 7.1, 7.1, 8.2, 9.2, 14.5, 15.4, 14.3,
                               17.7, 25.5, 36.9, 37.8, 39.4, 24.1])

        lats_should = np.array([62.60224, 62.74015, 62.877, 63.01276,
                                63.14743, 63.28098, 63.41341, 63.54468,
                                63.67479, 63.80372, 63.93144, 64.05795,
                                64.18323, 64.30725, 64.42999, 64.55145,
                                64.6716, 64.79042, 64.9079, 65.02401,
                                65.13873, 67.85438, 67.91597, 67.97556,
                                68.03314])

        ssm_mean_should = np.array([21.3, 21.4, 23.4, 26., 27., 27.1, 27.4,
                                    26.5, 28.2, 28.8, 30., 31.3, 32.1, 30.6,
                                    27.8, 28.9, 29.5, 32.1, 33.8, 32.9, 28.9,
                                    41.1, 40.8, 34.4, 31.])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_ASCAT_B_L2_SSM_250_BUFR_PDU(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_250', 'PDU')
        self.reader = AscatBL2Ssm250PDU(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 5, 9),
                             datetime.datetime(2017, 2, 20, 5, 12),
                             datetime.datetime(2017, 2, 20, 5, 15)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 5, 9))

        ssm_should = np.array([28.8, 31., 35.8, 38.7, 39.3, 38.9, 39.6, 40.7,
                               40.9, 35.5, 28.7, 25.2, 25.8, 27.2, 26.3, 29.1,
                               30., 27.1, 25.5, 23.9, 25.7, 44.9, 38.7, 36.7,
                               40.6])

        lats_should = np.array([64.74398, 64.89284, 65.04066, 65.18739,
                                65.33304, 65.47758, 65.62099, 65.76324,
                                65.90432, 66.04422, 66.1829, 66.32034,
                                66.45653, 66.59144, 66.72505, 66.85734,
                                66.98829, 67.11787, 67.24605, 67.37283,
                                67.49816, 70.48423, 70.55154, 70.61658,
                                70.67934])

        ssm_mean_should = np.array([36.7, 33.4, 32.5, 31.2, 28.7, 25.8, 25.5,
                                    24.4, 22.3, 20.4, 19.9, 20.3, 22.9, 23.7,
                                    23.5, 22.2, 22.2, 22.4, 25.3, 27.8, 27.7,
                                    30.7, 30.7, 31.6, 33.6])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               :25], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][:25],
                               ssm_mean_should,
                               atol=0.01)


class Test_ASCAT_A_L2_SSM_125_NC(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'nc')
        self.reader = AscatAL2Ssm125Nc(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 4, 15),
                             datetime.datetime(2017, 2, 20, 5, 57)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 4, 15))

        ssm_should = np.array([3., 0., 0., 0., 0., 0., 0., 0., 0., 1.8, 3.3,
                               4.8, 4.3, 2.5, 0., 3.8, 5.8, 1.5, 2.4, 4.1, 2.3,
                               2.7, 5.6, 5.5, 4.9])

        lats_should = np.array([62.60224, 62.67133, 62.74015, 62.80871, 62.877,
                                62.94502, 63.01276, 63.08024, 63.14743,
                                63.21435, 63.28098, 63.34734, 63.41341,
                                63.47919, 63.54468, 63.60988, 63.67479,
                                63.7394, 63.80372, 63.86773, 63.93144,
                                63.99485, 64.05795, 64.12075, 64.18323])

        ssm_mean_should = np.array([21.3, 21.3, 21.4, 22.4, 23.4, 24.5, 26.,
                                    27.1, 27., 26.6, 27.1, 27.6, 27.4, 26.7,
                                    26.5, 27.5, 28.2, 28.4, 28.8, 29.2, 30.,
                                    31., 31.3, 31.9, 32.1])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][
                               :25], ssm_should, atol=0.1)
        nptest.assert_allclose(data['mean_soil_moisture'][:25],
                               ssm_mean_should,
                               atol=0.1)


class Test_ASCAT_B_L2_SSM_125_NC(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_125', 'nc')
        self.reader = AscatBL2Ssm125Nc(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 5, 9)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 5, 9))

        ssm_should = np.array([29.2, 30.2, 35.7, 38.6, 37.5, 37.6, 40.5, 44.5,
                               40.7, 39.7, 41.5, 38.8, 34.5, 36.8, 39.4, 41.2,
                               42.4, 42.9, 39.3, 30.5, 26.7, 26.5, 26.7, 23.9,
                               26.2])

        lats_should = np.array([64.74398, 64.81854, 64.89284, 64.96688,
                                65.04066, 65.11416, 65.18739, 65.26036,
                                65.33304, 65.40545, 65.47758, 65.54942,
                                65.62099, 65.69226, 65.76324, 65.83393,
                                65.90432, 65.97442, 66.04422, 66.11371,
                                66.1829, 66.25177, 66.32034, 66.38859,
                                66.45653])

        ssm_mean_should = np.array([36.7, 35.4, 33.4, 32.5, 32.5, 32., 31.2,
                                    29.4, 28.7, 27.6, 25.8, 25.4, 25.5, 25.3,
                                    24.4, 23.4, 22.3, 21.3, 20.4, 20.4, 19.9,
                                    19.7, 20.3, 21.5, 22.9])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][
                               :25], ssm_should, atol=0.1)
        nptest.assert_allclose(data['mean_soil_moisture'][:25],
                               ssm_mean_should,
                               atol=0.1)


class Test_ASCAT_A_L2_SSM_250_NC(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_250', 'nc')
        self.reader = AscatAL2Ssm250Nc(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 4, 15),
                             datetime.datetime(2017, 2, 20, 5, 57)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 4, 15))

        ssm_should = np.array([1.8, 0., 0., 0., 0., 4.6, 2.8, 4., 4.1, 4.2,
                               4.7, 5.4, 7.1, 7.1, 8.2, 9.2, 14.5, 15.4, 14.3,
                               17.7, 25.5, 36.9, 37.8, 39.4, 24.1])

        lats_should = np.array([62.60224, 62.74015, 62.877, 63.01276,
                                63.14743, 63.28098, 63.41341, 63.54468,
                                63.67479, 63.80372, 63.93144, 64.05795,
                                64.18323, 64.30725, 64.42999, 64.55145,
                                64.6716, 64.79042, 64.9079, 65.02401,
                                65.13873, 67.85438, 67.91597, 67.97556,
                                68.03314])

        ssm_mean_should = np.array([21.3, 21.4, 23.4, 26., 27., 27.1, 27.4,
                                    26.5, 28.2, 28.8, 30., 31.3, 32.1, 30.6,
                                    27.8, 28.9, 29.5, 32.1, 33.8, 32.9, 28.9,
                                    41.1, 40.8, 34.4, 31.])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][
                               :25], ssm_should, atol=0.1)
        nptest.assert_allclose(data['mean_soil_moisture'][:25],
                               ssm_mean_should,
                               atol=0.1)


class Test_ASCAT_B_L2_SSM_250_NC(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L2_SM_250', 'nc')
        self.reader = AscatBL2Ssm250Nc(data_path)

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(2017, 2, 20, 5, 9)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 5, 9))

        ssm_should = np.array([28.8, 31., 35.8, 38.7, 39.3, 38.9, 39.6, 40.7,
                               40.9, 35.5, 28.7, 25.2, 25.8, 27.2, 26.3, 29.1,
                               30., 27.1, 25.5, 23.9, 25.7, 44.9, 38.7, 36.7,
                               40.6])

        lats_should = np.array([64.74398, 64.89284, 65.04066, 65.18739,
                                65.33304, 65.47758, 65.62099, 65.76324,
                                65.90432, 66.04422, 66.1829, 66.32034,
                                66.45653, 66.59144, 66.72505, 66.85734,
                                66.98829, 67.11787, 67.24605, 67.37283,
                                67.49816, 70.48423, 70.55154, 70.61658,
                                70.67934])

        ssm_mean_should = np.array([36.7, 33.4, 32.5, 31.2, 28.7, 25.8, 25.5,
                                    24.4, 22.3, 20.4, 19.9, 20.3, 22.9, 23.7,
                                    23.5, 22.2, 22.2, 22.4, 25.3, 27.8, 27.7,
                                    30.7, 30.7, 31.6, 33.6])

        nptest.assert_allclose(lats[:25], lats_should, atol=1e-5)
        nptest.assert_allclose(data['soil_moisture'][
                               :25], ssm_should, atol=0.1)
        nptest.assert_allclose(data['mean_soil_moisture'][:25],
                               ssm_mean_should,
                               atol=0.1)
