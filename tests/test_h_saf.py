'''
Created on May 21, 2014

@author: Christoph Paulik
'''
import unittest
import datetime
import numpy as np
import numpy.testing as nptest
import os
import sys
import pytest

import ascat.h_saf as H_SAF


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H08(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__), 'test-data', 'sat', 'h_saf', 'h08')
        self.reader = H_SAF.H08img(data_path)

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
        # do not check data content at the moment just shapes and structure
        assert sorted(data.keys()) == sorted(
            ['ssm', 'corr_flag', 'ssm_noise', 'proc_flag'])
        assert lons.shape == (3120, 7680)
        assert lats.shape == (3120, 7680)
        for var in data:
            assert data[var].shape == (3120, 7680)

    def test_image_reading_bbox_empty(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(datetime.datetime(2010, 5, 1, 8, 33, 1),
                                                                       lat_lon_bbox=[45, 48, 15, 18])
        # do not check data content at the moment just shapes and structure
        assert data is None
        assert lons is None
        assert lats is None

    def test_image_reading_bbox(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(datetime.datetime(2010, 5, 1, 8, 33, 1),
                                                                       lat_lon_bbox=[60, 70, 15, 25])
        # do not check data content at the moment just shapes and structure
        assert sorted(data.keys()) == sorted(
            ['ssm', 'corr_flag', 'ssm_noise', 'proc_flag'])
        assert lons.shape == (2400, 2400)
        assert lats.shape == (2400, 2400)
        for var in data:
            assert data[var].shape == (2400, 2400)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H07(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h07')
        self.reader = H_SAF.H07img(data_path)

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
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               15:-1:20], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][15:-1:20],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H16(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h16')
        self.reader = H_SAF.H16img(data_path, month_path_str='')

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(
            2017, 2, 20, 11, 0, 0) + datetime.timedelta(minutes=3) * n for n in range(8)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_offset_getting_datetime_boundary(self):
        """
        test getting the image offsets for a known daterange,
        checks if exact datetimes are used
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20, 11, 3), datetime.datetime(2017, 2, 20, 11, 12))
        timestamps_should = [datetime.datetime(
            2017, 2, 20, 11, 3, 0) + datetime.timedelta(minutes=3) * n for n in range(4)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 11, 15, 0))

        ssm_should = np.array([0., 3.6, 7.8, 8.2, 12.3, 14.7, 21.6, 26.7,
                               30.6, 32.2, 43., 50.5, 46.3, 47.6, 58.])

        lats_should = np.array([-28.25222, -28.21579, -28.1789, -28.14155,
                                -28.10374, -28.06547, -28.02674, -27.98755, -27.94791, -27.90782,
                                -27.86727, -27.82627, -27.78482, -27.74292, -27.70058, ])

        ssm_mean_should = np.array([32.8, 35.1, 37.8, 39.2, 39., 38., 36.2,
                                    39.6, 43.8, 45.6, 46.1, 47., 43.7, 46., 46.7])

        nptest.assert_allclose(lats[253:268], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               253:268], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][253:268],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H101(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h101')
        self.reader = H_SAF.H101img(data_path, month_path_str='')

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(
            2017, 2, 20, 10, 24, 0) + datetime.timedelta(minutes=3) * n for n in range(8)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 10, 42, 0))

        ssm_should = np.array([26.1, 31.5, 49.5, 64.3, 80.3, 87.9, 24.5, 18.5,
                               20., 12.9, 6.3, 3.1, 3.5, 5.7, 2., 0., 4.6, 8., 9.6, 11.8])

        lats_should = np.array([56.26292, 56.29267, 56.32112, 56.34827,
                                56.37413, 56.39868, 51.98611, 52.08869, 52.19038, 52.29117, 52.39104,
                                52.48999, 52.58802, 52.68512, 52.78127, 52.87648, 52.97073, 53.06402,
                                53.15634, 53.24768])

        ssm_mean_should = np.array([22.8, 27., 31.8, 38.3, 44.3, 52.4, 24.8,
                                    27.8, 24.7, 23.9, 22.8, 25., 25.6, 26.1, 25.9, 26.9, 28.5, 27.8, 24.3,
                                    22.7])

        nptest.assert_allclose(lats[577:597], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               577:597], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][577:597],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H102(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h102')
        self.reader = H_SAF.H102img(data_path, month_path_str='')

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(
            2017, 2, 20, 10, 24, 0) + datetime.timedelta(minutes=3) * n for n in range(8)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 10, 42, 0))

        ssm_should = np.array([45.8, 43.5, 41.7, 42.7, 38.6, 31.1, 23.4, 21.7,
                               23.6, 26.8, 30.5, 30.1, 32.7, 34.7, 35.8, 38.3, 46.2, 53.7, 46.2])

        lats_should = np.array([43.16844, 43.21142, 43.25423, 43.29686,
                                43.33931, 43.38158, 43.42367, 43.46558, 43.50731, 43.54887, 43.59024,
                                43.63143, 43.67243, 43.71326, 43.7539, 43.79436, 43.83463, 43.87472,
                                43.91463])

        ssm_mean_should = np.array([50.8, 44.5, 36.3, 29.9, 28.7, 29.1, 30.,
                                    30.1, 30.8, 33., 34.9, 35.7, 35.2, 34.2, 32.9, 32.9, 34.5, 32.7,
                                    30.6])

        nptest.assert_allclose(lats[0:19], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               0:19], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][0:19],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H103(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h103')
        self.reader = H_SAF.H103img(data_path, month_path_str='')

    def tearDown(self):
        self.reader = None

    def test_offset_getting(self):
        """
        test getting the image offsets for a known day
        2010-05-01
        """
        timestamps = self.reader.tstamps_for_daterange(
            datetime.datetime(2017, 2, 20), datetime.datetime(2017, 2, 21))
        timestamps_should = [datetime.datetime(
            2017, 2, 20, 10, 30, 0) + datetime.timedelta(minutes=3) * n for n in range(8)]
        assert sorted(timestamps) == sorted(timestamps_should)

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2017, 2, 20, 10, 30, 0))

        ssm_should = np.array([20.4, 15.2, 4.2, 0., 0.7, 6.8, 14.2, 21.9,
                               23.7, 17.4, 14.8, 19.8, 15.4, 17.1, 28.9, 28.5, 23.9, 18.1, 25.4])

        lats_should = np.array([8.82896, 8.85636, 8.88373, 8.91107, 8.93837,
                                8.96564, 8.99288, 9.02009, 9.04726, 9.0744, 9.1015, 9.12858, 9.15561,
                                9.18262, 9.20959, 9.23653, 9.26343, 9.2903, 9.31713])

        ssm_mean_should = np.array([14., 14., 13.8, 13.5, 13.3, 13.1, 12.8,
                                    12.6, 12.6, 12., 12.1, 12.6, 13.2, 14.1, 13.9, 12.4, 10.9, 10.3,
                                    10.7])

        nptest.assert_allclose(lats[0:19], lats_should, atol=1e-5)
        nptest.assert_allclose(data['Surface Soil Moisture (Ms)'][
                               0:19], ssm_should, atol=0.01)
        nptest.assert_allclose(data['Mean Surface Soil Moisture'][0:19],
                               ssm_mean_should,
                               atol=0.01)


@pytest.mark.skipif(sys.platform == 'win32', reason="Does not work on Windows")
class Test_H14(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(__file__),  'test-data', 'sat', 'h_saf', 'h14')
        self.reader = H_SAF.H14img(data_path, expand_grid=False)
        self.expand_reader = H_SAF.H14img(data_path, expand_grid=True)

    def tearDown(self):
        self.reader = None
        self.expand_reader = None

    def test_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.reader.read(
            datetime.datetime(2014, 5, 15))
        assert sorted(data.keys()) == sorted(['SM_layer1_0-7cm', 'SM_layer2_7-28cm',
                                              'SM_layer3_28-100cm', 'SM_layer4_100-289cm'])
        assert lons.shape == (843490,)
        assert lats.shape == (843490,)
        for var in data:
            assert data[var].shape == (843490,)
            assert meta[var]['name']
            assert meta[var]['units'] in ['dimensionless', 'unknown']

    def test_expanded_image_reading(self):
        data, meta, timestamp, lons, lats, time_var = self.expand_reader.read(
            datetime.datetime(2014, 5, 15))
        assert sorted(data.keys()) == sorted(['SM_layer1_0-7cm', 'SM_layer2_7-28cm',
                                              'SM_layer3_28-100cm', 'SM_layer4_100-289cm'])
        assert lons.shape == (800, 1600)
        assert lats.shape == (800, 1600)
        for var in data:
            assert data[var].shape == (800, 1600)
            assert meta[var]['name']
            assert meta[var]['units'] in ['dimensionless', 'unknown']

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
