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
            datetime.datetime(2010, 5, 1), datetime.datetime(2010, 5, 1))
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
