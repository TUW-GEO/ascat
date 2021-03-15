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
Tests for download.
"""

import unittest
import configparser
from tempfile import mkdtemp
from datetime import datetime, timedelta

from ascat.download.connectors import HSAFConnector, EumetsatConnector


class TestDownload(unittest.TestCase):

    def setUp(self):

        self.credential_file = 'test_credentials.ini'
        yesterday = (datetime.now() - timedelta(1)).strftime('%Y%m%d')
        today = datetime.today().strftime('%Y%m%d')
        self.start_date = yesterday
        self.end_date = today
        self.path = mkdtemp()

    def tearDown(self):

        self.credential_file = None
        self.start_date = None
        self.end_date = None

    def test_creds(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        assert len(credentials.sections()) > 0

    def test_eumetsat_connect(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = EumetsatConnector()
        connector.connect(credentials=credentials)

    def test_eumetsat_download(self):

        product = "EO:EUM:DAT:METOP:SOMO12"
        coords = [[-1.0, -1.0], [4.0, -4.0], [8.0, -2.0],
                  [9.0, 2.0], [6.0, 4.0], [1.0, 5.0], [-1.0, -1.0]]

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = EumetsatConnector()
        connector.connect(credentials=credentials)

        connector.download(product, self.path, coords, self.start_date,
                           self.end_date, file_limit=1)

    def test_hsaf_connect(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)

        connector.close()

    def test_hsaf_h08_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h08', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h10_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h10', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h16_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h16', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h101_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h101', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h102_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h102', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h103_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h103', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h104_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h104', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()

    def test_hsaf_h105_download(self):

        credentials = configparser.ConfigParser()
        credentials.read(self.credential_file)

        connector = HSAFConnector()
        connector.connect(credentials=credentials)
        connector.download('h105', self.path, self.start_date, self.end_date,
                           file_limit=1)
        connector.close()


if __name__ == '__main__':
    unittest.main()
