# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test download.
"""

import os
import unittest
from tempfile import mkdtemp
from datetime import datetime, timedelta

from ascat.download.connectors import HsafConnector
from ascat.download.connectors import EumConnector

credentials = {
    'EUM': {
        'consumer_key': os.getenv('EUM_CONSUMER_KEY'),
        'consumer_secret': os.getenv('EUM_CONSUMER_SECRET')
    },
    'HSAF': {
        'user': os.getenv('HSAF_FTP_USER'),
        'password': os.getenv('HSAF_FTP_PASSWORD')
    }
}


class TestDownload(unittest.TestCase):

    def setUp(self):
        """
        Get connection details.
        """
        self.local_path = mkdtemp()

        yesterday = (datetime.now() - timedelta(1))
        today = datetime.today()
        self.start_date = yesterday
        self.end_date = today

    @unittest.skipIf(credentials['EUM']['consumer_key'] is None,
                     "Skip EUMETSAT connection test")
    def test_eumetsat_connect(self):
        """
        Test EUMETSAT connection
        """
        connector = EumConnector()
        connector.connect(credentials['EUM'])

    @unittest.skipIf(credentials['EUM']['consumer_key'] is None,
                     "Skip EUMETSAT download test")
    def test_eumetsat_download(self):
        """
        Test EUMETSAT download.
        """
        product = "EO:EUM:DAT:METOP:SOMO12"

        connector = EumConnector()
        connector.connect(credentials['EUM'])
        connector.download(
            product, self.local_path, self.start_date, self.end_date, limit=1)

    @unittest.skipIf(credentials['HSAF']['user'] is None,
                     "Skip H SAF connection test")
    def test_hsaf_connect(self):
        """
        Test H SAF connection.
        """
        connector = HsafConnector()
        connector.connect(credentials['HSAF'])
        connector.close()

    @unittest.skipIf(credentials['HSAF']['user'] is None,
                     "Skip H SAF download test")
    def test_hsaf_download(self):
        """
        Test H SAF download.
        """
        remote_path = '/products/h08/h08_cur_mon_nc'

        connector = HsafConnector()
        connector.connect(credentials['HSAF'])
        connector.download(
            remote_path,
            self.local_path,
            self.start_date,
            self.end_date,
            limit=1)
        connector.close()


if __name__ == '__main__':
    unittest.main()
