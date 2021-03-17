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
Test download.
"""

import os
import unittest
from tempfile import mkdtemp
from datetime import datetime, timedelta

from ascat.download.connectors import HsafConnector
from ascat.download.connectors import EumConnector

credentials = {'EUM': {'consumer_key': os.getenv('EUM_CONSUMER_KEY'),
                       'consumer_secret': os.getenv('EUM_CONSUMER_SECRET')},
               'HSAF': {'user': os.getenv('HSAF_FTP_USER'),
                        'password': os.getenv('HSAF_FTP_PASSWORD')}}

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
        connector.download(product, self.local_path, self.start_date,
                           self.end_date, limit=1)

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
        connector.download(remote_path, self.local_path, self.start_date,
                           self.end_date, limit=1)
        connector.close()

if __name__ == '__main__':
    unittest.main()
