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
import urllib
import requests
import logging
from ftplib import FTP
from datetime import datetime
from datetime import timedelta

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


class Connector:

    """
    Base class for connecting and downloading from remote source.
    """

    def __init__(self, base_url):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string
            Location of remote resource.
        """
        self.base_url = base_url

    def connect(self, credentials):
        """
        Establish connection to remote source.

        Parameters
        ----------
        credentials : dict
            Dictionary of needed authentication parameters.
        """
        pass

    def download(self, path, start_date, end_date):
        """
        Fetch resource location for download of multiple files in date range.

        Parameters
        ----------
        path : string
            Local directory, where found datasets are stored.
        start_date : datetime
            Start date of date range interval.
        end_date : datetime
            End date of date range interval.
        """
        pass

    def grab_file(self, file_remote, file_local):
        """
        Download single file from passed url to local file.

        Parameters
        ----------
        file_remote : string
            Path of file to download.
        file_local : string
            Path (local) where to save file.
        """
        pass

    def close(self):
        """
        Close connection.
        """
        pass


class HttpConnector(Connector):

    """
    Class for http requests.
    """

    def __init__(self, base_url):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string
            Location of remote resource.
        """
        self.base_url = base_url

    def grab_file(self, file_remote, file_local):
        """
        Download single file from passed url to local file.

        Parameters
        ----------
        file_remote : string
            Path of file to download
        file_local : string
            Path (local) where to save file
        """
        stream_response = requests.get(
            file_remote,
            params={'format': 'json'},
            stream=True,
            headers={'Authorization': 'Bearer {}'.format(self.access_token)})

        self._assert_response(stream_response)

        tail = ''
        if stream_response.headers['Content-Type'] == 'application/zip':
            tail = '.zip'

        # Download the file (and display progress)
        progress = 0

        logging.info('Start download: {}'.format(file_local))

        with open(file_local+tail, 'wb') as f:
            for chunk in stream_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    if(progress % 1024 == 0):
                        sys.stdout.write("\r%dkB" % progress)
                        sys.stdout.flush()
                    progress += 1
            sys.stdout.write("\r%dkB" % progress)
            sys.stdout.flush()

        if os.path.exists(file_local+tail):
            logging.info('Download finished')
        else:
            logging.error('Downloaded file not found')


class FtpConnector(Connector):

    """
    Class for downloading via FTP
    """

    def __init__(self, base_url):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string
            Location of remote resource.
        """
        super().__init__(base_url)
        self.ftp = FTP(self.base_url)

    def connect(self, credentials):
        """
        Establish connection to FTP source.

        Parameters
        ----------
        credentials : dict
            Dictionary of needed authentication parameters.
        """
        try:
            self.ftp.login(credentials['user'], credentials['password'])
            logging.info('FTP connection successfully established')
        except:
            logging.error('FTP connection failed. User or password incorrect')

    def grab_file(self, file_remote, file_local):
        """
        Download single file from passed url to local file

        Parameters
        ----------
        file_remote : string
            path of file to download
        file_local : string
            path (local) where to save file
        """
        if file_remote not in self.ftp.nlst():
            logging.warning('File not accessible on FTP: {}'.format(
                file_remote))
        else:
            localfile = open(file_local, 'wb')
            logging.info('Start download: {}'.format(file_remote))
            self.ftp.retrbinary('RETR ' + file_remote, localfile.write, 1024)
            localfile.close()
            if os.path.exists(file_local):
                logging.info('Finished download: {}'.format(file_local))
            else:
                logging.error('Downloaded file not found')

    def close(self):
        """
        Close connection.
        """
        self.ftp.close()
        logging.info('FTP disconnect')


class HsafConnector(FtpConnector):

    """
    Class for downloading from HSAF via FTP.
    """

    def __init__(self, base_url='ftphsaf.meteoam.it'):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string, optional
            Location of remote resource (default: ftphsaf.meteoam.it).
        """
        super().__init__(base_url)

    def download(self, remote_path, local_path, start_date, end_date,
                 limit=None):
        """
        Fetch resource location for download of multiple files in date range.

        Parameters
        ----------
        remote_path : string
            Remote directory, where found datasets are stored.
        local_path : string
            Local directory, where found datasets are stored.
        start_date : datetime
            Start date of date range interval.
        end_date : datetime
            End date of date range interval.
        limit : int, optional
            Filter used to limit the returned results (default: 1).
        """
        i = 0
        for daily_files in self.files(remote_path, start_date, end_date):
            for file_remote in daily_files:
                file_local = os.path.join(local_path, file_remote)
                self.grab_file(file_remote, file_local)
                i = i + 1
                if limit and limit == i:
                    break

    def files(self, remote_path, start_date, end_date):
        """
        Generator retrieving file list for given date range.

        Parameters
        ----------
        remote_path : string
            Remote directory, where found datasets are stored.
        local_path : string
            Local directory, where found datasets are stored.
        start_date : datetime
            Start date of date range interval.
        end_date : datetime
            End date of date range interval.

        Yields
        ------
        matches : list
            List of daily files.
        """
        self.ftp.cwd(remote_path)

        list_of_files = []
        self.ftp.retrlines('NLST ', list_of_files.append)

        days = end_date - start_date
        for i in tqdm(range(days.days)):
            date = ((start_date + timedelta(days=i)).strftime("%Y%m%d"))
            matches = sorted([x for x in list_of_files if date in x],
                             reverse=True)
            yield matches


class EumConnector(HttpConnector):

    """
    Class for downloading from EUMETSAT via HTTP requests.
    """

    def __init__(self, base_url="http://api.eumetsat.int"):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string, optional
            Location of remote resource (default: http://api.eumetsat.int).
        """
        super().__init__(base_url)

    def connect(self, credentials):
        """
        Establish connection to EUMETSAT.

        Parameters
        ----------
        credentials: dict
            Dictionary of needed authentication parameters.
        """
        self.access_token = self._generate_token(
            consumer_key=credentials['consumer_key'],
            consumer_secret=credentials['consumer_secret'])

    def download(self, product, local_path, start_date, end_date,
                 coords=None, limit=None):
        """
        Fetch resource location for download of multiple files in daterange.

        Parameters
        ----------
        product : string
            Product.
        coords : list
            Coordinates of polygon, where files will be downloaded
        local_path : string
            Local directory, where found datasets are stored.
        start_date : datetime
            Start date of date range interval.
        end_date : datetime
            End date of date range interval.
        coords : list of float, optional
            A custom polygon using EPSG:4326 decimal degrees (default: None).
        limit : int, optional
            Filter used to limit the returned results (default: None).
        """
        service_search = self.base_url + "/data/search-products/os"
        service_download = self.base_url + "/data/download/"

        dataset_parameters = {'format': 'json', 'pi': product}

        dataset_parameters['start'] = start_date.strftime(
            '%Y-%m-%dT%H:%M:%S.%fZ')
        dataset_parameters['end'] = end_date.strftime(
            '%Y-%m-%dT%H:%M:%S.%fZ')

        if coords:
            dataset_parameters['geo'] = 'POLYGON(({}))'.format(
                ','.join(["{} {}".format(*coord) for coord in coords]))

        url = service_search
        response = requests.get(url, dataset_parameters)
        found_data_sets = response.json()

        url_temp = ('collections/{coll_id}/dates/{year}/{month}/{day}/'
                    'times/{hour}/{minute}')

        i = 0
        for selected_data_set in tqdm(found_data_sets['features']):

            coll_id = selected_data_set['properties']['parentIdentifier']
            date = datetime.strptime(selected_data_set[
                'properties']['date'].split("/", 1)[0], '%Y-%m-%dT%H:%M:%SZ')

            download_url = service_download + urllib.parse.quote(
                url_temp.format(
                    coll_id=coll_id, year=date.strftime('%Y'),
                    month=date.strftime('%m'), day=date.strftime('%d'),
                    hour=date.strftime('%H'), minute=date.strftime('%M')))

            file_local = os.path.join(
                local_path, selected_data_set['properties']['identifier'])

            self.grab_file(download_url, file_local)

            i = i + 1
            if limit and limit == i:
                break

    def _generate_token(self, consumer_key, consumer_secret):
        """
        Function to generate an access token for interacting with
        EUMETSAT Data Service APIs.

        Parameters
        ----------
        consumer_key : str
            The consumer key as a string
        consumer_secret : str
            The consumer secret as a string.

        Returns
        -------
        access_token : str
            An access token (if pass) or None (if fail).
        """
        token_url = self.base_url + "/token"

        response = requests.post(
            token_url,
            auth=requests.auth.HTTPBasicAuth(consumer_key, consumer_secret),
            data={'grant_type': 'client_credentials'},
            headers={"Content-Type": "application/x-www-form-urlencoded"})

        self._assert_response(response)
        return response.json()['access_token']

    def _assert_response(self, response, success_code=200):
        """
        Function to check API key generation response. Will return an error
        if the key retrieval was not successful.

        Parameters
        ----------
        response : obj
            The authentication response.
        success_code : int, optional
            The expected sucess code (default: 200).

        Returns
        -------
        result : None or str
            Nothing if success, error message if fail.
        """
        assert response.status_code == success_code,\
            "API Request Failed: {}\n{}".format(response.status_code,
                                                response.content)
