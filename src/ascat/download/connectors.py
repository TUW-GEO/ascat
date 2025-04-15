# Copyright (c) 2025, TU Wien
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
HTTP and FTP download module.
"""

import base64
import urllib
import requests
import logging
from pathlib import Path, WindowsPath, PosixPath
from ftplib import FTP
from datetime import timedelta
import concurrent.futures

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")


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

    def download_date_range(self, path, start_date, end_date):
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

    def download_file(self, file_remote, file_local):
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

    def download_file(self, file_remote, file_local, overwrite=False, n_retry=5):
        """
        Download single file from passed url to local file.

        Parameters
        ----------
        file_remote : string
            Path of file to download
        file_local : string
            Path (local) where to save file
        overwrite : bool, optional
            If True, existing files will be overwritten.
        """
        request_flag = False

        i = 0
        while i < n_retry:
            logging.debug("Send request")

            try:
                stream_response = requests.get(
                    file_remote,
                    params={"format": "json"},
                    stream=True,
                    headers={"Authorization": f"Bearer {self.access_token}"})

                self._assert_response(stream_response)
            except AssertionError:
                i += 1
                logging.debug(f"API Request failed - retry #{i}")
            else:
                logging.debug("API Request successful")
                request_flag = True
                break
        else:
            logging.debug("Maximum number of API requests failed. Abort.")

        if request_flag:
            total = int(stream_response.headers["content-length"])

            suffix = ""
            if stream_response.headers["Content-Type"] == "application/zip":
                suffix = ".zip"

            filename = file_local.parent / (file_local.name + suffix)

            if not overwrite and filename.exists():
                lstat = filename.lstat()
                if lstat.st_size == total:
                    logging.info("Skip download. File exits.")
            else:
                pbar = tqdm(desc=file_local.name,
                            total=total,
                            unit="B",
                            unit_divisor=1024,
                            unit_scale=True,
                            leave=False)

                with open(filename, "wb") as fp:
                    for chunk in stream_response.iter_content(chunk_size=1024):
                        if chunk:
                            fp.write(chunk)
                            fp.flush()
                            pbar.update(len(chunk))
                pbar.close()

                if filename.exists():
                    lstat = filename.lstat()
                    if lstat.st_size == total:
                        logging.debug("Download successful")
                    else:
                        logging.error("Download unsuccessful (file size mismatch)")
                else:
                    logging.error("Downloaded file not found")


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
            self.ftp.login(credentials["user"], credentials["password"])
            logging.info("FTP connection successfully established")
        except:
            logging.error("FTP connection failed. User or password incorrect")

    def download_file(self, file_remote, file_local, overwrite=False):
        """
        Download single file from passed url to local file

        Parameters
        ----------
        file_remote : string
            path of file to download
        file_local : string
            path (local) where to save file
        overwrite : bool, optional
            If True, existing files will be overwritten.
        """
        if file_remote not in self.ftp.nlst():
            logging.warning(f"File not accessible on FTP: {file_remote}")
        else:
            logging.debug(f"Start download: {file_remote}")

            total = self.ftp.size(file_remote)

            if not overwrite and file_local.exists():
                lstat = file_local.lstat()
                if lstat.st_size == total:
                    logging.info("Skip download. File exits.")
            else:
                pbar = tqdm(desc=file_local.name,
                            total=total,
                            unit="B",
                            unit_divisor=1024,
                            unit_scale=True,
                            dynamic_ncols=True,
                            leave=False)

                with open(file_local, "wb") as fp:

                    def cb(data):
                        pbar.update(len(data))
                        fp.write(data)

                    self.ftp.retrbinary(f"RETR {file_remote}", cb)

                pbar.close()

                if file_local.exists():
                    lstat = file_local.lstat()
                    if lstat.st_size == total:
                        logging.debug("Download successful")
                    else:
                        logging.error(
                            "Download unsuccessful (file size mismatch)")
                else:
                    logging.error("Downloaded file not found")

    def close(self):
        """
        Close connection.
        """
        self.ftp.close()
        logging.info("FTP disconnected")


class HsafConnector(FtpConnector):
    """
    Class for downloading from HSAF via FTP.
    """

    def __init__(self, base_url="ftphsaf.meteoam.it"):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string, optional
            Location of remote resource (default: ftphsaf.meteoam.it).
        """
        super().__init__(base_url)

    def download(self,
                 remote_path,
                 local_path,
                 start_date,
                 end_date,
                 limit=None,
                 overwrite=False):
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
        overwrite : bool, optional
            If True, existing files will be overwritten.
        """
        download_url_list = []
        local_file_list = []

        local_path = str2path(local_path)

        i = 0
        for daily_files in self.files(remote_path, start_date, end_date):
            for file_remote in daily_files:
                file_local = local_path / file_remote
                local_file_list.append(file_local)
                download_url_list.append(file_remote)

                i = i + 1
                if limit and limit == i:
                    break
            else:
                # continue if the inner loop wasn't broken
                continue

            # inner loop was broken, break the outer
            break

        download_url_list, local_file_list = zip(*sorted(zip(
            download_url_list, local_file_list)))

        with tqdm(desc="Downloads", total=len(download_url_list)) as pbar:
            for download_url, local_file in zip(download_url_list,
                                                local_file_list):
                self.download_file(download_url,
                                   local_file,
                                   overwrite=overwrite)
                pbar.update()

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
        self.ftp.retrlines("NLST ", list_of_files.append)

        days = end_date - start_date
        for i in range(days.days):
            date = (start_date + timedelta(days=i)).strftime("%Y%m%d")
            matches = sorted([x for x in list_of_files if date in x],
                             reverse=True)
            yield matches


class EumConnector(HttpConnector):
    """
    Class for downloading from EUMETSAT via HTTP requests.
    """

    def __init__(self, base_url="https://api.eumetsat.int"):
        """
        Initialize connector.

        Parameters
        ----------
        base_url : string, optional
            Location of remote resource (default: https://api.eumetsat.int).
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
            consumer_key=credentials["consumer_key"],
            consumer_secret=credentials["consumer_secret"])

    def download(self,
                 product,
                 local_path,
                 start_date,
                 end_date,
                 max_workers=1,
                 coords=None,
                 limit=None):
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
        max_workers : int, optional
            Number of parallel downloads (default: 1).
        coords : list of float, optional
            A custom polygon using EPSG:4326 decimal degrees (default: None).
        limit : int, optional
            Filter used to limit the returned results (default: None).
        """
        local_path = str2path(local_path)

        service_search = f"{self.base_url}/data/search-products/os"
        service_download = f"{self.base_url}/data/download/"

        dataset_parameters = {"format": "json", "pi": product}

        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        dataset_parameters["dtstart"] = start_date.strftime(fmt)
        dataset_parameters["dtend"] = end_date.strftime(fmt)

        if coords:
            dataset_parameters["geo"] = "POLYGON(({}))".format(",".join(
                [f"{coord[0]} {coord[1]}" for coord in coords]))

        url = service_search
        response = requests.get(url, dataset_parameters)
        found_data_sets = response.json()

        url = service_search
        dataset_parameters['si'] = 0
        items_per_page = 10

        if "type" in found_data_sets:
            if found_data_sets["type"] == "ExceptionReport":
                msg = found_data_sets["exceptions"][0]["exceptionText"]
                raise RuntimeError(msg)

        all_found_data_sets = []
        while dataset_parameters['si'] < found_data_sets['totalResults']:
            response = requests.get(url, dataset_parameters)
            found_data_sets = response.json()
            all_found_data_sets.append(found_data_sets)
            dataset_parameters[
                'si'] = dataset_parameters['si'] + items_per_page

        download_url_list = []
        local_file_list = []

        if all_found_data_sets:

            for found_data_sets in all_found_data_sets:
                for selected_data_set in found_data_sets["features"]:
                    coll_id = selected_data_set["properties"][
                        "parentIdentifier"]
                    product_id = selected_data_set["properties"]["identifier"]
                    url_temp = f"collections/{coll_id}/products/{product_id}"
                    download_url_list.append(service_download +
                                             urllib.parse.quote(url_temp))
                    local_file_list.append(local_path / product_id)

            print(f"Found {found_data_sets['totalResults']} data sets")

            if limit and len(download_url_list) > limit:
                print(f"Limited to {limit} data sets")
                download_url_list = download_url_list[:limit]
                local_file_list = local_file_list[:limit]

            download_url_list, local_file_list = zip(*sorted(zip(
                download_url_list, local_file_list)))

            concurrent_download(self.download_file, download_url_list,
                                local_file_list, max_workers)
        else:
            print("No data sets found")

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
        token_url = f"{self.base_url}/token"
        userpass = f"{consumer_key}:{consumer_secret}"
        encoded_userpass = base64.b64encode(userpass.encode()).decode()
        headers = {"Authorization": f"Basic {encoded_userpass}"}
        data_payload = {"grant_type": "client_credentials"}
        response = requests.post(token_url, headers=headers, data=data_payload)

        self._assert_response(response)

        return response.json()["access_token"]

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
        msg = f"API Request Failed: {response.status_code}\n{response.content}"
        assert (response.status_code == success_code), msg


def concurrent_download(download_func,
                        download_url_list,
                        local_file_list,
                        max_workers=1):
    """
    Threaded file download.

    Parameters
    ----------
    download_func : function
        Download function.
    download_url_list : list
        Download URLs.
    local_file_list : list
        Local filenames.
    max_workers : int, optional
        Number of concurrent downloads (default: 1).
    """
    if max_workers == 1:
        with tqdm(desc="Downloads", total=len(download_url_list)) as pbar:
            for download_url, local_file in zip(download_url_list,
                                                local_file_list):
                download_func(download_url, local_file)
                pbar.update()
    else:
        with tqdm(desc="Downloads", total=len(download_url_list)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers) as executor:

                future_down = {
                    executor.submit(download_func, url, local_file): url
                    for url, local_file in zip(download_url_list,
                                               local_file_list)
                }

                for future in concurrent.futures.as_completed(future_down):
                    _ = future_down[future]
                    pbar.update()


def str2path(path):
    """
    Convert str path to pathlib.Path object.

    Parameters
    ----------
    path : str, pathlib.Path
        Path.

    Returns
    -------
    path : pathlib.Path
        Pathlib path.
    """
    if not isinstance(path, (WindowsPath, PosixPath)):
        path = Path(path)

    return path
