import json
import os
import requests
from ftplib import FTP
from datetime import datetime, timedelta
import urllib
import re
import sys
#TODO: progress bar? tqdm?

class Connector:

    def __init__(self,base_url=None):
        self.base_url = base_url

    def connect(self, credentials=None):
        """
        connect to remote source.
        """
        pass

    def download(self, download_dir=None, start_date=None, end_date=None):
        """
        prepare download of files in daterange.
        """
        pass
    def grab_file(self, file_remote=None, file_local=None):
        """
        download single file from passed url to local file
        """
        pass

    def close(self):
        pass

class HTTPConnector(Connector):

    def __init__(self, base_url=None):
        self.base_url = base_url

    def grab_file(self, file_remote, file_local):

        stream_response = requests.get(
        file_remote,
        params= {'format': 'json'},
        stream=True,
        headers={'Authorization': 'Bearer {}'.format(self.access_token)}
        )

        assert_response(stream_response)
        tail=''
        if stream_response.headers['Content-Type'] == 'application/zip':
            tail='.zip'
        # Download the file (and display progress)
        progress = 0
        print("Downloading", file_local)
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

        print("\nDone")

class FTPConnector(Connector):

    def __init__(self, base_url=None):
        self.base_url = base_url
        self.ftp = FTP(base_url)
        super(FTPConnector, self).__init__(base_url = base_url)

    def connect(self, credentials=None):
        
        try:
            
            self.ftp.login(credentials["username"],
                      credentials["password"])
            print('Connected')
        except:
            print("Username or Password is incorrect")
   
    def grab_file(self, file_remote, file_local):

        if file_remote not in self.ftp.nlst():
            print(file_remote, "given file is not accesible in the FTP")
        else:
            
            localfile = open(file_local, 'wb')
            self.ftp.retrbinary('RETR ' + file_remote, localfile.write, 1024)
            localfile.close()

class HSAFConnector(FTPConnector):

    def __init__(self, base_url='ftphsaf.meteoam.it'):
        super(HSAFConnector, self).__init__(base_url = base_url)

    def download(self, product=None,
                       download_dir=None,
                       start_date=None,
                       end_date=None):

        dir = product + '/' + product + '_cur_mon_data'
        self.ftp.cwd(dir)

        init_date = datetime.strptime(start_date, "%Y%m%d")
        last_date = datetime.strptime(end_date, "%Y%m%d")
        filelist = [] 
        days = last_date - init_date

        
        tail = None
        if product == 'h10' or product == 'h34':
            tail = "_day_merged.H5.gz"
        else:
            tail = "_day_merged.grib2.gz"

        
        for i in range(days.days):
            date = ((init_date + timedelta(days=i)).strftime("%Y%m%d"))
            print(date)
            file_remote = product+"_"+date+tail
            file_local = os.path.join(download_dir, file_remote)
            
            self.grab_file(file_remote=file_remote, file_local=file_local)

class EumetsatConnector(HTTPConnector):

    def __init__(self, base_url="http://api.eumetsat.int/"):
        super(EumetsatConnector,self).__init__(base_url = base_url)

    def connect(self, credentials):
        
        self.access_token = self.__generate_token(consumer_key=credentials['consumer_key'],
                                        consumer_secret=credentials['consumer_secret'])


    def download(self, product=None,
                       download_dir=None,
                       coords=None,
                       start_date=None,
                       end_date=None):

        service_search = self.base_url + "data/search-products/os"
        service_download = self.base_url + "data/download/"

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        dataset_parameters = {'format': 'json', 'pi': product}
        dataset_parameters['dtstart'] = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        dataset_parameters['dtend'] = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        dataset_parameters['geo'] = 'POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in coords]))

        url = service_search
        response = requests.get(url, dataset_parameters)
        found_data_sets = response.json()
        
        for selected_data_set in found_data_sets['features']:

            collID = selected_data_set['properties']['parentIdentifier']
            date = datetime.strptime(selected_data_set['properties']['date'].split("/",1)[0] , '%Y-%m-%dT%H:%M:%SZ')
            download_url = service_download + urllib.parse.quote(
                    'collections/{collID}/dates/{year}/{month}/{day}/times/{hour}/{minute}'.format(
                    collID=collID, year=date.strftime('%Y'),
                    month=date.strftime('%m'),
                    day=date.strftime('%d'), hour=date.strftime('%H'), minute=date.strftime('%M')))
            
            file_local = os.path.join(download_dir, selected_data_set['properties']['identifier'])

            self.grab_file(file_remote=download_url,
                           file_local=file_local)



        
    def __generate_token(self, consumer_key=None, consumer_secret=None):
        '''
        Function to generate an access token for interacting with EUMETSAT Data 
        Service APIs

        Args:
            consumer_key (str):     The consumer key as a string
            consumer_secret (str):  The consumer secret as a string.
        
        Returns:
            An access token (if pass) or None (if fail).
        '''

        # build the token URL:
        token_url = self.base_url + "/token"

        response = requests.post(
            token_url,
            auth=requests.auth.HTTPBasicAuth(consumer_key, consumer_secret),
            data = {'grant_type': 'client_credentials'},
            headers = {"Content-Type" : "application/x-www-form-urlencoded"}
        )
        self.__assert_response(response)
        return response.json()['access_token']

    def __assert_response(self, response, success_code=200):
        '''
        Function to check API key generation response. Will return an error 
        if the key retrieval was not successful.

        Args:
            response (obj):      The authentication response.
            success_code (int):  The expected sucess code (200).

        Returns:
            Nothing if success, error message if fail.
        '''
        
        assert response.status_code == success_code,\
          "API Request Failed: {}\n{}".format(response.status_code, \
                                              response.content)


