import json
import os
import requests
from ftplib import FTP
from datetime import datetime, timedelta
import urllib
import re
import sys
from tqdm import tqdm


class Connector:


    """
    Base Class for connecting and downloading from remote source.
    
        
    Parameters
    ----------
    base_url : string
        location of remote ressource.
    """


    def __init__(self,base_url):

        self.base_url = base_url


    def connect(self, credentials):

        """
        Establish connection to remote source.
        
        Parameters
        ----------
        credentials: dict
            Dictionary of needed authentication parameters.

        """
        
        pass


    def download(self, download_dir, start_date, end_date):
     
        """
        Fetch resource location for download of multiple files in daterange.

        Parameters
        ----------
        download_dir : string
            local directory, where found datasets are stored.
        start_date : string
            start date of daterange interval, format: YYYYmmdd
        end_date : string
            end date of daterange interval, format: YYYYmmdd
                  
        """

        pass


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


        pass

    def close(self):
        """
        Close connection.
        """

        pass

class HTTPConnector(Connector):
    
    """
    Class for HTTP Requests.

    """

    def __init__(self, base_url):
        self.base_url = base_url

    def grab_file(self, file_remote, file_local):

        stream_response = requests.get(
        file_remote,
        params= {'format': 'json'},
        stream=True,
        headers={'Authorization': 'Bearer {}'.format(self.access_token)}
        )

        self._assert_response(stream_response)
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
        
        if os.path.exists(file_local+tail):
            print("\nDone")
        else:
            raise RuntimeError('Error locating downloaded file!')
            
class FTPConnector(Connector):

    """
    Class for downloading via FTP
    """

    def __init__(self, base_url):
        self.base_url = base_url
        self.ftp = FTP(base_url)
        super(FTPConnector, self).__init__(base_url = base_url)

    def connect(self, credentials):
        """
        Establish connection to FTP source.
        
        Parameters
        ----------
        credentials: dict
            Dictionary of needed authentication parameters.

        """
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
            print("Downloading file ...")
            self.ftp.retrbinary('RETR ' + file_remote, localfile.write, 1024)
            localfile.close()
            if os.path.exists(file_local):
                
                print(str(file_local)+" finished downloading")
            else:
                raise RuntimeError('Error locating downloaded file!')

    def close(self):
        
        """
        Close connection.
        """

        self.ftp.close()

class HSAFConnector(FTPConnector):

    """
    Class for downloading from HSAF via FTP.
    
    """

    def __init__(self, base_url='ftphsaf.meteoam.it'):
        super(HSAFConnector, self).__init__(base_url = base_url)

    def connect(self, credentials):
        """
        Establish connection to FTP source.
        
        Parameters
        ----------
        credentials: configparser
            configparser for needed authentication parameters.

        """
        try:
            
            self.ftp.login(credentials['HSAF']["username"],
                      credentials['HSAF']["password"])
            print('Connected')
        except:
            raise RuntimeError("Username or Password is incorrect")
            
        
    def download(self, product,
                       download_dir,
                       start_date,
                       end_date,
                       file_limit=-1):
        """
        Fetch resource location for download of multiple files in daterange.

        Parameters
        ----------
        product : string
            product string
        download_dir : string
            local directory, where found datasets are stored.
        start_date : string
            start date of daterange interval, format: YYYYmmdd
        end_date : string
            end date of daterange interval, format: YYYYmmdd
        file_limit : int, default: -1
            cancel download after file limit, ignored if -1
                  
        """
        
        #paths reference on hsaf server:
        #h08    ftphsaf.meteoam.it/h08/h08_cur_mon_nc/h08_20201221_072400_metopb_42857_ZAMG.nc
        #h10    ftphsaf.meteoam.it/h10/h10_cur_mon_data/h10_20201221_day_merged.H5.gz
        #h16    ftphsaf.meteoam.it/h16/h16_cur_mon_data/h16_20201221_000000_METOPB_42852_EUM.buf
        #h101   ftphsaf.meteoam.it/h101/h101_cur_mon_data/h101_20201221_000000_METOPA_73539_EUM.buf
        #h102   ftphsaf.meteoam.it/h102/h102_cur_mon_data/h102_20201221_000000_METOPA_73539_EUM.buf
        #h103   ftphsaf.meteoam.it/h103/h103_cur_mon_data/h103_20201221_000000_METOPB_42852_EUM.buf
        #h104   ftphsaf.meteoam.it/h104/h104_cur_mon_data/h104_20200909_083900_METOPC_09552_EUM.buf
        #h105   ftphsaf.meteoam.it/h105/h105_cur_mon_data/h105_20200909_083900_METOPC_09552_EUM.buf
        
        
        dict_dirs = {'h08' : 'h08/h08_cur_mon_nc',
                     'h10' : 'h10/h10_cur_mon_data',
                     'h16' : 'h16/h16_cur_mon_data',
                     'h101': 'h101/h101_cur_mon_data/',
                     'h102': 'h102/h102_cur_mon_data/',
                     'h103': 'h103/h103_cur_mon_data/',
                     'h104': 'h104/h104_cur_mon_data/',
                     'h105': 'h105/h105_cur_mon_data/'}
        dir = dict_dirs[product]
        self.ftp.cwd(dir)
        
        init_date = datetime.strptime(start_date, "%Y%m%d")
        last_date = datetime.strptime(end_date, "%Y%m%d")
        filelist = [] 
        days = last_date - init_date


        list_of_files = []
        self.ftp.retrlines('NLST ', list_of_files.append)
        
        for i in tqdm(range(days.days)):
            
            date = ((init_date + timedelta(days=i)).strftime("%Y%m%d"))
            
            #FIXME: could be made more performant
            matches = [x for x in list_of_files if date in x]
            n_grabbed_files=0
            for file_remote in matches:
                
                file_local = os.path.join(download_dir, file_remote)
                
                self.grab_file(file_remote=file_remote, file_local=file_local)
                n_grabbed_files+=1
                if n_grabbed_files - file_limit == 0:
                    break
            
class EumetsatConnector(HTTPConnector):

    """
    Class for downloading from eumetsat via HTTP requests.

    """


    def __init__(self, base_url="http://api.eumetsat.int"):
        super(EumetsatConnector,self).__init__(base_url = base_url)

    def connect(self, credentials):
        
        """
        Establish connection to EUMETSAT.
        
        Parameters
        ----------
        credentials: dict
            Dictionary of needed authentication parameters.

        """
        
        self.access_token = self._generate_token(consumer_key=credentials['EUMETSAT']['consumer_key'],
                                        consumer_secret=credentials['EUMETSAT']['consumer_secret'])


    def download(self, product,
                       download_dir,
                       coords,
                       start_date,
                       end_date,
                       file_limit=-1):
        """
        Fetch resource location for download of multiple files in daterange.

        Parameters
        ----------
        product : string
            product string
        download_dir : string
            local directory, where found datasets are stored.
        coords: list
            coordinates of polygon, where files will be downloaded in
        start_date : string
            start date of daterange interval, format: YYYYmmdd
        end_date : string
            end date of daterange interval, format: YYYYmmdd
        file_limit : int, default: -1
            cancel download after file limit, ignored if -1
                  
        """
        service_search = self.base_url + "/data/search-products/os"
        service_download = self.base_url + "/data/download/"

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        dataset_parameters = {'format': 'json', 'pi': product}
        dataset_parameters['start'] = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        dataset_parameters['end'] = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        dataset_parameters['geo'] = 'POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in coords]))
        

        
        url = service_search
        response = requests.get(url, dataset_parameters)
        found_data_sets = response.json()
        
        n_grabbed_files = 0
        for selected_data_set in tqdm(found_data_sets['features']):

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
            
            n_grabbed_files+=1
            if n_grabbed_files - file_limit == 0:
                break
                


        
    def _generate_token(self, consumer_key, consumer_secret):
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
        
        self._assert_response(response)
        return response.json()['access_token']

    def _assert_response(self, response, success_code=200):
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


