import argparse
import sys
from ftplib import FTP
import getpass

import re
from datetime import datetime
import requests

#TODO: create general FTP and GET objects to pass queries to

def parse_main_args_download(args):
    parser = argparse.ArgumentParser(
        description='ASCAT download command line interface.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--source', help='Download source, EUMETSAT or HSAF')#FIXME: too specific
    parser.add_argument('-u', '--username', help='Username')
    parser.add_argument('-pw','--password', help='Password')
    parser.add_argument('-pr','--product', help='Name of product')
    
    #TODO: error handling when parameter is not passed

    return parser.parse_args(args), parser

def main_download(cli_args):
    args, parser = parse_main_args_download(cli_args)
    password = args.password#FIXME: getpass.getpass()
    if args.source.upper() == 'HSAF':
        download_hsaf(username=args.username,
                     password=password,
                     product=args.product)
    elif args.source.upper() == 'EUMETSAT':
        download_eumetsat(username=args.username,
                          password=password,
                          product=args.product)


def download_hsaf(username, password, product):
    #product options=('h10', 'h13', 'h34')
    with FTP('ftphsaf.meteoam.it') as ftp:
        try:
            import pdb;pdb.set_trace()
            ftp.login(username, password)
            print('Connected')
        except:
            print("Username or Password is incorrect")
        path = product + '/' + product + '_cur_mon_data'
        ftp.cwd(path)
        import pdb;pdb.set_trace()

def download_eumetsat(username, password, product):#TODO: date and productID to parse args
    
    access_token = 'TBD'
    apis_endpoint = "http://api.eumetsat.int/"

    service_search = apis_endpoint + "data/search-products/os"
    service_download = apis_endpoint + "data/download/"

    selected_collection_id = "EO:EUM:DAT:METOP:SOMO12"

    coords = [[-1.0, -1.0],[4.0, -4.0],[8.0, -2.0],[9.0, 2.0],[6.0, 4.0],[1.0, 5.0],[-1.0, -1.0]]
    start_date = datetime(2018, 12, 31)
    end_date = datetime(2019, 1, 2)
    #TODO: move search,product and collection to seperate methods
    # Format our paramters for searching
    dataset_parameters = {'format': 'json', 'pi': selected_collection_id}
    dataset_parameters['dtstart'] = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    dataset_parameters['dtend'] = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    dataset_parameters['geo'] = 'POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in coords]))

    # Retrieve datasets that match our filter
    url = service_search
    response = requests.get(url, dataset_parameters)
    
    found_data_sets = response.json()

    for selected_data_set in found_data_sets['features']:
        collID = selected_data_set['properties']['parentIdentifier']#FIXME: same as selected collid?
        import pdb;pdb.set_trace()
        
    date = datetime.strptime(selected_data_set['properties']['date'].split("/",1)[0] , '%Y-%m-%dT%H:%M:%SZ')
    download_url = service_download + urllib.parse.quote(
            'collections/{collID}/dates/{year}/{month}/{day}/times/{hour}/{minute}'.format(
            collID=collID, year=date.strftime('%Y'), 
            month=date.strftime('%m'), 
            day=date.strftime('%d'), hour=date.strftime('%H'), minute=date.strftime('%M')))
        

def run_download():
    main_download(sys.argv[1:])
    
if __name__ == "__main__":
    run_download()
