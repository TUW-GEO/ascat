import argparse
import sys
from ftplib import FTP
import getpass
import urllib
import gzip
import re
import os
import sys
import json
from datetime import datetime, timedelta
import requests


from ascat.download_connectors import HSAFConnector, EumetsatConnector

def parse_main_args_download(args):
    parser = argparse.ArgumentParser(
        description='ASCAT download command line interface.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--source', choices=['hsaf','eumetsat'], help='Download source, EUMETSAT or HSAF')
    parser.add_argument('-pr','--product', help='Name of product')
    parser.add_argument('-cred','--credential_file', help='File where credentials are stored')
    parser.add_argument('-conf','--config_file', help='File where configs are stored')
    parser.add_argument('-o','--output_dir', help='Directory to write output')
    parser.add_argument('-from','--start_date', help='start date in YYYYMMDD format')
    parser.add_argument('-to','--end_date', help='end date in YYYYMMDD format')

    #TODO: error handling when parameter is not passed

    return parser.parse_args(args), parser


def read_json(filename):
    '''
    Function to read credentials or config from a JSON format
    file.
    
    Args:
        fiename (str):      The credentials/config filename

    Returns:
        data from file if success, error message if fail.
    '''

    try:
        with open(filename,'r') as json_file:
            output  = json.load(json_file)
    except:
        print(filename+' does not exist or is not in the correct format')
        return 
        
    print('Successfully retrieved data from file:'+filename)
    return output

def main_download(cli_args):
    args, parser = parse_main_args_download(cli_args)

    credentials = read_json(args.credential_file)
    config=None
    if args.config_file:
        config = read_json(args.config_file)
    #FIXME: seperate credential files per source?
    if args.source.upper() == 'HSAF':
        
        download_hsaf(credentials=credentials,
                     config=config,
                     product=args.product,
                     download_dir=args.output_dir,
                     start_date=args.start_date,
                     end_date=args.end_date)

    elif args.source.upper() == 'EUMETSAT':

        download_eumetsat(credentials=credentials,
                          config=config,
                          product=args.product,
                          download_dir=args.output_dir,
                          start_date=args.start_date,
                          end_date=args.end_date)
    

def download_hsaf(credentials=None,
                  config=None,
                  download_dir=None,
                  product=None,
                  start_date=None,
                  end_date=None):

    if config:
        download_dir = config['download_dir']
        product = config['product']
        start_date = config['start_date']
        end_date = config['end_date']

    connector = HSAFConnector()
    connector.connect(credentials=credentials)
    connector.download(product=product,
                       download_dir=download_dir,
                       start_date=start_date,
                       end_date=end_date)

def download_eumetsat(credentials=None,
                      config=None,
                      coords=None,
                      download_dir=None,
                      product=None,
                      start_date=None,
                      end_date=None):
    
    if config:
         download_dir = config['download_dir']
         product = config['product']
         coords = config['coords']
         start_date = config['start_date']
         end_date = config['end_date']


    connector = EumetsatConnector()
    connector.connect(credentials=credentials)

    connector.download(product=product,
                       download_dir=download_dir,
                       coords=coords,
                       start_date=start_date,
                       end_date=end_date)

def run_download():
    main_download(sys.argv[1:])
    
if __name__ == "__main__":
    run_download()
