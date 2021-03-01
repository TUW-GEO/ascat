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
import configparser

from src.ascat.download_connectors import HSAFConnector, EumetsatConnector

def parse_main_args_download(args):
    """
    Parse command line arguments.

    Parameters
    ----------
    args : list
        Command line arguments.

    Returns
    -------
    args : list
        Parsed arguments.
    parser : ArgumentParser
        Argument Parser object.
    """

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
    """
    Command line interface routine.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """

    args, parser = parse_main_args_download(cli_args)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = sys.argv[0]
    if args.product is not None and args.start_date is not None \
            and args.end_date is not None or args.config_file:
        
        credentials = configparser.ConfigParser()
        credentials.read(args.credential_file)
        
        if args.source.upper() == 'HSAF':
            
            if args.config_file:
                config = read_json(args.config_file)
                download_dir = config['download_dir']
                product = config['product']
                start_date = config['start_date']
                end_date = config['end_date']
            else:
                download_dir = output_dir
                product = args.product
                start_date = args.start_date
                end_date = args.end_date
                
            download_hsaf(credentials=credentials,
                         product=product,
                         download_dir=download_dir,
                         start_date=start_date,
                         end_date=end_date)

        elif args.source.upper() == 'EUMETSAT':

            if args.config_file:
                 
                config = read_json(args.config_file)
                download_dir = config['download_dir']
                product = config['product']
                start_date = config['start_date']
                end_date = config['end_date']
                coords = config['coords']
                
            else:
                
                download_dir = output_dir
                product = args.product
                start_date = args.start_date
                end_date = args.end_date
                
            download_eumetsat(credentials=credentials,
                              product=product,
                              download_dir=download_dir,
                              start_date=start_date,
                              end_date=end_date,
                              coords=coords)
        else:

            raise RuntimeError('Specified source not recognized!')
    else:
        raise RuntimeError('Product and date range need to be specified!')


def download_hsaf(credentials,
                  download_dir,
                  product,
                  start_date,
                  end_date):
    """
    Routine to connect to HSAF and download specified files.

    Parameters
    ----------
    credentials : dict
        Contains credentials needed for connecting to HSAF service.
    download_dir : string
        Where to download files to.
    product : string
        Product to search for.
    start_date : string
        start date of files to search for, format YYYYMMDD.
    end_date : string
        end date of files to search for, format YYYYMMDD.

    Returns
    -------
    None.

    """
    connector = HSAFConnector()
    connector.connect(credentials=credentials)
    
    connector.download(product=product,
                       download_dir=download_dir,
                       start_date=start_date,
                       end_date=end_date)

def download_eumetsat(credentials,
                      download_dir,
                      product,
                      start_date,
                      end_date,
                      coords):
    """
    Routine to connect to EUMETSAT and download specified files

    Parameters
    ----------
    credentials : dict
        Contains credentials needed for connecting to EUMETSAT service.
    download_dir : string
        Where to download files to.
    product : string
        Product to search for.
    start_date : string
        start date of files to search for, format YYYYMMDD.
    end_date : string
        end date of files to search for, format YYYYMMDD.
    coords : list
        coordinates of polygon in which files are searched.

    Returns
    -------
    None.

    """
    

    connector = EumetsatConnector()
    connector.connect(credentials=credentials)

    connector.download(product=product,
                       download_dir=download_dir,
                       coords=coords,
                       start_date=start_date,
                       end_date=end_date)

def run_download():
    """
    Run command line interface.
    """

    main_download(sys.argv[1:])
    

