# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Download interface.
"""

import sys
import datetime
import argparse
import configparser

from ascat.download.connectors import HsafConnector
from ascat.download.connectors import EumConnector


def parse_date(s):
    """
    Convert date string into datetime timestamp.

    Parameters
    ----------
    s : str
        Date string.

    Returns
    -------
    data : datetime
        Timestamp.
    """
    return datetime.datetime.strptime(s, '%Y%m%d')


def hsaf_download(credentials,
                  remote_path,
                  local_path,
                  start_date,
                  end_date,
                  limit=None):
    """
    Function to start H SAF download.

    Parameters
    ----------
    credentials : dict
        Dictionary of needed authentication parameters ('user', 'password').
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
    connector = HsafConnector()
    connector.connect(credentials)
    connector.download(remote_path, local_path, start_date, end_date, limit)
    connector.close()


def eumetsat_download(credentials,
                      product,
                      local_path,
                      start_date,
                      end_date,
                      max_workers=1,
                      coords=None,
                      limit=None):
    """
    Function to start EUMETSAT download.

    Parameters
    ----------
    credentials : dict
        Dictionary of needed authentication parameters ('user', 'password').
    remote_path : string
        Remote directory, where found datasets are stored.
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
    connector = EumConnector()
    connector.connect(credentials)
    connector.download(product, local_path, start_date, end_date, max_workers,
                       coords, limit)


def parse_args_hsaf_download(args):
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
        description='H SAF download command line interface.',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-cf',
                        '--credential_file',
                        help='File where credentials are stored')

    parser.add_argument('-r', '--remote_path', help='FTP remote path')

    parser.add_argument('-o', '--output_dir', help='Directory to write output')

    parser.add_argument('-from',
                        '--start_date',
                        type=parse_date,
                        help='start date in YYYYMMDD format')

    parser.add_argument('-to',
                        '--end_date',
                        type=parse_date,
                        help='end date in YYYYMMDD format')

    parser.add_argument(
        '-co',
        '--coords',
        help='A custom polygon using EPSG:4326 decimal degrees')

    parser.add_argument('-l', '--limit', help='Filter number of results')

    return parser.parse_args(args), parser


def parse_args_eumetsat_download(args):
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
        description='EUMETSAT download command line interface.',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-cf',
                        '--credential_file',
                        help='File where credentials are stored')

    parser.add_argument('-p', '--product', help='Name of product')

    parser.add_argument('-o', '--output_dir', help='Directory to write output')

    parser.add_argument('-from',
                        '--start_date',
                        type=parse_date,
                        help='start date in YYYYMMDD format')

    parser.add_argument('-to',
                        '--end_date',
                        type=parse_date,
                        help='end date in YYYYMMDD format')

    parser.add_argument('-mw',
                        '--max_workers',
                        type=int,
                        help='Number of parallel downloads')

    parser.add_argument(
        '-co',
        '--coords',
        help='A custom polygon using EPSG:4326 decimal degrees')

    parser.add_argument('-l', '--limit', help='Filter number of results')

    return parser.parse_args(args), parser


def hsaf_main(cli_args):
    """
    Command line interface routine.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args, parser = parse_args_hsaf_download(cli_args)

    credentials = configparser.ConfigParser()
    credentials.read(args.credential_file)

    hsaf_download(credentials['hsaf'], args.remote_path, args.output_dir,
                  args.start_date, args.end_date, args.limit)


def eumetsat_main(cli_args):
    """
    Command line interface routine.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args, parser = parse_args_eumetsat_download(cli_args)

    credentials = configparser.ConfigParser()
    credentials.read(args.credential_file)

    eumetsat_download(credentials['eumetsat'], args.product, args.output_dir,
                      args.start_date, args.end_date, args.max_workers,
                      args.coords, args.limit)


def run_hsaf_download():
    """
    Run command line interface for H SAF download.
    """
    hsaf_main(sys.argv[1:])


def run_eumetsat_download():
    """
    Run command line interface for EUMETSAT download.
    """
    eumetsat_main(sys.argv[1:])
