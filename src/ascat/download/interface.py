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


def hsaf_download(credentials, remote_path, local_path, start_date,
                  end_date, limit=None):
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


def eumetsat_download(credentials, product, local_path, start_date,
                      end_date, coords=None, limit=None):
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
    coords : list of float, optional
        A custom polygon using EPSG:4326 decimal degrees (default: None).
    limit : int, optional
        Filter used to limit the returned results (default: None).
    """
    connector = EumConnector()
    connector.connect(credentials)
    connector.download(product, local_path, start_date, end_date, coords,
                       limit)


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

    parser.add_argument('-cf', '--credential_file',
                        help='File where credentials are stored')

    parser.add_argument('-r', '--remote_path', help='FTP remote path')

    parser.add_argument('-o', '--output_dir', help='Directory to write output')

    parser.add_argument('-from', '--start_date', type=parse_date,
                        help='start date in YYYYMMDD format')

    parser.add_argument('-to', '--end_date', type=parse_date,
                        help='end date in YYYYMMDD format')

    parser.add_argument('-co', '--coords',
                        help='end date in YYYYMMDD format')

    parser.add_argument('-l', '--limit',
                        help='Filter number of results')

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

    parser.add_argument('-cf', '--credential_file',
                        help='File where credentials are stored')

    parser.add_argument('-p', '--product', help='Name of product')

    parser.add_argument('-o', '--output_dir', help='Directory to write output')

    parser.add_argument('-from', '--start_date', type=parse_date,
                        help='start date in YYYYMMDD format')

    parser.add_argument('-to', '--end_date', type=parse_date,
                        help='end date in YYYYMMDD format')

    parser.add_argument('-co', '--coords',
                        help='end date in YYYYMMDD format')

    parser.add_argument('-l', '--limit',
                        help='Filter number of results')

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
                      args.start_date, args.end_date, args.coords, args.limit)


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
