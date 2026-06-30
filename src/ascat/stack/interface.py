# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

import sys
import argparse
from pathlib import Path
from datetime import datetime

import re

from ascat.cell import CellGridFiles
from ascat.swath import SwathGridFiles

# based on https://stackoverflow.com/a/42865957/2002471
units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def parse_size(size):
    size = size.upper()
    #print("parsing size ", size)
    if not re.match(r' ', size):
        size = re.sub(r'([KMGT]?B)', r' \1', size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])


class KeywordsAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        keyword_dict = {}

        for arg in values:  #values => The args found for keyword_args
            pieces = arg.split('=')
            keyword_dict[pieces[0]] = pieces[1]

        setattr(namespace, self.dest, keyword_dict)


def parse_args_swath_stacker(args):
    """
    Parse command line arguments for stacking ASCAT swath files into a cell grid.

    Parameters
    ----------
    args : list
        Command line arguments.

    Returns
    -------
    parser : ArgumentParser
        Argument Parser object.
    """
    parser = argparse.ArgumentParser(
        description="Stack ASCAT swath files to a cell grid")
    parser.add_argument(
        "filepath",
        metavar="FILEPATH",
        type=str,
        help="Path to folder containing swath files")
    parser.add_argument(
        "outpath", metavar="OUTPATH", type=str, help="Path to the output data")
    parser.add_argument(
        "product_id",
        metavar="PRODUCT_ID",
        type=str,
        help="Product identifier")
    parser.add_argument(
        "--start_date",
        metavar="START_DATE",
        type=str,
        help="Start date in format YYYY-MM-DD. Must also provide end date if this is provided."
    )
    parser.add_argument(
        "--end_date",
        metavar="END_DATE",
        type=str,
        help="End date in format YYYY-MM-DD. Must also provide start date if this is provided."
    )
    parser.add_argument(
        "--dump_size",
        metavar="DUMP_SIZE",
        type=str,
        help="Size at which to dump the data to disk before reading more (default: 1GB)"
    )
    parser.add_argument(
        "--cells",
        metavar="CELLS",
        type=int,
        nargs='+',
        help="Numbers of the cells to process (default: None)")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress information")
    parser.add_argument(
        "fmt_kwargs",
        help="Format keyword arguments, depends on the product format used. Example: 'sat=A year=2008'",
        nargs='*',
        action=KeywordsAction)

    return parser.parse_args(args)


def swath_stacker_main(cli_args):
    """
    Main function for stacking ASCAT swath files into a cell grid given command line arguments.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args = parse_args_swath_stacker(cli_args)
    filepath = Path(args.filepath)
    product_id = args.product_id

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    dump_size = parse_size(
        args.dump_size) if args.dump_size else parse_size("4GB")

    date_range = (args.start_date, args.end_date)
    if any(date_range) and not all(date_range):
        raise ValueError(
            "Both start and end date must be provided, or neither.")
    date_range = tuple((
        datetime.strptime(d, "%Y-%m-%d") for d in date_range if d is not None))
    date_range = None if not any(date_range) else date_range

    cells = args.cells

    quiet = args.quiet

    fmt_kwargs = args.fmt_kwargs

    swath_files = SwathGridFiles.from_product_id(filepath, product_id)
    if not quiet:
        print("Initializing...")
    swath_files.stack_to_cell_files(
        outpath,
        dump_size,
        date_range=date_range,
        fmt_kwargs=fmt_kwargs,
        cells=cells,
        print_progress=(not quiet),
    )


def parse_args_cell_format_converter(args):
    """
    Parse command line arguments for

    Parameters
    ----------
    args : list
        Command line arguments.

    Returns
    -------
    parser : ArgumentParser
        Argument Parser object.
    """
    parser = argparse.ArgumentParser(
        description="Reformat ASCAT cell files to various CF Array formats")
    parser.add_argument(
        "filepath",
        metavar="FILEPATH",
        type=str,
        help="Path to folder containing swath files")
    parser.add_argument(
        "outpath", metavar="OUTPATH", type=str, help="Path to the output data")
    parser.add_argument(
        "product_id",
        metavar="PRODUCT_ID",
        type=str,
        help="Product identifier")
    parser.add_argument(
        "arr_format",
        metavar="FORMAT",
        type=str,
        help="Output format (indexed, contiguous, or point.)")
    parser.add_argument(
        "--sf_format",
        metavar="SF_FORMAT",
        type=str,
    )

    return parser.parse_args(args)


def cell_format_converter_main(cli_args):

    args = parse_args_cell_format_converter(cli_args)
    filepath = Path(args.filepath)
    product_id = args.product_id
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    array_format = args.arr_format
    sf_format = args.sf_format

    cell_files = CellGridFiles.from_product_id(
        filepath, product_id, sf_format=sf_format)

    if array_format == "contiguous":
        cell_files.convert_to_contiguous(outpath)
    elif array_format == "indexed":
        cell_files.reprocess(outpath, lambda ds: ds, ra_type="indexed")
    elif array_format == "point":
        cell_files.reprocess(outpath, lambda ds: ds, ra_type="point")
    else:
        raise ValueError(f"Invalid array format: {array_format}")


def run_swath_stacker():
    """Run command line interface for temporal aggregation of ASCAT data."""
    swath_stacker_main(sys.argv[1:])


def run_cell_format_converter():
    """Run command line interface for temporal aggregation of ASCAT data."""
    cell_format_converter_main(sys.argv[1:])
