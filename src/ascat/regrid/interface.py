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

import sys
import argparse
from pathlib import Path

import xarray as xr

from ascat.grids.grid_registry import GridRegistry
from ascat.product_info import get_swath_product_id
from ascat.product_info import swath_io_catalog
from ascat.regrid.regrid import regrid_swath_ds
from ascat.regrid.regrid import retrieve_or_store_grid_lut


def parse_args_swath_regrid(args):
    """
    Parse command line arguments for regridding an ASCAT swath file
    to a regular grid.

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
        description="Regrid an ASCAT swath file to a regular grid")
    parser.add_argument(
        "filepath", metavar="FILEPATH", help="Path to file or folder")
    parser.add_argument(
        "outpath", metavar="OUTPATH", help="Path to the output data")
    parser.add_argument(
        "regrid_deg",
        metavar="REGRID_DEG",
        type=float,
        help="Target grid spacing in degrees")
    parser.add_argument(
        "--product_id",
        metavar="PRODUCT_ID",
        help="Product identifier (e.g. H129, H125, H121, etc.). If not provided, an attempt is made to determine it from the file name.")
    parser.add_argument(
        "--grid_store",
        metavar="GRID_STORE",
        help="Path for storing/loading lookup tables")
    parser.add_argument(
        "--suffix",
        metavar="SUFFIX",
        help="File suffix (default: _REGRID_DEGdeg)")

    return parser.parse_args(args)


def swath_regrid_main(cli_args):
    """
    Regrid an ASCAT swath file or directory of swath files
    to a regular grid and write the results to disk.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args = parse_args_swath_regrid(cli_args)
    filepath = Path(args.filepath)
    trg_grid_size = args.regrid_deg

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if args.suffix:
        suffix = args.suffix
    else:
        suffix = f"_{args.regrid_deg}deg"

    if filepath.is_dir():
        files = list(filepath.glob("**/*.nc"))
    elif filepath.is_file() and filepath.suffix == ".nc":
        files = [filepath]
    else:
        raise RuntimeError("No files found at the provided filepath")

    first_file = files[0]

    if args.product_id:
        product_id = args.product_id
    else:
        try:
            product_id = get_swath_product_id(str(first_file.name))
        except ValueError:
            raise RuntimeError(
                f"Could not determine product identifier from file name {str(first_file.name)} "
                "Please provide the --product_id argument."
            )

    if product_id is None:
        raise RuntimeError(
            "Could not determine product identifier from file name. "
            "Please provide the --product_id argument."
        )

    registry = GridRegistry()
    product = swath_io_catalog[product_id]
    src_grid = registry.get(product.grid_name)
    src_grid_size = src_grid.res

    src_grid_id = f"fib_grid_{src_grid_size}km"
    trg_grid_id = f"reg_grid_{trg_grid_size}deg"

    trg_grid, grid_lut = retrieve_or_store_grid_lut(
        src_grid,
        src_grid_id,
        trg_grid_id,
        trg_grid_size,
        args.grid_store)

    for f in files:
        outfile = outpath / Path(f.stem + suffix + f.suffix)

        with xr.open_dataset(f, decode_cf=False, mask_and_scale=False) as ds:
            regrid_ds = regrid_swath_ds(ds, src_grid, trg_grid, grid_lut)
            regrid_ds.to_netcdf(outfile)


def run_swath_regrid():
    """Run command line interface for temporal aggregation of ASCAT data."""
    swath_regrid_main(sys.argv[1:])
