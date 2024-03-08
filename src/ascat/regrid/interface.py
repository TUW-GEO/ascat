# Copyright (c) 2024, TU Wien, Department of Geodesy and Geoinformation
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

from ascat.read_native.xarray_io import get_swath_product_id
from ascat.read_native.xarray_io import swath_io_catalog
from ascat.regrid.regrid import regrid_swath_ds
from ascat.regrid.regrid import retrieve_or_store_grid_lut
from ascat.regrid.regrid import grid_to_regular_grid


def parse_args_swath_regrid(args):
    """
    Parse command line arguments for regridding an ASCAT swath file to a regular grid.
    """
    parser = argparse.ArgumentParser(
        description='Regrid an ASCAT swath file to a regular grid'
    )
    parser.add_argument('filepath', metavar='FILEPATH', help='Path to the data')
    parser.add_argument('outpath', metavar='OUTPATH', help='Path to the output data')
    parser.add_argument(
        'regrid_deg',
        metavar='REGRID_DEG',
        help='Regrid the data to a regular grid with the given spacing in degrees'
    )
    parser.add_argument(
        '--grid_store',
        metavar='GRID_STORE',
        help='Path to a directory for storing grids and lookup tables between them'
    )
    return parser.parse_args(args)


def swath_regrid_main(cli_args):
    """
    Regrid an ASCAT swath file or directory of swath files to a regular grid and write
    the results to disk.
    """
    args = parse_args_swath_regrid(cli_args)
    filepath = Path(args.filepath)
    outpath = Path(args.outpath)
    new_grid_size = float(args.regrid_deg)
    new_grid = None

    if filepath.is_dir():
        files = filepath.glob("*.nc")
    else:
        files = [filepath]

    if files is None:
        raise ValueError("No .nc files found in the provided filepath.")

    first_file = files[0]
    product = swath_io_catalog[get_swath_product_id(str(first_file.name))]
    old_grid = product.grid
    old_grid_size = product.grid_sampling_km

    if args.grid_store:
        grid_store = args.grid_store
    else:
        grid_store = None

    for file in files:
        if new_grid is None:
            new_grid, _, new_grid_lut = grid_to_regular_grid(
                old_grid,
                new_grid_size
            )
        else:
            old_grid_id = f"fib_grid_{old_grid_size}km"
            new_grid_id = f"reg_grid_{args.regrid_deg}deg"
            new_grid, _, new_grid_lut = retrieve_or_store_grid_lut(
                grid_store,
                old_grid,
                old_grid_id,
                new_grid_id,
                new_grid_size
            )
        swath_ds = xr.open_dataset(file, decode_cf=False, mask_and_scale=False)
        regridded_ds = regrid_swath_ds(swath_ds, new_grid, new_grid_lut)
        regridded_ds.to_netcdf(outpath / file.name)


def run_swath_regrid():
    """ Run command line interface for temporal aggregation of ASCAT data. """
    swath_regrid_main(sys.argv[1:])
