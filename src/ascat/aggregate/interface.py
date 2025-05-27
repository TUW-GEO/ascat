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

import ascat.aggregate.aggregators as aggs
from ascat.regrid.interface import swath_regrid_main
from ascat.resample.interface import swath_resample_main


def parse_args_temporal_swath_agg(args):
    parser = argparse.ArgumentParser(
        description=("Generate aggregates of ASCAT swath data over "
                     "a given time period"))
    parser.add_argument(
        "filepath", metavar="FILEPATH", help="Path to the data")
    parser.add_argument(
        "outpath", metavar="OUTPATH", help="Path to the output data")
    parser.add_argument(
        "--start_dt",
        metavar="START_DT",
        help="Start datetime (formatted e.g. 2020-01-01T00:00:00)")
    parser.add_argument(
        "--end_dt",
        metavar="END_DT",
        help="End datetime (formatted e.g. 2020-02-01T00:00:00)")
    parser.add_argument(
        "--t_delta",
        metavar="T_DELTA",
        help="Time period for aggregation (e.g. 1D, 1W, 1M, 1Y, 2D, etc.)")
    parser.add_argument("--agg", metavar="AGG", help="Aggregation")
    parser.add_argument(
        "--snow_cover_mask",
        metavar="SNOW_COVER_MASK",
        type=int,
        default=90,
        help=("Snow cover probability (0-100 %) value above which "
              "to mask the source data (default: 90 %)"))
    parser.add_argument(
        "--frozen_soil_mask",
        metavar="FROZEN_SOIL_MASK",
        type=int,
        default=90,
        help=("Frozen soil probability (0-100 %) value above which "
              "to mask the source data (default: 90 %)"))
    parser.add_argument(
        "--subsurface_scattering_mask",
        metavar="SUBSURFACE_SCATTERING_MASK",
        type=int,
        default=10,
        help=("Subsurface scattering probability (0-100 %) value above which "
              "to mask the source data (default: 10 %)"))
    parser.add_argument(
        "--ssm_sensitivity_mask",
        metavar="SSM_SENSITIVITY_MASK",
        type=float,
        default=1.0,
        help=("Surface soil moisture sensitivity (in dB) value below which "
              "to mask the source data (default: 1 dB)"))
    parser.add_argument(
        "--no_masking",
        action='store_const',
        const=True,
        default=False,
        help="Ignore all masks")
    parser.add_argument(
        "--regrid",
        metavar="REGRID_DEG",
        type=float,
        help=("Regrid the data to a regular grid with the given "
              " spacing in degrees"))
    parser.add_argument(
        "--resample",
        metavar="RESAMPLE_DEG",
        type=float,
        help=("Resample the data to a regular grid with the given "
              " spacing in degrees"))
    parser.add_argument(
        "--resample_neighbours",
        metavar="RESAMPLE_NEIGHBOURS",
        type=int,
        default=6,
        help="Number of neighbours to consider for each grid point when resampling (default: 6)")
    parser.add_argument(
        "--resample_radius",
        metavar="RESAMPLE_RADIUS",
        type=float,
        default=10000,
        help="Cut off distance in meters (default: 10000)")
    parser.add_argument(
        "--grid_store",
        metavar="GRID_STORE",
        help=("Path to a directory for storing grids and "
              "lookup tables between them"))
    parser.add_argument(
        "--suffix",
        metavar="SUFFIX",
        help="File suffix (default: _REGRID_DEGdeg)")

    return parser.parse_args(args)


def temporal_swath_agg_main(cli_args):
    """
    Command line interface routine for temporal aggregation of ASCAT data.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args = parse_args_temporal_swath_agg(cli_args)

    transf = aggs.TemporalSwathAggregator(
        args.filepath, args.start_dt, args.end_dt, args.t_delta, args.agg,
        args.snow_cover_mask, args.frozen_soil_mask,
        args.subsurface_scattering_mask, args.ssm_sensitivity_mask,
        args.no_masking)

    product_id = transf.product

    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    filenames = transf.write_time_steps(outpath)

    if args.regrid is not None:
        for filename in filenames:
            regrid_args = [str(filename), str(outpath), str(args.regrid)]
            if args.grid_store is not None:
                regrid_args.extend(["--grid_store", args.grid_store])
            if args.suffix is not None:
                regrid_args.extend(["--suffix", args.suffix])
            regrid_args.extend(["--product_id", product_id])
            swath_regrid_main(regrid_args)

    if args.resample is not None:
        for filename in filenames:
            resample_args = [str(filename), str(outpath), str(args.resample)]
            if args.grid_store is not None:
                resample_args.extend(["--grid_store", args.grid_store])
            if args.suffix is not None:
                resample_args.extend(["--suffix", args.suffix])
            resample_args.extend(["--neighbours", str(args.resample_neighbours)])
            resample_args.extend(["--radius", str(args.resample_radius)])
            resample_args.extend(["--product_id", product_id])
            swath_resample_main(resample_args)

def run_temporal_swath_agg():
    """
    Run command line interface for temporal aggregation of ASCAT data.
    """
    temporal_swath_agg_main(sys.argv[1:])
