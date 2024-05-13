# Copyright (c) 2024, TU Wien
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

aggs.progress_to_stdout = True


def parse_args_temporal_swath_agg(args):
    parser = argparse.ArgumentParser(
        description="Generate aggregates of ASCAT swath data over a given time period"
    )
    parser.add_argument(
        "filepath", metavar="FILEPATH", help="Path to the data")
    parser.add_argument(
        "outpath", metavar="OUTPATH", help="Path to the output data")
    parser.add_argument(
        "--start_dt",
        metavar="START_DT",
        help="Start date and time (formatted e.g. 2020-01-01T00:00:00)")
    parser.add_argument(
        "--end_dt",
        metavar="END_DT",
        help="End date and time (formatted e.g. 2020-02-01T00:00:00)")
    parser.add_argument(
        "--t_delta",
        metavar="T_DELTA",
        help="Time period for aggregation (e.g. 1D, 1W, 1M, 1Y, 2D, 3M, 4Y, etc.)"
    )
    parser.add_argument("--agg", metavar="AGG", help="Aggregation")
    parser.add_argument(
        "--snow_cover_mask",
        metavar="SNOW_MASK",
        type=int,
        default=80,
        help="Snow cover probability value above which to mask the source data"
    )
    parser.add_argument(
        "--frozen_soil_mask",
        metavar="FROZEN_MASK",
        type=int,
        default=80,
        help="Frozen soil probability value above which to mask the source data"
    )
    parser.add_argument(
        "--subsurface_scattering_mask",
        metavar="SUBSCAT_MASK",
        type=int,
        default=10,
        help="Subsurface scattering probability value above which to mask the source data"
    )
    parser.add_argument(
        "--regrid",
        metavar="REGRID_DEG",
        type=float,
        help="Regrid the data to a regular grid with the given spacing in degrees"
    )
    parser.add_argument(
        "--grid_store",
        metavar="GRID_STORE",
        help="Path to a directory for storing grids and lookup tables between them"
    )
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

    transf = aggs.TemporalSwathAggregator(args.filepath, args.start_dt,
                                          args.end_dt, args.t_delta, args.agg,
                                          args.snow_cover_mask,
                                          args.frozen_soil_mask,
                                          args.subsurface_scattering_mask)

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
            swath_regrid_main(regrid_args)


def run_temporal_swath_agg():
    """
    Run command line interface for temporal aggregation of ASCAT data.
    """
    temporal_swath_agg_main(sys.argv[1:])
