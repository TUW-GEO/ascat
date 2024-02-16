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

import ascat.aggregate.aggregators as aggs

aggs.progress_to_stdout = True

def parse_args_temporal_swath_agg(args):
    parser = argparse.ArgumentParser(
        description='Calculate aggregates of ASCAT swath data over a given time period'
    )
    parser.add_argument('filepath', metavar='FILEPATH', help='Path to the data')
    parser.add_argument('outpath', metavar='OUTPATH', help='Path to the output data')
    parser.add_argument(
        '-start_dt',
        metavar='START_DT',
        help='Start date and time (formatted e.g. 2020-01-01T00:00:00)'
    )
    parser.add_argument(
        '-end_dt',
        metavar='END_DT',
        help='End date and time (formatted e.g. 2020-02-01T00:00:00)'
    )
    parser.add_argument(
        '-t_delta',
        metavar='T_DELTA',
        help='Time period for aggregation (e.g. 1D, 1W, 1M, 1Y, 2D, 3M, 4Y, etc.)'
    )
    parser.add_argument(
        '-agg',
        metavar='AGG',
        help='Aggregation'
    )
    parser.add_argument(
        '-product',
        metavar='PROD',
        help='Product id'
    )
    parser.add_argument(
        '-snow_cover_mask',
        metavar='SNOW_MASK',
        help='Snow cover probability value above which to mask the source data'
    )
    parser.add_argument(
        '-frozen_soil_mask',
        metavar='FROZEN_MASK',
        help='Frozen soil probability value above which to mask the source data'
    )
    parser.add_argument(
        '-subsurface_scattering_mask',
        metavar='SUBSCAT_MASK',
        help='Subsurface scattering probability value above which to mask the source data'
    )

    return parser.parse_args(args), parser

def temporal_swath_agg_main(cli_args):
    """
    Command line interface routine for temporal aggregation of ASCAT data.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args, parser = parse_args_temporal_swath_agg(cli_args)
    int_args = ["snow_cover_mask", "frozen_soil_mask", "subsurface_scattering_mask"]
    for arg in int_args:
        if getattr(args, arg) is not None:
            setattr(args, arg, int(getattr(args, arg)))

    transf = aggs.TemporalSwathAggregator(
        args.filepath,
        args.outpath,
        args.start_dt,
        args.end_dt,
        args.t_delta,
        args.agg,
        args.product,
        args.snow_cover_mask,
        args.frozen_soil_mask,
        args.subsurface_scattering_mask
    )

    transf.write_time_chunks(args.outpath)

def run_temporal_swath_agg():
    """
    Run command line interface for temporal aggregation of ASCAT data.
    """
    temporal_swath_agg_main(sys.argv[1:])
