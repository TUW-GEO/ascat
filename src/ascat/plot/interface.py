# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

import sys
import argparse

from ascat.plot.plot import default_beams
from ascat.plot.plot import default_max_points
from ascat.plot.plot import plot_szf


def parse_args_plot_szf(args):
    """
    Parse command line arguments for plotting a full resolution backscatter
    file of ASCAT or SCA.

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
        description="Plot an ASCAT or SCA Level 1b SZF file on a map")
    parser.add_argument(
        "filepath", metavar="FILEPATH", help="Path to the file")
    parser.add_argument(
        "--parameter",
        metavar="PARAMETER",
        default="sig",
        help="Field to colour the measurements by (default: sig)")
    parser.add_argument(
        "--beams",
        metavar="BEAM",
        nargs="+",
        help=f"Beams to plot (default: {' '.join(default_beams)})")
    parser.add_argument(
        "--vmin", metavar="VMIN", type=float,
        help="Lower end of the colour scale")
    parser.add_argument(
        "--vmax", metavar="VMAX", type=float,
        help="Upper end of the colour scale")
    parser.add_argument(
        "--size", metavar="SIZE", type=float, default=1,
        help="Size of a measurement on the map (default: 1)")
    parser.add_argument(
        "--cmap", metavar="CMAP", default="viridis",
        help="Colour map (default: viridis)")
    parser.add_argument(
        "--title", metavar="TITLE",
        help="Title of the map (default: the file name)")
    parser.add_argument(
        "--max_points",
        metavar="MAX_POINTS",
        type=int,
        default=default_max_points,
        help=f"Measurements to draw at most, 0 for all of them "
             f"(default: {default_max_points})")
    parser.add_argument(
        "--outpath",
        metavar="OUTPATH",
        help="Write the map to this file instead of showing it")
    parser.add_argument(
        "--dpi", metavar="DPI", type=int, default=150,
        help="Resolution of the written map (default: 150)")

    return parser.parse_args(args)


def plot_szf_main(cli_args):
    """
    Plot a full resolution backscatter file of ASCAT or SCA.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args = parse_args_plot_szf(cli_args)

    m = plot_szf(
        args.filepath,
        parameter=args.parameter,
        beams=args.beams,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        max_points=args.max_points,
        size=args.size,
    )

    import matplotlib.pyplot as plt

    if args.outpath:
        m.f.savefig(args.outpath, dpi=args.dpi, bbox_inches="tight")
        print(f"Map written to {args.outpath}")
    else:
        plt.show()


def run_plot_szf():
    """
    Run command line interface for plotting a full resolution backscatter
    file of ASCAT or SCA.
    """
    plot_szf_main(sys.argv[1:])
