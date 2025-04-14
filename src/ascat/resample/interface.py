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

import numpy as np
import xarray as xr
from pyresample import kd_tree, SwathDefinition

from ascat.read_native.xarray_io import get_swath_product_id
from ascat.read_native.xarray_io import swath_io_catalog
from ascat.regrid.regrid import retrieve_or_store_grid_lut


def parse_args_swath_resample(args):
    """
    Parse command line arguments for resampling an ASCAT swath file
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
        description="Resample an ASCAT swath file to a regular grid")
    parser.add_argument(
        "filepath", metavar="FILEPATH", help="Path to file or folder")
    parser.add_argument(
        "outpath", metavar="OUTPATH", help="Path to the output data")
    parser.add_argument(
        "resample_deg",
        metavar="RESAMPLE_DEG",
        type=float,
        help="Target grid spacing in degrees")
    parser.add_argument(
        "--grid_store",
        metavar="GRID_STORE",
        help="Path for storing/loading lookup tables")
    parser.add_argument(
        "--suffix",
        metavar="SUFFIX",
        help="File suffix (default: _RESAMPLE_DEGdeg)")
    parser.add_argument(
        "--neighbours",
        metavar="N",
        type=int,
        default=6,
        help="Number of neighbours to consider for each grid point (default: 6)"
    )
    parser.add_argument(
        "--radius",
        metavar="RADIUS",
        type=float,
        default=10000,
        help="Cut off distance in meters (default: 10000)")

    return parser.parse_args(args)


def swath_resample_main(cli_args):
    """
    Resample an ASCAT swath file or directory of swath files
    to a regular grid and write the results to disk.

    Parameters
    ----------
    cli_args : list
        Command line arguments.
    """
    args = parse_args_swath_resample(cli_args)
    filepath = Path(args.filepath)
    trg_grid_size = args.resample_deg

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if args.suffix:
        suffix = args.suffix
    else:
        suffix = f"_{args.resample_deg}deg"

    if filepath.is_dir():
        files = list(filepath.glob("**/*.nc"))
    elif filepath.is_file() and filepath.suffix == ".nc":
        files = [filepath]
    else:
        raise RuntimeError("No files found at the provided filepath")

    radius = args.radius
    if (radius < 1000) or (radius > 100000):
        raise ValueError(f"Radius outside limits: 1000 < {radius} < 100000")

    k = args.neighbours
    if (k < 1) or (k > 100):
        raise ValueError(
            f"Number of neighbours outside limits 1 < {k} < {100}")

    first_file = files[0]

    product_id = get_swath_product_id(str(first_file.name))

    if product_id is None:
        raise RuntimeError("Product identifier unknown")

    product = swath_io_catalog[product_id]
    src_grid = product.grid
    src_grid_size = product.grid_sampling_km

    src_grid_id = f"fib_grid_{src_grid_size}km"
    trg_grid_id = f"reg_grid_{trg_grid_size}deg"

    trg_grid, grid_lut = retrieve_or_store_grid_lut(src_grid, src_grid_id,
                                                    trg_grid_id, trg_grid_size,
                                                    args.grid_store)

    var_list = [
        ("time", "nn"),
        ("as_des_pass", "nn"),
        ("swath_indicator", "nn"),
        ("surface_flag", "nn"),
        ("surface_flag_source", "nn"),
        ("surface_soil_moisture", "idw"),
        ("surface_soil_moisture_noise", "idw"),
        ("surface_soil_moisture_sensitivity", "idw"),
        ("backscatter40", "idw"),
        ("slope40", "idw"),
        ("curvature40", "idw"),
        ("snow_cover_probability", "idw"),
        ("frozen_soil_probability", "idw"),
        ("topographic_complexity", "idw"),
        ("wetland_fraction", "idw"),
        ("subsurface_scattering_prob", "idw"),
        # ("processing_flag", "major"),
        ("correction_flag", "bitwise_or"),
        ("backscatter40_flag", "bitwise_or"),
    ]

    # defintion of target grid - regular lat/lon grid
    target_def = SwathDefinition(lons=trg_grid.arrlon, lats=trg_grid.arrlat)
    output_shape = trg_grid.shape
    dim = ("latitude", "longitude")

    for f in files:
        outfile = outpath / Path(f.stem + suffix + f.suffix)

        with xr.open_dataset(f, decode_cf=False, mask_and_scale=False) as ds:

            lons = ds["longitude"] * ds["longitude"].scale_factor
            lats = ds["latitude"] * ds["latitude"].scale_factor

            coords = {
                "latitude": np.int32(trg_grid.lat2d[:, 0] / 1e-6),
                "longitude": np.int32(trg_grid.lon2d[0] / 1e-6)
            }

            resampled_ds = xr.Dataset(coords=coords)
            resampled_ds.attrs = ds.attrs

            resampled_ds["latitude"].attrs = ds["latitude"].attrs
            resampled_ds["longitude"].attrs = ds["longitude"].attrs

            swath_def = SwathDefinition(lons=lons, lats=lats)
            valid_input_index, valid_output_index, index_array, distance_array = \
                    kd_tree.get_neighbour_info(swath_def, target_def,
                                                radius, neighbours=k)

            invalid_pos = index_array == lons.size
            index_array = index_array.astype(np.int32)
            index_array[invalid_pos] = -1

            for var, method in var_list:

                if var not in ds:
                    continue

                data = ds[var].data[valid_input_index][index_array]
                data[invalid_pos] = ds[var]._FillValue

                resam_data = np.zeros(
                    data.shape[0], dtype=ds[var].dtype) + ds[var]._FillValue

                if method == "idw":
                    p = 2
                    weights = 1 / (distance_array**p)
                    invalid = invalid_pos | (data == ds[var]._FillValue)
                    weights[invalid] = 0
                    total_weights = weights.sum(axis=1)
                    idx = total_weights != 0
                    resam_data[idx] = np.sum(
                        weights * data, axis=1)[idx] / total_weights[idx]
                elif method == "nn":
                    valid = index_array != -1
                    first_idx = np.where(
                        valid.any(axis=1), valid.argmax(axis=1), -1)
                    valid_mask = first_idx != -1
                    valid_rows = np.arange(index_array.shape[0])[valid_mask]
                    resam_data[valid_rows] = data[valid_rows,
                                                  first_idx[valid_rows]]
                elif method == "bitwise_or":
                    mask = index_array == -1
                    data[mask] = 0
                    resam_data = np.bitwise_or.reduce(data, axis=1)
                else:
                    raise ValueError("Resampling method unknown")

                resampled_ds[var] = (dim, resam_data.reshape(output_shape))
                resampled_ds[var].attrs = ds[var].attrs
                resampled_ds[var].encoding = {"zlib": True, "complevel": 4}

            resampled_ds.to_netcdf(outfile)


def run_swath_resample():
    """Run command line interface for reample ASCAT swath data."""
    swath_resample_main(sys.argv[1:])
