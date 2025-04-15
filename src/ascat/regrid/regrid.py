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

from pathlib import Path

import numpy as np
import xarray as xr

from pygeogrids.grids import genreg_grid
from pygeogrids.netcdf import load_grid, save_grid

from ascat.utils import dtype_to_nan


def retrieve_or_store_grid_lut(src_grid,
                               src_grid_id,
                               trg_grid_id,
                               trg_grid_size,
                               store_path=None):
    """
    Get a grid and its lookup table either from a store directory or
    create and return them.

    Parameters
    ----------
    src_grid : pygeogrids.BasicGrid
        Source grid.
    src_grid_id : str
        The source grid's id.
    trg_grid_id : str
        The target grid's id.
    trg_grid_size : int
        The size of the target grid in degrees.
    store_path : str, optional
        Path to the store directory (default: None).

    Returns
    -------
    trg_grid : pygeogrids.grids.BasicGrid
        Target grid.
    grid_lut : numpy.ndarray
        Look-up table.
    """
    if store_path == None:
        trg_grid = genreg_grid(trg_grid_size, trg_grid_size)
        grid_lut = trg_grid.calc_lut(src_grid).reshape(trg_grid.shape)
    else:
        store_path = Path(store_path)

        trg_grid_file = store_path / f"{trg_grid_id}.nc"

        if trg_grid_file.exists():
            trg_grid = load_grid(trg_grid_file)
        else:
            store_path.mkdir(parents=True, exist_ok=True)
            trg_grid = genreg_grid(trg_grid_size, trg_grid_size)
            save_grid(trg_grid_file, trg_grid)

        grid_lut_file = store_path / f"lut_{src_grid_id}_{trg_grid_id}.npy"

        if grid_lut_file.exists():
            grid_lut = np.load(grid_lut_file, allow_pickle=True)
        else:
            grid_lut = trg_grid.calc_lut(src_grid).reshape(trg_grid.shape)
            store_path.mkdir(parents=True, exist_ok=True)
            grid_lut.dump(grid_lut_file)

    return trg_grid, grid_lut


def regrid_swath_ds(ds, src_grid, trg_grid, grid_lut):
    """
    Convert a swath dataset to their nearest neighbors
    on a regular lat/lon grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Swath dataset.
    src_grid : pygeogrids.grids.BasicGrid
        Sourde grid.
    trg_grid : pygeogrids.grids.BasicGrid
        Target grid.
    trg_grid_lut : numpy.ndarray
        Grid look-up table.

    Returns
    -------
    ds : xarray.Dataset
        Swath dataset resampled on a regular lat/lon grid.
    """
    index_lut = np.zeros(src_grid.n_gpi, dtype=np.int32) - 1
    index_lut[ds["location_id"].data] = np.arange(ds["location_id"].size)
    idx = index_lut[grid_lut]
    nan_pos = idx == -1

    coords = {
        "latitude": np.int32(trg_grid.lat2d[:, 0] / 1e-6),
        "longitude": np.int32(trg_grid.lon2d[0] / 1e-6)
    }

    regrid_ds = xr.Dataset(coords=coords)
    regrid_ds.attrs = ds.attrs

    regrid_ds["latitude"].attrs = ds["latitude"].attrs
    regrid_ds["longitude"].attrs = ds["longitude"].attrs
    dim = ("latitude", "longitude")

    for var in ds.variables:
        if var in ["latitude", "longitude"]:
            continue

        if ds[var].size == 1:
            continue

        regrid_ds[var] = (dim, ds[var].data[idx])
        regrid_ds[var].attrs = ds[var].attrs
        regrid_ds[var].encoding = {"zlib": True, "complevel": 4}

        if hasattr(ds[var], "_FillValue"):
            regrid_ds[var].data[nan_pos] = ds[var]._FillValue
        else:
            if var == "time":
                regrid_ds[var].data[nan_pos] = 0
            else:
                regrid_ds[var].data[nan_pos] = dtype_to_nan[ds[var].dtype]

    return regrid_ds
