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

from pathlib import Path

import numpy as np

from pygeogrids.grids import genreg_grid
from pygeogrids.netcdf import save_grid, load_grid


# move this to pygeogrids

def grid_to_regular_grid(old_grid, new_grid_size):
    """ Create a regular grid of a given size and a lookup table from it to another grid.

    Parameters
    ----------
    old_grid : pygeogrids.grids.BasicGrid
        The grid to create a lookup table to.
    new_grid_size : int
        Size of the new grid in degrees.
    """
    new_grid = genreg_grid(new_grid_size, new_grid_size)
    old_grid_lut = new_grid.calc_lut(old_grid)
    new_grid_lut = old_grid.calc_lut(new_grid)
    return new_grid, old_grid_lut, new_grid_lut


def retrieve_or_store_grid_lut(
        store_path,
        old_grid,
        old_grid_id,
        new_grid_id,
        regrid_degrees
):
    """Get a grid and its lookup table from a store directory or create, store, and return them.

    Parameters
    ----------
    store_path : str
        Path to the store directory.
    old_grid : pygeogrids.BasicGrid
        The old grid.
    old_grid_id : str
        The old grid's id.
    new_grid_id : str
        The new grid's id.
    regrid_degrees : int
        The size of the new grid in degrees.
    """
    store_path = Path(store_path)
    old_lut_path = store_path / f"lut_{old_grid_id}_{new_grid_id}.npy"
    new_lut_path = store_path / f"lut_{new_grid_id}_{old_grid_id}.npy"
    grid_path = store_path / f"grid_{new_grid_id}.nc"
    if grid_path.exists() and cur_new_lut_path.exists() and new_cur_lut_path.exists():
        new_grid = load_grid(grid_path)
        old_grid_lut = np.load(old_lut_path, allow_pickle=True)
        new_grid_lut = np.load(new_lut_path, allow_pickle=True)

    else:
        new_grid, old_grid_lut, new_grid_lut = grid_to_regular_grid(old_grid,
                                                                    regrid_degrees)
        old_lut_path.parent.mkdir(parents=True, exist_ok=True)
        new_lut_path.parent.mkdir(parents=True, exist_ok=True)
        old_grid_lut.dump(old_lut_path)
        new_grid_lut.dump(new_lut_path)
        save_grid(grid_path, new_grid)

    return new_grid, old_grid_lut, new_grid_lut

def regrid_swath_ds(ds, new_grid, new_grid_lut):
    """Convert a swath dataset's location_ids to their nearest neighbors in a new grid."""
    new_gpis = new_grid_lut[ds["location_id"].values]
    new_lons = new_grid.arrlon[new_gpis]
    new_lats = new_grid.arrlat[new_gpis]
    ds["location_id"] = ("obs", new_gpis)
    ds["longitude"] = ("obs", new_lons)
    ds["latitude"] = ("obs", new_lats)
    return ds


def regrid_global_raster_ds(ds, new_grid, old_grid_lut):
    """Convert a global dataset from a Fibonacci grid to a standard grid.

    Assumes the input dataset has a unique location_id dimension.

    The output data will cover the entire globe and have lat and lon dimensions. The data
    for each point will be taken from the nearest location on the old grid. If multiple
    new points have the same nearest location, that data will be duplicated for each. If
    the nearest old location is NaN or does not exist in `ds`, the new point will be NaN.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a location_id variable derived from a pygeogrids.grids.BasicGrid.
    new_grid : pygeogrids.grids.BasicGrid
        Instance of BasicGrid that the dataset should be regridded to.
    ds_grid_lut : dict
        Lookup table from the new grid to the dataset's grid.

    Returns
    -------
    xarray.Dataset
        Dataset with lon and lat dimensions according to the new grid system.
    """
    new_gpis = new_grid.gpis
    new_lons = new_grid.arrlon
    new_lats = new_grid.arrlat
    nearest_ds_gpis = old_grid_lut[new_gpis]
    ds = ds.reindex(location_id=nearest_ds_gpis)

    # put the new gpi/lon/lat data onto the grouped_ds as well
    ds["location_id"] = ("location_id", new_gpis)
    ds["lon"] = ("location_id", new_lons)
    ds["lat"] = ("location_id", new_lats)

    # finally we turn lat and lon into their own dimensions by making a multiindex
    # out of them and then unstacking it.
    ds = ds.set_index(location_id=["lat", "lon"])
    ds = ds.unstack()

    return ds
