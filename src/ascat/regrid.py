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

from fibgrid.realization import FibGrid
from pygeogrids.grids import genreg_grid

def fib_to_standard(fibgrid, outgrid_size):
    """
    Convert a Fibonacci grid to a standard grid.

    Parameters
    ----------
    lon_arr : numpy.ndarray
        1D array of longitudes in degrees.
    lat_arr : numpy.ndarray
        1D array of latitudes in degrees.
    fibgrid : fibgrid.realization.FibGrid
        Instance of FibGrid from which lon_arr and lat_arr were generated.
    outgrid_size : int
        Size of the output grid in degrees.

    Returns
    -------
    numpy.ndarray
        1D array of values on the standard grid.
    """
    reg_grid = genreg_grid(outgrid_size, outgrid_size)
    fib_to_reg_lut = fibgrid.calc_lut(reg_grid)
    # new_data_gpis = fib_to_reg_lut[fib_gpis]
    # out_lons, out_lats = reg_grid.gpi2lonlat(out_gpis)
    return reg_grid

def fib_to_standard_ds(ds, fibgrid, outgrid_size):
    """
    Convert a dataset from a Fibonacci grid to a standard grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with lon and lat dimensions.
    fibgrid : fibgrid.realization.FibGrid
        Instance of FibGrid from which lon_arr and lat_arr were generated.
    outgrid_size : int
        Size of the output grid in degrees.

    Returns
    -------
    xarray.Dataset
        Dataset with lon and lat dimensions.
    """

    new_grid = fib_to_standard(
        fibgrid,
        outgrid_size,
    )
    new_gpis = new_grid.gpis
    new_lons = new_grid.arrlon
    new_lats = new_grid.arrlat
    fibgrid_lut = new_grid.calc_lut(fibgrid)
    nearest_old_gpis = fibgrid_lut[new_gpis]
    ds = ds.reindex(location_id=nearest_old_gpis)

    # put the new gpi/lon/lat data onto the grouped_ds as well
    ds["location_id"] = ("location_id", new_gpis)
    ds["lon"] = ("location_id", new_lons)
    ds["lat"] = ("location_id", new_lats)

    # finally we turn lat and lon into their own dimensions by making a multiindex
    # out of them and then unstacking it.
    ds = ds.set_index(location_id=["lat", "lon"])
    ds = ds.unstack()

    return ds
