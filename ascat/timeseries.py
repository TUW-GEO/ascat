# Copyright (c) 2017, Vienna University of Technology, Department of Geodesy
# and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of
#      Geodesy and Geoinformation nor the names of its contributors may be
#      used to endorse or promote products derived from this software without
#      specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import glob
import warnings

import numpy as np
import netCDF4

from pygeobase.object_base import TS
import pygeogrids.grids as grids
import pygeogrids.netcdf as ncgrid

from pynetcf.time_series import GriddedNcContiguousRaggedTs
from pynetcf.time_series import GriddedNcOrthoMultiTs
from pynetcf.point_data import GriddedPointData


class AscatTimeSeries(TS):

    """
    Container class for ASCAT time series.

    Parameters
    ----------
    gpi : int
        Grid point index
    lon : float
        Longitude of grid point
    lat : float
        Latitude of grid point
    cell : int
        Cell number of grid point
    data : pandas.DataFrame
        DataFrame which contains the data
    topo_complex : int, optional
        Topographic complexity at the grid point
    wetland_frac : int, optional
        Wetland fraction at the grid point
    porosity_gldas : float, optional
        Porosity taken from GLDAS model
    porosity_hwsd : float, optional
        Porosity calculated from Harmonised World Soil Database

    Attributes
    ----------
    gpi : int
        Grid point index
    lon : float
        Longitude of grid point
    lat : float
        Latitude of grid point
    cell : int
        Cell number of grid point
    data : pandas.DataFrame
        DataFrame which contains the data
    topo_complex : int
        Topographic complexity at the grid point
    wetland_frac : int
        Wetland fraction at the grid point
    porosity_gldas : float
        Porosity taken from GLDAS model
    porosity_hwsd : float
        Porosity calculated from Harmonised World Soil Database
    """

    def __init__(self, gpi, lon, lat, cell, data,
                 topo_complex=None, wetland_frac=None,
                 porosity_gldas=None, porosity_hwsd=None):

        super(AscatTimeSeries, self).__init__(gpi, lon, lat, data, {})

        self.cell = cell
        self.topo_complex = topo_complex
        self.wetland_frac = wetland_frac
        self.porosity_gldas = porosity_gldas
        self.porosity_hwsd = porosity_hwsd

        # kept for backwards compatibility
        self.longitude = self.lon
        self.latitude = self.lat

    def __repr__(self):

        msg = "GPI: {:d} Lon: {:2.3f} Lat: {:3.3f}".format(self.gpi, self.lon,
                                                           self.lat)

        return msg


def load_grid(grid_filename):
    """
    Load grid file.

    Parameters
    ----------
    grid_filename : str
        Grid filename.

    Returns
    -------
    grid : pygeogrids.CellGrid
        Grid.
    """
    with netCDF4.Dataset(grid_filename) as grid_nc:
        land_gp = np.where(grid_nc.variables['land_flag'][:] == 1)[0]
        lons = grid_nc.variables['lon'][:]
        lats = grid_nc.variables['lat'][:]
        gpis = grid_nc.variables['gpi'][:]
        cells = grid_nc.variables['cell'][:]

    grid = grids.CellGrid(lons[land_gp], lats[land_gp], cells[land_gp],
                          gpis[land_gp])

    return grid


class StaticLayers(object):

    """
    Class to read static layer files.

    Parameters
    ----------
    path : str
        Path of static layer files.
    grid_filename : str
        Grid filename.

    Attributes
    ----------
    topo_complex : pynetcf.point_data.GriddedPointData
        Topographic complexity.
    wetland_frac : pynetcf.point_data.GriddedPointData
        Inundation and wetland fraction.
    frozen_prob : pynetcf.time_series.GriddedNcOrthoMultiTs
        Frozen soil/canopy probability.
    snow_prob : pynetcf.time_series.GriddedNcOrthoMultiTs
        Snow cover probability.
    porosity : pynetcf.time_series.GriddedNcOrthoMultiTs
        Soil porosity information.
    """

    def __init__(self, path, grid_filename):

        grid = load_grid(grid_filename)
        grid.arrcell[:] = 0

        fn_format = 'topographic_complexity.nc'
        self.topo_complex = GriddedPointData(path, grid, fn_format=fn_format)

        fn_format = 'inundation_and_wetlands.nc'
        self.wetland_frac = GriddedPointData(path, grid, fn_format=fn_format)

        fn_format = 'frozen_probability'
        self.frozen_prob = GriddedNcOrthoMultiTs(path, grid,
                                                 fn_format=fn_format)

        fn_format = 'snow_probability'
        self.snow_prob = GriddedNcOrthoMultiTs(path, grid,
                                               fn_format=fn_format)

        fn_format = 'porosity.nc'
        self.porosity = GriddedPointData(path, grid, fn_format=fn_format)


class AscatNc(GriddedNcContiguousRaggedTs):

    """
    Class reading Metop ASCAT soil moisture Climate Data Record (CDR).

    Parameters
    ----------
    path : str
        Path to Climate Data Record (CDR) data set.
    fn_format : str
        Filename format string, typical '<prefix>_{:04d}'
    grid_filename : str
        Grid filename.
    static_layer_path : str
        Path to static layer files.
    thresholds : dict, optional
        Thresholds for topographic complexity (default 50) and
        wetland fraction (default 50).

    Attributes
    ----------
    grid : pygeogrids.CellGrid
        Cell grid.
    thresholds : dict
        Thresholds for topographic complexity (default 50) and
        wetland fraction (default 50).
    slayer : str
        StaticLayer object
    """

    def __init__(self, path, fn_format, grid_filename, static_layer_path,
                 thresholds=None, **kwargs):

        grid = load_grid(grid_filename)

        self.thresholds = {'topo_complex': 5, 'wetland_frac': 5}

        if thresholds is not None:
            self.thresholds.update(thresholds)

        if static_layer_path is not None:
            self.slayer = StaticLayers(static_layer_path, grid_filename)
        else:
            self.slayer = None

        super(AscatNc, self).__init__(path, grid, fn_format=fn_format,
                                      **kwargs)

    def _read_gp(self, gpi, **kwargs):
        """
        Read time series for specific grid point.

        Parameters
        ----------
        gpi : int
            Grid point index.
        mask_ssf : boolean, optional
            Default False, if True only SSF values of 1 and 0 will be
            allowed, all others are removed
        mask_frozen_prob : int, optional
            If included in kwargs then all observations taken when
            frozen probability > mask_frozen_prob are removed from the data
            Default: no masking
        mask_snow_prob : int, optional
            If included in kwargs then all observations taken when
            snow probability > mask_snow_prob are removed from the data

        Returns
        -------
        ts : AscatTimeSeries
            Time series object.
        """
        absolute_sm = kwargs.pop('absolute_sm', None)
        mask_frozen_prob = kwargs.pop('mask_frozen_prob', None)
        mask_snow_prob = kwargs.pop('mask_snow_prob', None)
        mask_ssf = kwargs.pop('mask_ssf', None)

        data = super(AscatNc, self)._read_gp(gpi, **kwargs)
        lon, lat = self.grid.gpi2lonlat(gpi)
        cell = self.grid.gpi2cell(gpi)

        if self.slayer is not None:
            topo_complex = self.slayer.topo_complex.read(gpi)['topo'][0]
            wetland_frac = self.slayer.wetland_frac.read(gpi)['in_wet'][0]
            snow_prob = self.slayer.snow_prob.read(gpi)['snow_prob']
            frozen_prob = self.slayer.frozen_prob.read(gpi)['frozen_prob']
            porosity_gldas = self.slayer.porosity.read(gpi)['por_gldas'][0]
            porosity_hwsd = self.slayer.porosity.read(gpi)['por_hwsd'][0]

            data['snow_prob'] = snow_prob[data.index.dayofyear - 1].values
            data['frozen_prob'] = frozen_prob[data.index.dayofyear - 1].values
        else:
            topo_complex = np.nan
            wetland_frac = np.nan
            porosity_gldas = np.nan
            porosity_hwsd = np.nan
            data['snow_prob'] = np.nan
            data['frozen_prob'] = np.nan

        if absolute_sm:
            data['abs_sm_gldas'] = data['sm'] / 100.0 * porosity_gldas
            data['abs_sm_hwsd'] = data['sm'] / 100.0 * porosity_hwsd
        else:
            data['abs_sm_gldas'] = np.nan
            data['abs_sm_hwsd'] = np.nan

        if mask_ssf is not None:
            data = data[data['ssf'] < 2]

        if mask_frozen_prob is not None:
            data = data[data['frozen_prob'] < mask_frozen_prob]

        if mask_snow_prob is not None:
            data = data[data['snow_prob'] < mask_snow_prob]

        if (topo_complex is not None and
                topo_complex >= self.thresholds['topo_complex']):
            msg = "Topographic complexity >{:2d} ({:2d})".format(
                self.thresholds['topo_complex'], topo_complex)
            warnings.warn(msg)

        if (wetland_frac is not None and
                wetland_frac >= self.thresholds['wetland_frac']):
            msg = "Wetland fraction >{:2d} ({:2d})".format(
                self.thresholds['wetland_frac'], wetland_frac)
            warnings.warn(msg)

        ts = AscatTimeSeries(gpi, lon, lat, cell, data, topo_complex,
                             wetland_frac, porosity_gldas, porosity_hwsd)

        return ts


class AscatSsmCdr(AscatNc):

    """
    Class reading Metop ASCAT soil moisture Climate Data Record (CDR).

    Parameters
    ----------
    cdr_path : str
        Path to Climate Data Record (CDR) data set.
    grid_path : str
        Path to grid file.
    grid_filename : str
        Name of grid file.
    static_layer_path : str
        Path to static layer files.

    Attributes
    ----------
    grid : pygeogrids.CellGrid
        Cell grid.
    """

    def __init__(self, cdr_path, grid_path,
                 grid_filename='TUW_WARP5_grid_info_2_1.nc',
                 static_layer_path=None, **kwargs):

        first_file = glob.glob(os.path.join(cdr_path, '*.nc'))[0]
        version = os.path.basename(first_file).rsplit('_', 1)[0]
        fn_format = '{:}_{{:04d}}'.format(version)
        grid_filename = os.path.join(grid_path, grid_filename)

        super(AscatSsmCdr, self).__init__(cdr_path, fn_format, grid_filename,
                                          static_layer_path, **kwargs)


class AscatVODTs(GriddedNcContiguousRaggedTs):
    """
    Class that provides access to ASCAT VOD data stored in netCDF format.

    Parameters
    ----------
    path : string
        path to data folder which contains the zip files from the FTP server
    grid_path : string
        path to grid_info folder which contains a netcdf file with information about
        grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        name of the grid info netCDF file in grid_path
        default 'TUW_WARP5_grid_info_2_1.nc'

    Attributes
    ----------
    path : string
        path to data folder which contains the zip files from the FTP server
    grid_path : string
        path to grid_info folder which contains txt files with information about
        grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        name of the grid info netCDF file in grid_path
        default 'TUW_WARP5_grid_info_2_1.nc'
    grid : grids.CellGrid object
        CellGrid object, which provides nearest neighbor search and other features
    variables : list of string
    """

    def __init__(self, path, grid_path,
                 grid_info_filename='TUW_WARP5_grid_info_2_1.nc',
                 variables=None):

        grid = ncgrid.load_grid(os.path.join(grid_path, grid_info_filename),
                                subset_flag='land')

        self.path = path
        self.grid_path = grid_path
        self.grid_info_filename = grid_info_filename

        self.variables = variables

        if self.variables is None:
            self.variables = ['vod']

        super(AscatVODTs, self).__init__(path, grid)
