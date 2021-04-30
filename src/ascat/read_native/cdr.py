# Copyright (c) 2021, TU Wien, Department of Geodesy and Geoinformation
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

import os
import warnings

import netCDF4
import numpy as np
import pygeogrids.grids as grids
from pynetcf.time_series import GriddedNcContiguousRaggedTs

float32_nan = -999999.0


class TimeSeries:

    """
    Container class for a time series.

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

        self.gpi = gpi
        self.lon = lon
        self.lat = lat
        self.data = data
        self.cell = cell
        self.topo_complex = topo_complex
        self.wetland_frac = wetland_frac
        self.porosity_gldas = porosity_gldas
        self.porosity_hwsd = porosity_hwsd

    def __repr__(self):
        msg = "GPI: {:d} Lon: {:2.3f} Lat: {:3.3f}".format(
            self.gpi, self.lon, self.lat)

        return msg


def load_grid(filename):
    """
    Load grid file.

    Parameters
    ----------
    filename : str
        Grid filename.

    Returns
    -------
    grid : pygeogrids.CellGrid
        Grid.
    """
    with netCDF4.Dataset(filename) as grid_nc:
        land_gp = np.where(grid_nc.variables['land_flag'][:] == 1)[0]
        lons = grid_nc.variables['lon'][:]
        lats = grid_nc.variables['lat'][:]
        gpis = grid_nc.variables['gpi'][:]
        cells = grid_nc.variables['cell'][:]

    grid = grids.CellGrid(lons[land_gp], lats[land_gp], cells[land_gp],
                          gpis[land_gp])

    return grid


class StaticLayers():

    """
    Class to read static layer files.

    Parameters
    ----------
    path : str
        Path of static layer files.
    topo_wetland_file : str, optional
        Topographic and complexity file (default: None).
    frozen_snow_file : str, optional
        Frozen and snow cover probability file (default: None).
    porosity_file : str, optional
        Porosity file (default: None).
    cache : bool, optional
        If true all static layers are loaded into memory (default: False).

    Attributes
    ----------
    topo_wetland : dict
        Topographic complexity and inundation and wetland fraction.
    frozen_snow_prob : dict
        Frozen soil/canopy probability and snow cover probability.
    porosity : dict
        Soil porosity information.
    """

    def __init__(self, path, topo_wetland_file=None,
                 frozen_snow_file=None, porosity_file=None, cache=False):

        if cache:
            print("Static layers will be loaded, this may take some time.")

        if topo_wetland_file is None:
            topo_wetland_file = os.path.join(path, 'topo_wetland.nc')

        self.topo_wetland = StaticFile(topo_wetland_file,
                                       ['wetland', 'topo'],
                                       cache=cache)

        if frozen_snow_file is None:
            frozen_snow_file = os.path.join(path, 'frozen_snow_probability.nc')

        self.frozen_snow_prob = StaticFile(frozen_snow_file,
                                           ['snow_prob', 'frozen_prob'],
                                           cache=cache)
        if porosity_file is None:
            porosity_file = os.path.join(path, 'porosity.nc')

        self.porosity = StaticFile(porosity_file,
                                   ['por_gldas', 'por_hwsd'],
                                   cache=cache)


class StaticFile:

    """
    StaticFile class.

    Parameters
    ----------
    filename : str
        File name.
    variables : list of str
        List of variables.
    cache : bool, optional
        Flag to cache data stored in file (default: False).

    Attributes
    ----------
    filename : str
        Static layer file name.
    variables : list of str
        List of variables.
    cache : bool
        Flag to cache data stored in file.
    data : dict
        Dictionary containing static layer data.
    """

    def __init__(self, filename, variables, cache=False):
        self.filename = filename
        self.cache = cache
        self.variables = variables
        self.data = {}

        if self.cache:
            with netCDF4.Dataset(self.filename) as nc_file:
                for v in self.variables:
                    self.data[v] = nc_file.variables[v][:].filled()

    def __getitem__(self, gpi):
        """
        Get data at given GPI.

        Parameters
        ----------
        gpi : int
            Grid point index.
        """
        data = {}
        if self.cache:
            for v in self.variables:
                data[v] = self.data[v][gpi]
        else:
            with netCDF4.Dataset(self.filename) as nc_file:
                for v in self.variables:
                    data[v] = nc_file.variables[v][[gpi]].filled()[0]

        return data


class AscatGriddedNcTs(GriddedNcContiguousRaggedTs):

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
    static_layer_path : str, optional
        Path to static layer files (default: None).
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

    def __init__(self, path, fn_format, grid_filename, static_layer_path=None,
                 cache_static_layer=False, thresholds=None, **kwargs):

        grid = load_grid(grid_filename)

        self.thresholds = {'topo_complex': 50, 'wetland_frac': 50}

        if thresholds is not None:
            self.thresholds.update(thresholds)

        self.slayer = None

        if static_layer_path is not None:
            self.slayer = StaticLayers(static_layer_path,
                                       cache=cache_static_layer)

        super().__init__(path, grid, fn_format=fn_format,
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

        data = super()._read_gp(gpi, **kwargs)
        data.attrs = {}
        data.attrs['gpi'] = gpi
        data.attrs['lon'], data.attrs['lat'] = self.grid.gpi2lonlat(gpi)
        data.attrs['cell'] = self.grid.gpi2cell(gpi)

        if self.slayer is not None:
            data.attrs['topo_complex'] = self.slayer.topo_wetland[gpi]['topo']
            data.attrs['wetland_frac'] = self.slayer.topo_wetland[gpi]['wetland']
            snow_prob = self.slayer.frozen_snow_prob[gpi]['snow_prob']
            frozen_prob = self.slayer.frozen_snow_prob[gpi]['frozen_prob']
            data.attrs['porosity_gldas'] = self.slayer.porosity[gpi]['por_gldas']
            data.attrs['porosity_hwsd'] = self.slayer.porosity[gpi]['por_hwsd']

            if data.attrs['porosity_gldas'] == float32_nan:
                data.attrs['porosity_gldas'] = np.nan

            if data.attrs['porosity_hwsd'] == float32_nan:
                data.attrs['porosity_hwsd'] = np.nan

            if data is not None:
                data['snow_prob'] = snow_prob[data.index.dayofyear - 1]
                data['frozen_prob'] = frozen_prob[data.index.dayofyear - 1]
        else:
            data.attrs['topo_complex'] = np.nan
            data.attrs['wetland_frac'] = np.nan
            data.attrs['porosity_gldas'] = np.nan
            data.attrs['porosity_hwsd'] = np.nan
            data['snow_prob'] = np.nan
            data['frozen_prob'] = np.nan

        if absolute_sm:
            # no error assumed for porosity values, i.e. variance = 0
            por_var = 0.

            data['abs_sm_gldas'] = data['sm'] / \
                100.0 * data.attrs['porosity_gldas']
            data['abs_sm_noise_gldas'] = np.sqrt(
                por_var * (data['sm'] / 100.0)**2 + data['sm_noise']**2 *
                (data.attrs['porosity_gldas'] / 100.0)**2)

            data['abs_sm_hwsd'] = data['sm'] / \
                100.0 * data.attrs['porosity_hwsd']
            data['abs_sm_noise_hwsd'] = np.sqrt(
                por_var * (data['sm'] / 100.0)**2 + data['sm_noise']**2 *
                (data.attrs['porosity_hwsd'] / 100.0)**2)
        else:
            data['abs_sm_gldas'] = np.nan
            data['abs_sm_noise_gldas'] = np.nan
            data['abs_sm_hwsd'] = np.nan
            data['abs_sm_noise_hwsd'] = np.nan

        if mask_ssf is not None:
            data = data[data['ssf'] < 2]

        if mask_frozen_prob is not None:
            data = data[data['frozen_prob'] < mask_frozen_prob]

        if mask_snow_prob is not None:
            data = data[data['snow_prob'] < mask_snow_prob]

        if (data.attrs['topo_complex'] is not None and
                data.attrs['topo_complex'] >= self.thresholds['topo_complex']):
            msg = "Topographic complexity >{:2d} ({:2d})".format(
                self.thresholds['topo_complex'], data.attrs['topo_complex'])
            warnings.warn(msg)

        if (data.attrs['wetland_frac'] is not None and
                data.attrs['wetland_frac'] >= self.thresholds['wetland_frac']):
            msg = "Wetland fraction >{:2d} ({:2d})".format(
                self.thresholds['wetland_frac'], data.attrs['wetland_frac'])
            warnings.warn(msg)

        return data
