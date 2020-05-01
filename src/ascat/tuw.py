# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
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
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

import pygeogrids.grids as grids
import pygeogrids.netcdf as ncgrid
from pynetcf.time_series import GriddedNcContiguousRaggedTs

from ascat.read_native.cdr import AscatTimeSeries


class Ascat_data(object):

    """
    Class that provides access to ASCAT data stored in userformat which
    is available from the TU Wien FTP Server after registration at
    http://rs.geo.tuwien.ac.at.

    Parameters
    ----------
    path : string
        Path to data folder which contains the zip files from the FTP server
    grid_path : string
        Path to grid_info folder which contains txt files with information
        about grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        Name of the grid info txt file in grid_path
    advisory_flags_path : string, optional
        Path to advisory flags .dat files, if not provided they will
        not be used
    topo_threshold : int, optional
        If topographic complexity of read grid point is above this
        threshold a warning is output during reading
    wetland_threshold : int, optional
        If wetland fraction of read grid point is above this
        threshold a warning is output during reading

    Attributes
    ----------
    path : string
        Path to data folder which contains the zip files from the FTP server
    grid_path : string
        Path to grid_info folder which contains txt files with information
        about grid point index,latitude, longitude and cell
    grid_info_filename : string
        Name of the grid info txt file in grid_path
    grid_info_np_filename : string
        Name of the numpy save file to the grid information
    topo_threshold : int
        If topographic complexity of read grid point is above this
        threshold a warning is output during reading
    wetland_threshold : int
        If wetland fraction of read grid point is above this
        threshold a warning is output during reading
    grid_info_loaded : boolean
        True if the grid information has already been loaded
    grid : :class:`pygeogrids.grids.CellGrid` object
        CellGrid object, which provides nearest neighbor search and
        other features
    advisory_flags_path : string
        Path to advisory flags .dat files, if not provided they will
        not be used
    include_advflags : boolean
        True if advisory flags are available

    Methods
    -------
    unzip_cell(cell)
        unzips zipped grid point files into subdirectory
    read_advisory_flags(gpi)
        reads the advisory flags for a given grid point index
    """

    def __init__(self, path, grid_path,
                 grid_info_filename='TUW_W54_01_lonlat-ld-land.txt',
                 advisory_flags_path=None, topo_threshold=50,
                 wetland_threshold=50):
        """
        Set path and threshold
        """
        self.path = path
        self.grid_path = grid_path
        self.grid_info_filename = grid_info_filename
        self.grid_info_np_filename = 'TUW_W54_01_lonlat-ld-land.npy'
        self.topo_threshold = topo_threshold
        self.wetland_threshold = wetland_threshold
        self.grid_info_loaded = False
        self.grid = None
        self.advisory_flags_path = advisory_flags_path
        if self.advisory_flags_path is None:
            self.include_advflags = False
        else:
            self.include_advflags = True
            self.adv_flags_struct = np.dtype([('gpi', np.int32),
                                              ('snow', np.uint8, 366),
                                              ('frozen', np.uint8, 366),
                                              ('water', np.uint8),
                                              ('topo', np.uint8)])

    def _load_grid_info(self):
        """
        Reads the grid info for all land points from the txt file provided
        by TU Wien. The first time the actual txt file is parsed and saved
        as a numpy array to speed up future data access.
        """
        grid_info_np_filepath = os.path.join(
            self.grid_path, self.grid_info_np_filename)

        if os.path.exists(grid_info_np_filepath):
            grid_info = np.load(grid_info_np_filepath)

        else:
            grid_info_filepath = os.path.join(
                self.grid_path, self.grid_info_filename)
            grid_info = np.loadtxt(
                grid_info_filepath, delimiter=',', skiprows=1)
            np.save(os.path.join(self.grid_path,
                                 self.grid_info_np_filename), grid_info)

        self.grid = grids.CellGrid(grid_info[:, 2], grid_info[:, 1],
                                   grid_info[:, 3].astype(np.int16),
                                   gpis=grid_info[:, 0].astype(np.int32))

        self.grid_info_loaded = True

    def unzip_cell(self, cell):
        """
        Unzips the downloaded .zip cell file into the directory of
        os.path.join(self.path, cell).

        Parameters
        ----------
        cell : int
            cell number
        """
        filepath = os.path.join(self.path, '%4d.zip' % cell)
        unzip_file_path = os.path.join(self.path, '%4d' % cell)

        if not os.path.exists(unzip_file_path):
            os.mkdir(unzip_file_path)

        zfile = zipfile.ZipFile(filepath)
        for name in zfile.namelist():
            (dirname, filename) = os.path.split(name)
            fd = open(os.path.join(unzip_file_path, filename), "w")
            fd.write(zfile.read(name))
            fd.close()
        zfile.close()

    def _datetime_arr(self, longdate):
        """
        Parsing function that takes a number of type long which contains
        YYYYMMDDHH and returns a datetime object

        Parameters
        ----------
        longdate : long
            Date including hour as number of type long in format YYYYMMDDHH

        Returns
        -------
        datetime : datetime
        """
        string = str(longdate)
        year = int(string[0:4])
        month = int(string[4:6])
        day = int(string[6:8])
        hour = int(string[8:])

        return datetime(year, month, day, hour)

    def _read_ts(self, *args, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it
        """

        if not self.grid_info_loaded:
            self._load_grid_info()

        if len(args) == 1:
            return self._read_gp(args[0], **kwargs)
        if len(args) == 2:
            return self._read_lonlat(args[0], args[1], **kwargs)

    def _read_gp(self, gpi, **kwargs):
        """
        Reads the time series of the given grid point index. Masks frozen
        and snow observations if keywords are present

        Parameters
        ----------
        gpi : long
            grid point index
        mask_frozen_prob : int,optional
            if included in kwargs then all observations taken when
            frozen probability > mask_frozen_prob are removed from the result
        mask_snow_prob : int,optional
            if included in kwargs then all observations taken when
            snow probability > mask_snow_prob are removed from the result

        Returns
        -------
        df : pandas.DataFrame
            containing all fields in the list self.include_in_df
            plus frozen_prob and snow_prob if a path to advisory flags was
            set during initialization
        """
        cell = self.grid.gpi2cell(int(gpi))

        gp_file = os.path.join(
            self.path, '%4d' % cell, self.gp_filename_template % gpi)

        if not os.path.exists(gp_file):
            print('first time reading from cell %4d unzipping ...' % cell)
            self.unzip_cell(cell)

        data = np.fromfile(gp_file, dtype=self.gp_filestruct)
        dates = data['DAT']

        datetime_parser = np.vectorize(self._datetime_arr)

        datetimes_correct = datetime_parser(dates)

        dict_df = {}

        for into_df in self.include_in_df:
            d = np.ma.asarray(data[into_df], dtype=self.datatype[into_df])
            d = np.ma.masked_equal(d, self.nan_values[into_df])
            if into_df in self.scale_factor.keys():
                d = d * self.scale_factor[into_df]
            dict_df[into_df] = d

        df = pd.DataFrame(dict_df, index=datetimes_correct)

        if self.include_advflags:
            adv_flags, topo, wetland = self.read_advisory_flags(gpi)

            if topo >= self.topo_threshold:
                warnings.warn(
                    "Warning gpi shows topographic complexity of %d %%. Data might not be usable." % topo)
            if wetland >= self.wetland_threshold:
                warnings.warn(
                    "Warning gpi shows wetland fraction of %d %%. Data might not be usable." % wetland)

            df['doy'] = df.index.dayofyear
            df = df.join(adv_flags, on='doy', how='left')
            del df['doy']

            if 'mask_frozen_prob' in kwargs:
                mask_frozen = kwargs['mask_frozen_prob']
                df = df[df['frozen_prob'] <= mask_frozen]

            if 'mask_snow_prob' in kwargs:
                mask_snow = kwargs['mask_snow_prob']
                df = df[df['snow_prob'] <= mask_snow]

        lon, lat = self.grid.gpi2lonlat(gpi)

        return df, gpi, lon, lat, cell

    def _read_lonlat(self, lon, lat, **kwargs):
        """
        Get nearest grid point and read data.

        Parameters
        ----------
        lon : float
          Longitude.
        lat : float
          Latitude.

        Returns
        -------
        data : numpy.ndarray
          Read data.
        """
        gpi, dist = self.grid.find_nearest_gpi(lon, lat)
        return self._read_gp(int(gpi), **kwargs)

    def read_advisory_flags(self, gpi):
        """
        Read the advisory flags located in the self.advisory_flags_path
        Advisory flags include frozen probability, snow cover probability
        topographic complexity and wetland fraction.

        Parameters
        ----------
        gpi : long
            grid point index

        Returns
        -------
        df : pandas.DataFrame
            containing the columns frozen_prob and snow_prob. lenght 366
            with one entry for every day of the year, including February 29th
        topo : numpy.uint8
            topographic complexity ranging from 0-100
        wetland : numpy.uint8
            wetland fraction of pixel in percent
        """
        if not self.include_advflags:
            raise RuntimeError("Error: advisory_flags_path is not set")

        if not self.grid_info_loaded:
            self._load_grid_info()

        cell = self.grid.gpi2cell(gpi)
        adv_file = os.path.join(
            self.advisory_flags_path, '%d_advisory-flags.dat' % cell)
        data = np.fromfile(adv_file, dtype=self.adv_flags_struct)
        index = np.where(data['gpi'] == gpi)[0]

        data = data[index]

        snow = data['snow'][0]
        snow[snow == 0] += 101
        snow -= 101

        df = pd.DataFrame({'snow_prob': snow,
                           'frozen_prob': data['frozen'][0]})

        return df, data['topo'][0], data['water'][0]


class Ascat_SSM(Ascat_data):

    """
    Class for reading ASCAT SSM data. It extends Ascat_data and provides the
    information necessary for reading SSM data

    Parameters
    ----------
    path : string
        Path to data folder which contains the zip files from the FTP server
    grid_path : string
        Path to grid_info folder which contains txt files with information
        about grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        Name of the grid info txt file in grid_path
    advisory_flags_path : string, optional
        Path to advisory flags .dat files, if not provided they will not
        be used
    topo_threshold : int, optional
        If topographic complexity of read grid point is above this
        threshold a warning is output during reading
    wetland_threshold : int, optional
        If wetland fraction of read grid point is above this
        threshold a warning is output during reading

    Attributes
    ----------
    gp_filename_template : string
        Defines how the gpi is put into the template string to make the
        filename
    gp_filestruct : numpy.dtype
        Structure template of the SSM .dat file
    scale_factor : dict
        Factor by which to multiply the raw data to get the correct values
        for each field in the gp_filestruct
    include_in_df : list
        List of fields that should be returned to the user after reading
    nan_values : dict
        Nan value saved in the file which will be replaced by numpy.nan values
        during reading
    datatype : dict
        Datatype of the fields that the return data should have

    Methods
    -------
    read_ssm(*args,**kwargs)
        read surface soil moisture
    """

    def __init__(self, *args, **kwargs):
        super(Ascat_SSM, self).__init__(*args, **kwargs)
        self.gp_filename_template = 'TUW_ASCAT_SSM_W55_gp%d.dat'
        self.gp_filestruct = np.dtype([('DAT', np.int32),
                                       ('SSM', np.uint8),
                                       ('ERR', np.uint8),
                                       ('SSF', np.uint8)])

        self.scale_factor = {'SSM': 0.5, 'ERR': 0.5}
        self.include_in_df = ['SSM', 'ERR', 'SSF']
        self.nan_values = {'SSM': 255, 'ERR': 255, 'SSF': 255}
        self.datatype = {'SSM': np.float, 'ERR': np.float, 'SSF': np.int}

    def read_ssm(self, *args, **kwargs):
        """
        Function to read SSM takes either 1 or 2 arguments.
        It can be called as read_ssm(gpi, **kwargs) or
        read_ssm(lon, lat, **kwargs)

        Parameters
        ----------
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point
        mask_ssf : boolean, optional
            Default False, if True only SSF values of 1 will be allowed,
            all others are removed
        mask_frozen_prob : int,optional
            If included in kwargs then all observations taken when
            frozen probability > mask_frozen_prob are removed from the result
        mask_snow_prob : int,optional
            If included in kwargs then all observations taken when
            snow probability > mask_snow_prob are removed from the result

        Returns
        -------
        AscatTimeSeries : object
            :class:`ascat.AscatTimeSeries` instance
        """
        df, gpi, lon, lat, cell = super(
            Ascat_SSM, self)._read_ts(*args, **kwargs)
        if 'mask_ssf' in kwargs:
            mask_ssf = kwargs['mask_ssf']
            if mask_ssf:
                df = df[df['SSF'] == 1]

        return AscatTimeSeries(gpi, lon, lat, cell, df)


class Ascat_SWI(Ascat_data):

    """
    Class for reading ASCAT SWI data. It extends Ascat_data and provides the
    information necessary for reading SWI data

    Parameters
    ----------
    path : string
        Path to data folder which contains the zip files from the FTP server
    grid_path : string
        Path to grid_info folder which contains txt files with information
        about grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        Name of the grid info txt file in grid_path
    advisory_flags_path : string, optional
        Path to advisory flags .dat files, if not provided they will not
        be used
    topo_threshold : int, optional
        If topographic complexity of read grid point is above this
        threshold a warning is output during reading
    wetland_threshold : int, optional
        If wetland fraction of read grid point is above this
        threshold a warning is output during reading

    Attributes
    ----------
    gp_filename_template : string
        Defines how the gpi is put into the template string to make
        the filename
    gp_filestruct : numpy.dtype
        Structure template of the SSM .dat file
    scale_factor : dict
        Factor by which to multiply the raw data to get the correct values
        for each field in the gp_filestruct
    include_in_df : list
        List of fields that should be returned to the user after reading
    nan_values : dict
        Nan value saved in the file which will be replaced by numpy.nan values
        during reading
    datatype : dict
        Datatype of the fields that the return data should have
    T_SWI : dict
        Information about which numerical T-Value maps to which entry in the
        datastructure
    T_QFLAG : dict
        Information about which numerical T-Value maps to which entry in the
        datastructure

    Methods
    -------
    read_swi(*args,**kwargs)
        read soil water index
    """

    def __init__(self, *args, **kwargs):
        super(Ascat_SWI, self).__init__(*args, **kwargs)
        self.gp_filename_template = 'TUW_ASCAT_SWI_W55_gp%d.dat'
        self.gp_filestruct = np.dtype([('DAT', np.int32),
                                       ('SWI_T=1', np.uint8),
                                       ('SWI_T=5', np.uint8),
                                       ('SWI_T=10', np.uint8),
                                       ('SWI_T=15', np.uint8),
                                       ('SWI_T=20', np.uint8),
                                       ('SWI_T=40', np.uint8),
                                       ('SWI_T=60', np.uint8),
                                       ('SWI_T=100', np.uint8),
                                       ('QFLAG_T=1', np.uint8),
                                       ('QFLAG_T=5', np.uint8),
                                       ('QFLAG_T=10', np.uint8),
                                       ('QFLAG_T=15', np.uint8),
                                       ('QFLAG_T=20', np.uint8),
                                       ('QFLAG_T=40', np.uint8),
                                       ('QFLAG_T=60', np.uint8),
                                       ('QFLAG_T=100', np.uint8)])

        self.scale_factor = {'SWI_T=1': 0.5, 'SWI_T=5': 0.5,
                             'SWI_T=10': 0.5, 'SWI_T=15': 0.5,
                             'SWI_T=20': 0.5, 'SWI_T=40': 0.5,
                             'SWI_T=60': 0.5, 'SWI_T=100': 0.5,
                             'QFLAG_T=1': 0.5, 'QFLAG_T=5': 0.5,
                             'QFLAG_T=10': 0.5, 'QFLAG_T=15': 0.5,
                             'QFLAG_T=20': 0.5, 'QFLAG_T=40': 0.5,
                             'QFLAG_T=60': 0.5, 'QFLAG_T=100': 0.5}

        self.include_in_df = ['SWI_T=1', 'SWI_T=5', 'SWI_T=10', 'SWI_T=15',
                              'SWI_T=20', 'SWI_T=40', 'SWI_T=60', 'SWI_T=100',
                              'QFLAG_T=1', 'QFLAG_T=5', 'QFLAG_T=10',
                              'QFLAG_T=15', 'QFLAG_T=20', 'QFLAG_T=40',
                              'QFLAG_T=60', 'QFLAG_T=100']

        self.nan_values = {'SWI_T=1': 255, 'SWI_T=5': 255, 'SWI_T=10': 255,
                           'SWI_T=15': 255, 'SWI_T=20': 255, 'SWI_T=40': 255,
                           'SWI_T=60': 255, 'SWI_T=100': 255,
                           'QFLAG_T=1': 255, 'QFLAG_T=5': 255,
                           'QFLAG_T=10': 255, 'QFLAG_T=15': 255,
                           'QFLAG_T=20': 255, 'QFLAG_T=40': 255,
                           'QFLAG_T=60': 255, 'QFLAG_T=100': 255}

        self.datatype = {'SWI_T=1': np.float, 'SWI_T=5': np.float,
                         'SWI_T=10': np.float, 'SWI_T=15': np.float,
                         'SWI_T=20': np.float, 'SWI_T=40': np.float,
                         'SWI_T=60': np.float, 'SWI_T=100': np.float,
                         'QFLAG_T=1': np.float, 'QFLAG_T=5': np.float,
                         'QFLAG_T=10': np.float, 'QFLAG_T=15': np.float,
                         'QFLAG_T=20': np.float, 'QFLAG_T=40': np.float,
                         'QFLAG_T=60': np.float, 'QFLAG_T=100': np.float}

        self.T_SWI = {1: 'SWI_T=1', 5: 'SWI_T=5', 10: 'SWI_T=10',
                      15: 'SWI_T=15', 20: 'SWI_T=20', 40: 'SWI_T=40',
                      60: 'SWI_T=60', 100: 'SWI_T=100'}

        self.T_QFLAG = {1: 'QFLAG_T=1', 5: 'QFLAG_T=5', 10: 'QFLAG_T=10',
                        15: 'QFLAG_T=15', 20: 'QFLAG_T=20', 40: 'QFLAG_T=40',
                        60: 'QFLAG_T=60', 100: 'QFLAG_T=100'}

    def read_swi(self, *args, **kwargs):
        """
        Function to read SWI takes either 1 or 2 arguments being.
        It can be called as read_swi(gpi, **kwargs) or
        read_swi(lon, lat, **kwargs)

        Parameters
        ----------
        gpi : int
            grid point index
        lon : float
            longitude of point
        lat : float
            latitude of point
        T : int, optional
            if set only the SWI and QFLAG of this T-Value will be returned
        mask_qf : int, optional
            if set, SWI values with a QFLAG value lower than the mask_qf
            value will be masked. This is done for each T value independently
        mask_frozen_prob : int,optional
            if included in kwargs then all observations taken when
            frozen probability > mask_frozen_prob are removed from the result
        mask_snow_prob : int,optional
            if included in kwargs then all observations taken when
            snow probability > mask_snow_prob are removed from the result
        Returns
        -------
        df : pandas.DataFrame
            containing all fields in self.include_in_df plus frozen_prob
            and snow_prob if advisory_flags_path was set. If T was set then
            only SWI and QFLAG values for the selected T value are included
            plut frozen_prob and snow_prob if applicable
        """
        df, gpi, lon, lat, cell = super(Ascat_SWI, self)._read_ts(*args,
                                                                  **kwargs)

        if 'T' in kwargs:
            T = kwargs['T']
            if T in self.T_SWI.keys():
                if self.include_advflags:
                    df = df[[self.T_SWI[T], self.T_QFLAG[T],
                             'frozen_prob', 'snow_prob']]
                else:
                    df = df[[self.T_SWI[T], self.T_QFLAG[T]]]
            else:
                raise RuntimeError(
                    "Invalid T value. Choose one of " + str(sorted(self.T_SWI.keys())))

            # remove rows that have to small QFLAG
            if 'mask_qf' in kwargs:
                mask_qf = kwargs['mask_qf']
                if mask_qf:
                    df = df[df[self.T_QFLAG[T]] >= mask_qf]

        else:
            # mask each T value according to qf threshold
            if 'mask_qf' in kwargs:
                mask_qf = kwargs['mask_qf']
                for key in self.T_SWI:
                    masked = df[self.T_QFLAG[key]] <= mask_qf
                    df[self.T_SWI[key]][masked] = np.NAN
                    df[self.T_QFLAG[key]][masked] = np.NAN

        return AscatTimeSeries(gpi, lon, lat, cell, df)


class AscatVodTs(GriddedNcContiguousRaggedTs):

    """
    Class that provides access to ASCAT VOD data stored in netCDF format.

    Parameters
    ----------
    path : string
        Path to data folder which contains the zip files from the FTP server
    grid_path : string
        Path to grid_info folder which contains a netcdf file with
        information about grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        Name of the grid info netCDF file in grid_path
        default 'TUW_WARP5_grid_info_2_1.nc'

    Attributes
    ----------
    path : string
        Path to data folder which contains the zip files from the FTP server
    grid_path : string
        Path to grid_info folder which contains txt files with information
        about grid point index,latitude, longitude and cell
    grid_info_filename : string, optional
        Name of the grid info netCDF file in grid_path
        default 'TUW_WARP5_grid_info_2_1.nc'
    grid : grids.CellGrid object
        CellGrid object, which provides nearest neighbor search and
        other features
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

        super(AscatVodTs, self).__init__(path, grid)
