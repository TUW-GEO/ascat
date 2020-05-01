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


"""
Readers for lvl1b and lvl2 data in nc format.
"""

import os
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import pandas as pd

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.object_base import Image

ref_dt = np.datetime64('1970-01-01')
ref_jd = 2440587.5  # julian date on 1970-01-01 00:00:00


class AscatL1NcFile(ImageBase):
    """
    Read ASCAT L1 File in netCDF format, as downloaded from EUMETSAT

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    nc_variables: list, optional
        list of variables to read from netCDF.
        Default: read all available variables
    """

    def __init__(self, filename, mode='r', nc_variables=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL1NcFile, self).__init__(filename, mode=mode,
                                            **kwargs)
        self.nc_variables = nc_variables
        self.ds = None

    def read(self, timestamp=None):
        """
        reads from the netCDF file given by the filename

        Returns
        -------
        data : pygeobase.object_base.Image
        """

        if self.ds is None:
            self.ds = netCDF4.Dataset(self.filename)

        if self.nc_variables is None:
            var_to_read = self.ds.variables.keys()
        else:
            var_to_read = self.nc_variables

        # make sure that essential variables are read always:
        if 'latitude' not in var_to_read:
            var_to_read.append('latitude')
        if 'longitude' not in var_to_read:
            var_to_read.append('longitude')

        # store data in dictionary
        dd = {}
        metadata = {}
        beams = ['f_', 'm_', 'a_']

        metadata['sat_id'] = self.ds.platform[-1]
        metadata['orbit_start'] = self.ds.start_orbit_number
        metadata['processor_major_version'] = self.ds.processor_major_version
        metadata['product_minor_version'] = self.ds.product_minor_version
        metadata['format_major_version'] = self.ds.format_major_version
        metadata['format_minor_version'] = self.ds.format_minor_version

        num_cells = self.ds.dimensions['numCells'].size
        for name in var_to_read:
            variable = self.ds.variables[name]

            if len(variable.shape) == 1:
                # If the data is 1D then we repeat it for each cell
                dd[name] = variable[:].flatten()
                dd[name] = np.repeat(dd[name], num_cells)
            elif len(variable.shape) == 2:
                dd[name] = variable[:].flatten()
            elif len(variable.shape) == 3:
                # length of 3 means it is triplet data, so we split it
                for i, beam in enumerate(beams):
                    dd[beam + name] = variable[:, :, i].flatten()
                    if name == 'azi_angle_trip':
                        mask = dd[beam + name] < 0
                        dd[beam + name][mask] += 360
            else:
                raise RuntimeError("Unexpected variable shape.")

            if name == 'utc_line_nodes':
                utc_dates = netCDF4.num2date(
                    dd[name], variable.units).astype('datetime64[ns]')
                dd['jd'] = (utc_dates - ref_dt)/np.timedelta64(1, 'D') + ref_jd

        dd['as_des_pass'] = (dd['sat_track_azi'] < 270).astype(np.uint8)

        longitude = dd.pop('longitude')
        latitude = dd.pop('latitude')

        n_records = latitude.shape[0]
        n_lines = n_records // num_cells
        dd['node_num'] = np.tile((np.arange(num_cells) + 1), n_lines)
        dd['line_num'] = np.arange(n_lines).repeat(num_cells)

        return Image(longitude, latitude, dd, metadata, timestamp,
                     timekey='utc_line_nodes')

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class AscatL2SsmNcFile(ImageBase):
    """
    Read ASCAT L2 SSM File in netCDF format, as downloaded from EUMETSAT

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    nc_variables: list, optional
        list of variables to read from netCDF.
        Default: read all available variables
    """

    def __init__(self, filename, mode='r', nc_variables=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL2SsmNcFile, self).__init__(filename, mode=mode,
                                               **kwargs)
        self.nc_variables = nc_variables
        self.ds = None

    def read(self, timestamp=None, ssm_masked=False):
        """
        reads from the netCDF file given by the filename

        Returns
        -------
        data : pygeobase.object_base.Image
        """

        if self.ds is None:
            self.ds = netCDF4.Dataset(self.filename)

        if self.nc_variables is None:
            var_to_read = self.ds.variables.keys()
        else:
            var_to_read = self.nc_variables

        # make sure that essential variables are read always:
        if 'latitude' not in var_to_read:
            var_to_read.append('latitude')
        if 'longitude' not in var_to_read:
            var_to_read.append('longitude')

        # store data in dictionary
        dd = {}
        metadata = {}

        metadata['sat_id'] = self.ds.platform_long_name[-1]
        metadata['orbit_start'] = self.ds.start_orbit_number
        metadata['processor_major_version'] = self.ds.processor_major_version
        metadata['product_minor_version'] = self.ds.product_minor_version
        metadata['format_major_version'] = self.ds.format_major_version
        metadata['format_minor_version'] = self.ds.format_minor_version

        num_cells = self.ds.dimensions['numCells'].size
        for name in var_to_read:
            variable = self.ds.variables[name]
            dd[name] = variable[:].flatten()
            if len(variable.shape) == 1:
                # If the data is 1D then we repeat it for each cell
                dd[name] = np.repeat(dd[name], num_cells)

            if name == 'utc_line_nodes':
                utc_dates = netCDF4.num2date(
                    dd[name], variable.units).astype('datetime64[ns]')
                dd['jd'] = (utc_dates - ref_dt)/np.timedelta64(1, 'D') + ref_jd

        # if the ssm_masked is True we mask out data with missing ssm value
        if 'soil_moisture' in dd and ssm_masked is True:
            # mask all the arrays based on fill_value of latitude
            valid_data = ~dd['soil_moisture'].mask
            for name in dd:
                dd[name] = dd[name][valid_data]

        longitude = dd.pop('longitude')
        latitude = dd.pop('latitude')

        n_records = latitude.shape[0]
        n_lines = n_records // num_cells
        dd['node_num'] = np.tile((np.arange(num_cells) + 1), n_lines)
        dd['line_num'] = np.arange(n_lines).repeat(num_cells)

        dd['as_des_pass'] = (dd['sat_track_azi'] < 270).astype(np.uint8)

        return Image(longitude, latitude, dd, metadata, timestamp,
                     timekey='utc_line_nodes')

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class AscatL2SsmNc(MultiTemporalImageBase):
    """
    Class for reading HSAF ASCAT SSM images in netCDF format.
    The images have to be uncompressed in the following folder structure

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on
        the HSAF FTP Server then please specify the string that should be used
        in datetime.datetime.strftime Default: ''
    day_search_str: string, optional
        to provide an iterator over all images of a day the method
        _get_possible_timestamps looks for all available images on a day on the
        harddisk. This string is used in datetime.datetime.strftime and in
        glob.glob to search for all files on a day.
    file_search_str: string, optional
        this string is used in datetime.datetime.strftime and glob.glob to find
        a 3 minute bufr file by the exact date.
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
    nc_variables: list, optional
        list of variables to read from netCDF.
        Default: read all available variables
    """

    def __init__(self, path, month_path_str='',
                 day_search_str='W_XX-EUMETSAT-Darmstadt,'
                                'SURFACE+SATELLITE,METOPA+'
                                'ASCAT_C_EUMP_%Y%m%d*_125_ssm_l2.nc',
                 file_search_str='W_XX-EUMETSAT-Darmstadt,'
                                 'SURFACE+SATELLITE,METOPA+'
                                 'ASCAT_C_EUMP_{datetime}*_125_ssm_l2.nc',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(62, 76, '%Y%m%d%H%M%S'),
                 nc_variables=None):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.filename_datetime_format = filename_datetime_format
        super(AscatL2SsmNc, self).__init__(path, AscatL2SsmNcFile,
                                           subpath_templ=[month_path_str],
                                           fname_templ=file_search_str,
                                           datetime_format=datetime_format,
                                           exact_templ=False,
                                           ioclass_kws={
                                               'nc_variables': nc_variables})

    def _get_orbit_start_date(self, filename):
        orbit_start_str = \
            os.path.basename(filename)[self.filename_datetime_format[0]:
                                       self.filename_datetime_format[1]]
        return datetime.strptime(orbit_start_str,
                                 self.filename_datetime_format[2])

    def tstamps_for_daterange(self, startdate, enddate):
        """
        Get the timestamps as datetime array that are possible for the
        given day, if the timestamps are

        For this product it is not fixed but has to be looked up from
        the hard disk since bufr files are not regular spaced and only
        europe is in this product. For a global product a 3 minute
        spacing could be used as a fist approximation

        Parameters
        ----------
        startdate : datetime.date or datetime.datetime
            start date
        enddate : datetime.date or datetime.datetime
            end date

        Returns
        -------
        dates : list
            list of datetimes
        """
        file_list = []
        delta_all = enddate - startdate
        timestamps = []

        for i in range(delta_all.days + 1):
            timestamp = startdate + timedelta(days=i)

            files = self._search_files(
                timestamp, custom_templ=self.day_search_str)

            file_list.extend(sorted(files))

        for filename in file_list:
            timestamps.append(self._get_orbit_start_date(filename))

        timestamps = [dt for dt in timestamps if startdate <= dt <= enddate]
        return timestamps
