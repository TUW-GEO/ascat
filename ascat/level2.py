# Copyright (c) 2017,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Level 2 data readers for ASCAT data in all formats. Not specific to distributor.
'''

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import netCDF4

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.object_base import Image

from ascat.bufr import BUFRReader


class AscatL2SsmBufrFile(ImageBase):
    """
    Reads ASCAT SSM swath files in BUFR format. There are the
    following products:

    - H101 SSM ASCAT-A NRT O 12.5 Metop-A ASCAT NRT SSM orbit geometry 12.5 km sampling
    - H102 SSM ASCAT-A NRT O 25.0 Metop-A ASCAT NRT SSM orbit geometry 25 km sampling
    - H16  SSM ASCAT-B NRT O 12.5 Metop-B ASCAT NRT SSM orbit geometry 12.5 km sampling
    - H103 SSM ASCAT-B NRT O 25.0 Metop-B ASCAT NRT SSM orbit geometry 25 km sampling
    - H104 SSM ASCAT-C NRT O 12.5 Metop-C ASCAT NRT SSM orbit geometry 12.5 km sampling
    - H105 SSM ASCAT-C NRT O 25.0 Metop-C ASCAT NRT SSM orbit geometry 25 km sampling

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name. See `ASCAT BUFR format table`_
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None,  **kwargs):
        """
        Initialization of i/o object.

        """
        self.filename = filename
        self.mode = mode
        self.kwargs = kwargs
        if msg_name_lookup is None:
            self.msg_name_lookup = {
                6: "Direction Of Motion Of Moving Observing Platform",
                16: "Orbit Number",
                65: "Surface Soil Moisture (Ms)",
                66: "Estimated Error In Surface Soil Moisture",
                67: "Backscatter",
                68: "Estimated Error In Sigma0 At 40 Deg Incidence Angle",
                69: "Slope At 40 Deg Incidence Angle",
                70: "Estimated Error In Slope At 40 Deg Incidence Angle",
                71: "Soil Moisture Sensitivity",
                72: "Dry Backscatter",
                73: "Wet Backscatter",
                74: "Mean Surface Soil Moisture",
                75: "Rain Fall Detection",
                76: "Soil Moisture Correction Flag",
                77: "Soil Moisture Processing Flag",
                78: "Soil Moisture Quality",
                79: "Snow Cover",
                80: "Frozen Land Surface Fraction",
                81: "Inundation And Wetland Fraction",
                82: "Topographic Complexity"}

    def read(self, timestamp=None):
        """
        Read specific image for given datetime timestamp.

        Parameters
        ----------
        timestamp : datetime.datetime
            exact observation timestamp of the image that should be read

        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold the metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        time_var : string or None
            variable name of observation times in the data dict, if None all
            observations have the same timestamp
        """
        # lookup table between names and message number in the BUFR file

        data = {}
        dates = []
        # 13: Latitude (High Accuracy)
        latitude = []
        # 14: Longitude (High Accuracy)
        longitude = []

        with BUFRReader(self.filename) as bufr:
            for message in bufr.messages():
                # read fixed fields
                latitude.append(message[:, 12])
                longitude.append(message[:, 13])
                years = message[:, 6].astype(int)
                months = message[:, 7].astype(int)
                days = message[:, 8].astype(int)
                hours = message[:, 9].astype(int)
                minutes = message[:, 10].astype(int)
                seconds = message[:, 11].astype(int)

                df = pd.to_datetime(pd.DataFrame({'month': months,
                                                  'year': years,
                                                  'day': days,
                                                  'hour': hours,
                                                  'minute': minutes,
                                                  'second': seconds}))
                dates.append(pd.DatetimeIndex(df).to_julian_date().values)

                # read optional data fields
                for mid in msg_name_lookup:
                    name = msg_name_lookup[mid]

                    if name not in data:
                        data[name] = []

                    data[name].append(message[:, mid - 1])

        dates = np.concatenate(dates)
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)

        for mid in msg_name_lookup:
            name = msg_name_lookup[mid]
            data[name] = np.concatenate(data[name])
            if mid == 74:
                # ssm mean is encoded differently
                data[name] = data[name] * 100

        data['jd'] = dates

        return Image(longitude, latitude, data, {}, timestamp, timekey='jd')

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class AscatL2SsmBufr(MultiTemporalImageBase):

    """
    Class for reading HSAF ASCAt SSM images in bufr format.
    The images have the same structure as the ASCAT 3 minute pdu files
    and these 2 readers could be merged in the future
    The images have to be uncompressed in the following folder structure
    path -
         month_path_str (default 'h07_%Y%m_buf')

    For example if path is set to /home/user/hsaf07 and month_path_str is left to the default 'h07_%Y%m_buf'
    then the images for March 2012 have to be in
    the folder /home/user/hsaf07/h07_201203_buf/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h07_%Y%m_buf'
    day_search_str: string, optional
        to provide an iterator over all images of a day the method _get_possible_timestamps
        looks for all available images on a day on the harddisk. This string is used in
        datetime.datetime.strftime and in glob.glob to search for all files on a day.
        Default : 'h07_%Y%m%d_*.buf'
    file_search_str: string, optional
        this string is used in datetime.datetime.strftime and glob.glob to find
        a 3 minute bufr file by the exact date.
        Default: 'h07_{datetime}*.buf'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d_%H%M%S
    """

    def __init__(self, path, month_path_str='h07_%Y%m_buf',
                 day_search_str='h07_%Y%m%d_*.buf',
                 file_search_str='h07_{datetime}*.buf',
                 datetime_format='%Y%m%d_%H%M%S',
                 filename_datetime_format=(4, 19, '%Y%m%d_%H%M%S')):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.filename_datetime_format = filename_datetime_format
        super(AscatL2SsmBufr, self).__init__(path, AscatL2SsmBufrFile, subpath_templ=[month_path_str],
                                             fname_templ=file_search_str,
                                             datetime_format=datetime_format,
                                             exact_templ=False)

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
        start_date : datetime.date or datetime.datetime
            start date
        end_date : datetime.date or datetime.datetime
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
        return timestamps


class AscatL2Ssm125NcFile(ImageBase):

    def read(self, timestamp=None):
        """
        reads from the netCDF file given by the filename

        Returns:
        --------
        data : pygeobase.object_base.Image
        """

        if self.ds is None:
            self.ds = netCDF4.Dataset(self.filename)

        # store data in dictionary
        dd = {}

        for name in ['latitude', 'longitude', 'soil_moisture',
                     'soil_moisture_sensitivity', 'frozen_soil_probability',
                     'snow_cover_probability', 'topography_flag',
                     'soil_moisture_error', 'mean_soil_moisture',
                     'sigma40', 'sigma40_error', 'wetland_flag',
                     'wet_backscatter', 'dry_backscatter',
                     'slope40', 'slope40_error',
                     'proc_flag1', 'corr_flags']:
            dd[name] = self.ds.variables[name][:].flatten()

        # dates are stored as UTC line nodes
        # this means that each row in the array has the same
        # time stamp
        utc_ln = self.ds.variables['utc_line_nodes']
        utc_dates = netCDF4.num2date(utc_ln[:], utc_ln.units)
        dates = netCDF4.netcdftime.JulianDayFromDate(utc_dates)
        # get the shape of the initial arrays in the netCDF
        orig_shape = self.ds.variables['latitude'].shape
        dd['dates'] = np.repeat(dates,
                                orig_shape[1]).reshape(orig_shape).flatten()
        # as_des_pass is stored as a 1D array in the netCDF
        # we convert it into 2D to have a boolean value for each observation
        dd['ascending pass'] = np.repeat(self.ds.variables['as_des_pass'][:],
                                         orig_shape[1]).reshape(orig_shape).flatten().astype(np.bool)

        # the variables below are not stored in the netCDF file so
        # they are filled with nan values
        dd['orbit_number'] = np.full_like(dd['latitude'], np.nan)
        dd['direction_of_motion'] = np.full_like(dd['latitude'], np.nan)

        # mask all the arrays based on fill_value of soil moisture
        valid_data = ~dd['soil_moisture'].mask
        for name in dd:
            dd[name] = dd[name][valid_data]

        data = {'ssm': dd['soil_moisture'],
                'ssm noise': dd['soil_moisture_error'],
                'topo complex': dd['topography_flag'],
                'ssm sensitivity': dd['soil_moisture_sensitivity'],
                'frozen prob': dd['frozen_soil_probability'],
                'snow prob': dd['snow_cover_probability'],
                'ssm mean': dd['mean_soil_moisture'],
                'sigma40': dd['sigma40'],
                'sigma40 noise': dd['sigma40_error'],
                'ascending pass': dd['ascending pass'],
                'wetland prob': dd['wetland_flag'],
                'wet reference': dd['wet_backscatter'],
                'dry reference': dd['dry_backscatter'],
                'slope40': dd['slope40'],
                'slope40 noise': dd['slope40_error'],
                'processing flag': dd['proc_flag1'],
                'correction flag': dd['corr_flags'],
                'jd': dd['dates']}

        return Image(dd['longitude'], dd['latitude'], data, {}, timestamp, timekey='jd')
