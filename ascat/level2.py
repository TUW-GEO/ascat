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
General Level 2 data readers for ASCAT data in all formats. Not specific to distributor.
'''

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import netCDF4

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.io_base import IntervalReadingMixin
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
    - EUMETSAT ASCAT Soil Moisture at 12.5 km Swath Grid - Metop in BUFR format
    - EUMETSAT ASCAT Soil Moisture at 25.0 km Swath Grid - Metop in BUFR format

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name. See :ref:`ascatformattable`.

        Default:

             === =====================================================
             Key Value
             === =====================================================
             6   'Direction Of Motion Of Moving Observing Platform',
             16  'Orbit Number',
             65  'Surface Soil Moisture (Ms)',
             66  'Estimated Error In Surface Soil Moisture',
             67  'Backscatter',
             68  'Estimated Error In Sigma0 At 40 Deg Incidence Angle',
             69  'Slope At 40 Deg Incidence Angle',
             70  'Estimated Error In Slope At 40 Deg Incidence Angle',
             71  'Soil Moisture Sensitivity',
             72  'Dry Backscatter',
             73  'Wet Backscatter',
             74  'Mean Surface Soil Moisture',
             75  'Rain Fall Detection',
             76  'Soil Moisture Correction Flag',
             77  'Soil Moisture Processing Flag',
             78  'Soil Moisture Quality',
             79  'Snow Cover',
             80  'Frozen Land Surface Fraction',
             81  'Inundation And Wetland Fraction',
             82  'Topographic Complexity'
             === =====================================================
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL2SsmBufrFile, self).__init__(filename, mode=mode,
                                                 **kwargs)
        if msg_name_lookup is None:
            msg_name_lookup = {
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
        self.msg_name_lookup = msg_name_lookup

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
                for mid in self.msg_name_lookup:
                    name = self.msg_name_lookup[mid]

                    if name not in data:
                        data[name] = []

                    data[name].append(message[:, mid - 1])

        dates = np.concatenate(dates)
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)

        for mid in self.msg_name_lookup:
            name = self.msg_name_lookup[mid]
            data[name] = np.concatenate(data[name])
            if mid == 74:
                # ssm mean is encoded differently
                data[name] = data[name] * 100

        data['jd'] = dates

        if 65 in self.msg_name_lookup:
            # mask all the arrays based on fill_value of soil moisture
            valid_data = np.where(data[self.msg_name_lookup[65]] != 1.7e+38)
            latitude = latitude[valid_data]
            longitude = longitude[valid_data]
            for name in data:
                data[name] = data[name][valid_data]

        return Image(longitude, latitude, data, {}, timestamp, timekey='jd')

    def read_masked_data(self, **kwargs):
        """
        It does not make sense to read a orbit file unmasked
        so we only have a masked implementation.
        """
        return self.read(**kwargs)

    def resample_data(self, image, index, distance, weights, **kwargs):
        """
        Takes an image and resample (interpolate) the image data to
        arbitrary defined locations given by index and distance.

        Parameters
        ----------
        image : object
            pygeobase.object_base.Image object
        index : np.array
            Index into image data defining a look-up table for data elements
            used in the interpolation process for each defined target
            location.
        distance : np.array
            Array representing the distances of the image data to the
            arbitrary defined locations.
        weights : np.array
            Array representing the weights of the image data that should be
            used during resampling.
            The weights of points not to use are set to np.nan
            This array is of shape (x, max_neighbors)

        Returns
        -------
        image : object
            pygeobase.object_base.Image object
        """
        total_weights = np.nansum(weights, axis=1)

        resOrbit = {}
        # resample backscatter
        for name in image.dtype.names:
            if name in ['Soil Moisture Correction Flag',
                        'Soil Moisture Processing Flag']:
                # The flags are resampled by taking the minimum flag This works
                # since any totally valid observation has the flag 0 and
                # overrides the flagged observations. This is true in cases
                # where the data was set to NaN by the flag as well as when the
                # data was set to 0 or 100. The last image element is the one
                # standing for NaN so we fill it with all flags filled to not
                # interfere with the minimum.
                image[name][-1] = 255
                bits = np.unpackbits(image[name].reshape(
                    (-1, 1)).astype(np.uint8), axis=1)
                resampled_bits = np.min(bits[index, :], axis=1)
                resOrbit[name] = np.packbits(resampled_bits)
            else:
                resOrbit[name] = np.nansum(
                    image[name][index] * weights, axis=1) / total_weights

        return resOrbit

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
    path - month_path_str (default 'h07_%Y%m_buf')

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
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name. See :ref:`ascatformattable`.

        Default:

             === =====================================================
             Key Value
             === =====================================================
             6   'Direction Of Motion Of Moving Observing Platform',
             16  'Orbit Number',
             65  'Surface Soil Moisture (Ms)',
             66  'Estimated Error In Surface Soil Moisture',
             67  'Backscatter',
             68  'Estimated Error In Sigma0 At 40 Deg Incidence Angle',
             69  'Slope At 40 Deg Incidence Angle',
             70  'Estimated Error In Slope At 40 Deg Incidence Angle',
             71  'Soil Moisture Sensitivity',
             72  'Dry Backscatter',
             73  'Wet Backscatter',
             74  'Mean Surface Soil Moisture',
             75  'Rain Fall Detection',
             76  'Soil Moisture Correction Flag',
             77  'Soil Moisture Processing Flag',
             78  'Soil Moisture Quality',
             79  'Snow Cover',
             80  'Frozen Land Surface Fraction',
             81  'Inundation And Wetland Fraction',
             82  'Topographic Complexity'
             === =====================================================
    """

    def __init__(self, path, month_path_str='h07_%Y%m_buf',
                 day_search_str='h07_%Y%m%d_*.buf',
                 file_search_str='h07_{datetime}*.buf',
                 datetime_format='%Y%m%d_%H%M%S',
                 filename_datetime_format=(4, 19, '%Y%m%d_%H%M%S'),
                 msg_name_lookup=None):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.filename_datetime_format = filename_datetime_format
        super(AscatL2SsmBufr, self).__init__(path, AscatL2SsmBufrFile, subpath_templ=[month_path_str],
                                             fname_templ=file_search_str,
                                             datetime_format=datetime_format,
                                             exact_templ=False,
                                             ioclass_kws={'msg_name_lookup': msg_name_lookup})

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

        timestamps = [dt for dt in timestamps if (
            dt >= startdate and dt <= enddate)]
        return timestamps


class AscatL2SsmBufrChunked(IntervalReadingMixin, AscatL2SsmBufr):
    """
    Reads BUFR files but does not return them on a file by file basis but in
    bigger chunks. For example it allows to read multiple 3 minute PDU's in
    half orbit chunks of 50 minutes. This speeds up operations like e.g.
    resampling of the data.

    Parameters
    ----------
    chunk_minutes: int, optional
        How many minutes should a chunk of data cover.
    """

    def __init__(self, path, month_path_str='h07_%Y%m_buf',
                 day_search_str='h07_%Y%m%d_*.buf',
                 file_search_str='h07_{datetime}*.buf',
                 datetime_format='%Y%m%d_%H%M%S',
                 filename_datetime_format=(4, 19, '%Y%m%d_%H%M%S'),
                 msg_name_lookup=None, chunk_minutes=50):

        super(AscatL2SsmBufrChunked, self).__init__(
            path,
            month_path_str=month_path_str,
            day_search_str=day_search_str,
            file_search_str=file_search_str,
            datetime_format=datetime_format,
            filename_datetime_format=filename_datetime_format,
            msg_name_lookup=msg_name_lookup,
            chunk_minutes=chunk_minutes)


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

        num_cells = self.ds.dimensions['numCells'].size
        for name in var_to_read:
            variable = self.ds.variables[name]
            dd[name] = variable[:].flatten()
            if len(variable.shape) == 1:
                # If the data is 1D then we repeat it for each cell
                dd[name] = np.repeat(dd[name], num_cells)

            if name == 'utc_line_nodes':
                utc_dates = netCDF4.num2date(dd[name], variable.units)
                dd['jd'] = netCDF4.netcdftime.JulianDayFromDate(utc_dates)

        if 'soil_moisture' in dd:
            # mask all the arrays based on fill_value of latitude
            valid_data = ~dd['soil_moisture'].mask
            for name in dd:
                dd[name] = dd[name][valid_data]

        longitude = dd.pop('longitude')
        latitude = dd.pop('latitude')

        return Image(longitude, latitude, dd, {}, timestamp, timekey='utc_line_nodes')

    def read_masked_data(self, **kwargs):
        """
        It does not make sense to read a orbit file unmasked
        so we only have a masked implementation.
        """
        return self.read(**kwargs)

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
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: ''
    day_search_str: string, optional
        to provide an iterator over all images of a day the method _get_possible_timestamps
        looks for all available images on a day on the harddisk. This string is used in
        datetime.datetime.strftime and in glob.glob to search for all files on a day.
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
                 day_search_str='W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_%Y%m%d*_125_ssm_l2.nc',
                 file_search_str='W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_{datetime}*_125_ssm_l2.nc',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(62, 76, '%Y%m%d%H%M%S'),
                 nc_variables=None):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.filename_datetime_format = filename_datetime_format
        super(AscatL2SsmNc, self).__init__(path, AscatL2SsmNcFile, subpath_templ=[month_path_str],
                                           fname_templ=file_search_str,
                                           datetime_format=datetime_format,
                                           exact_templ=False,
                                           ioclass_kws={'nc_variables': nc_variables})

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

        timestamps = [dt for dt in timestamps if dt >=
                      startdate and dt <= enddate]
        return timestamps
