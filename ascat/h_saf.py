# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of
#      Geodesy and Geoinformation nor the names of its contributors may be
#      used to endorse or promote products derived from this software without
#      specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on May 21, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import os
from datetime import datetime, timedelta
import numpy as np
import warnings

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.object_base import Image

import ascat.bufr as bufr_reader
from ascat.level2 import AscatL2SsmBufr
try:
    import pygrib
except ImportError:
    warnings.warn('pygrib can not be imported H14 images can not be read.')

import sys
if sys.version_info < (3, 0):
    range = xrange


class H08Single(ImageBase):

    def read(self, timestamp=None, lat_lon_bbox=None):
        """
        Read specific image for given datetime timestamp.

        Parameters
        ----------
        filename : string
            filename
        timestamp : datetime.datetime
            exact observation timestamp of the image that should be read
        lat_lon_bbox : list, optional
            list of lat,lon cooridnates of bounding box
            [lat_min, lat_max, lon_min, lon_max]

        Returns
        -------
        data : dict or None
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset, if no data was found None is returned
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

        with bufr_reader.BUFRReader(self.filename) as bufr:
            lons = []
            ssm = []
            ssm_noise = []
            ssm_corr_flag = []
            ssm_proc_flag = []
            data_in_bbox = True
            for i, message in enumerate(bufr.messages()):
                if i == 0:
                    # first message is just lat, lon extent
                    # check if any data in bbox
                    if lat_lon_bbox is not None:
                        lon_min, lon_max = message[0, 2], message[0, 3]
                        lat_min, lat_max = message[0, 4], message[0, 5]
                        if (lat_lon_bbox[0] > lat_max or lat_lon_bbox[1] < lat_min or
                                lat_lon_bbox[2] > lon_max or lat_lon_bbox[3] < lon_min):
                            data_in_bbox = False
                            break
                    # print 'columns', math.ceil((message[:, 3] - message[:, 2]) / 0.00416667)
                    # print 'rows', math.ceil((message[:, 5] - message[:, 4]) /
                    # 0.00416667)
                elif data_in_bbox:
                    # first 5 elements are there only once, after that, 4 elements are repeated
                    # till the end of the array these 4 are ssm, ssm_noise, ssm_corr_flag and
                    # ssm_proc_flag
                    # each message contains the values for 120 lons between lat_min and lat_max
                    # the grid spacing is 0.00416667 degrees
                    lons.append(message[:, 0])
                    lat_min = message[0, 1]
                    lat_max = message[0, 2]
                    ssm.append(message[:, 4::4])
                    ssm_noise.append(message[:, 5::4])
                    ssm_corr_flag.append(message[:, 6::4])
                    ssm_proc_flag.append(message[:, 7::4])

        if data_in_bbox:
            ssm = np.rot90(np.vstack(ssm)).astype(np.float32)
            ssm_noise = np.rot90(np.vstack(ssm_noise)).astype(np.float32)
            ssm_corr_flag = np.rot90(
                np.vstack(ssm_corr_flag)).astype(np.float32)
            ssm_proc_flag = np.rot90(
                np.vstack(ssm_proc_flag)).astype(np.float32)
            lats_dim = np.linspace(lat_max, lat_min, ssm.shape[0])
            lons_dim = np.concatenate(lons)

            data = {'ssm': ssm,
                    'ssm_noise': ssm_noise,
                    'proc_flag': ssm_proc_flag,
                    'corr_flag': ssm_corr_flag
                    }

            # if there are is a gap in the image it is not a 2D array in lon, lat space
            # but has a jump in latitude or longitude
            # detect a jump in lon or lat spacing
            lon_jump_ind = np.where(np.diff(lons_dim) > 0.00418)[0]
            if lon_jump_ind.size > 1:
                print("More than one jump in longitude")
            if lon_jump_ind.size == 1:
                diff_lon_jump = np.abs(
                    lons_dim[lon_jump_ind] - lons_dim[lon_jump_ind + 1])
                missing_elements = np.round(diff_lon_jump / 0.00416666)
                missing_lons = np.linspace(lons_dim[lon_jump_ind],
                                           lons_dim[
                                               lon_jump_ind + 1], missing_elements,
                                           endpoint=False)

                # fill up longitude dimension to full grid
                lons_dim = np.concatenate(
                    [lons_dim[:lon_jump_ind], missing_lons, lons_dim[lon_jump_ind + 1:]])
                # fill data with NaN values
                empty = np.empty((lats_dim.shape[0], missing_elements))
                empty.fill(1e38)
                for key in data:
                    data[key] = np.concatenate(
                        [data[key][:, :lon_jump_ind], empty, data[key][:, lon_jump_ind + 1:]], axis=1)

            lat_jump_ind = np.where(np.diff(lats_dim) > 0.00418)[0]
            if lat_jump_ind.size > 1:
                print("More than one jump in latitude")
            if lat_jump_ind.size == 1:
                diff_lat_jump = np.abs(
                    lats_dim[lat_jump_ind] - lats_dim[lat_jump_ind + 1])
                missing_elements = np.round(diff_lat_jump / 0.00416666)
                missing_lats = np.linspace(lats_dim[lat_jump_ind],
                                           lats_dim[
                                               lat_jump_ind + 1], missing_elements,
                                           endpoint=False)

                # fill up longitude dimension to full grid
                lats_dim = np.concatenate(
                    [lats_dim[:lat_jump_ind], missing_lats, lats_dim[lat_jump_ind + 1:]])
                # fill data with NaN values
                empty = np.empty((missing_elements, lons_dim.shape[0]))
                empty.fill(1e38)
                for key in data:
                    data[key] = np.concatenate(
                        [data[key][:lat_jump_ind, :], empty, data[key][lat_jump_ind + 1:, :]], axis=0)

            lons, lats = np.meshgrid(lons_dim, lats_dim)
            # only return data in bbox
            if lat_lon_bbox is not None:
                data_ind = np.where((lats >= lat_lon_bbox[0]) &
                                    (lats <= lat_lon_bbox[1]) &
                                    (lons >= lat_lon_bbox[2]) &
                                    (lons <= lat_lon_bbox[3]))
                # indexing returns 1d array
                # get shape of lats_dim and lons_dim to be able to reshape
                # the 1d arrays to the correct 2d shapes
                lats_dim_shape = np.where((lats_dim >= lat_lon_bbox[0]) &
                                          (lats_dim <= lat_lon_bbox[1]))[0].shape[0]
                lons_dim_shape = np.where((lons_dim >= lat_lon_bbox[2]) &
                                          (lons_dim <= lat_lon_bbox[3]))[0].shape[0]

                lons = lons[data_ind].reshape(lats_dim_shape, lons_dim_shape)
                lats = lats[data_ind].reshape(lats_dim_shape, lons_dim_shape)
                for key in data:
                    data[key] = data[key][data_ind].reshape(
                        lats_dim_shape, lons_dim_shape)

            return Image(lons, lats, data, {}, timestamp)

        else:
            return Image(None, None, None, {}, timestamp)

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class H08img(MultiTemporalImageBase):

    """
    Reads HSAF H08 images. The images have to be uncompressed in the following folder structure
    path - month_path_str (default 'h08_%Y%m_buf')

    For example if path is set to /home/user/hsaf08 and month_path_str is left to the default 'h08_%Y%m_buf'
    then the images for March 2012 have to be in
    the folder /home/user/hsaf08/h08_201203_buf/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h08_%Y%m_buf'
    day_search_str: string, optional
        to provide an iterator over all images of a day the method _get_possible_timestamps
        looks for all available images on a day on the harddisk. This string is used in
        datetime.datetime.strftime and in glob.glob to search for all files on a day.
        Default : 'h08_%Y%m%d_*.buf'
    file_search_str: string, optional
        this string is used in datetime.datetime.strftime and glob.glob to find
        a 3 minute bufr file by the exact date.
        Default: 'h08_{datetime}*.buf'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d_%H%M%S
    """

    def __init__(self, path, month_path_str='h08_%Y%m_buf',
                 day_search_str='h08_%Y%m%d_*.buf',
                 file_search_str='h08_{datetime}*.buf',
                 datetime_format='%Y%m%d_%H%M%S',
                 filename_datetime_format=(4, 19, '%Y%m%d_%H%M%S')):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.filename_datetime_format = filename_datetime_format
        super(H08img, self).__init__(path, H08Single, subpath_templ=[month_path_str],
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

        timestamps = [dt for dt in timestamps if dt >= startdate and dt <= enddate]
        return timestamps


class H07img(AscatL2SsmBufr):
    pass


class H16img(AscatL2SsmBufr):
    """
    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h16_%Y%m_buf'
    """

    def __init__(self, path, month_path_str='h16_%Y%m_buf'):
        day_search_str = 'h16_%Y%m%d_*.buf'
        file_search_str = 'h16_{datetime}*.buf'
        super(H16img, self).__init__(path, month_path_str=month_path_str,
                                     day_search_str=day_search_str,
                                     file_search_str=file_search_str)


class H101img(AscatL2SsmBufr):
    """
    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h101_%Y%m_buf'
    """

    def __init__(self, path, month_path_str='h101_%Y%m_buf'):
        day_search_str = 'h101_%Y%m%d_*.buf'
        file_search_str = 'h101_{datetime}*.buf'
        filename_datetime_format = (5, 20, '%Y%m%d_%H%M%S')
        super(H101img, self).__init__(path, month_path_str=month_path_str,
                                      day_search_str=day_search_str,
                                      file_search_str=file_search_str,
                                      filename_datetime_format=filename_datetime_format)


class H102img(AscatL2SsmBufr):
    """
    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h102_%Y%m_buf'
    """

    def __init__(self, path, month_path_str='h102_%Y%m_buf'):
        day_search_str = 'h102_%Y%m%d_*.buf'
        file_search_str = 'h102_{datetime}*.buf'
        filename_datetime_format = (5, 20, '%Y%m%d_%H%M%S')
        super(H102img, self).__init__(path, month_path_str=month_path_str,
                                      day_search_str=day_search_str,
                                      file_search_str=file_search_str,
                                      filename_datetime_format=filename_datetime_format)


class H103img(AscatL2SsmBufr):
    """
    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h103_%Y%m_buf'
    """

    def __init__(self, path, month_path_str='h103_%Y%m_buf'):
        day_search_str = 'h103_%Y%m%d_*.buf'
        file_search_str = 'h103_{datetime}*.buf'
        filename_datetime_format = (5, 20, '%Y%m%d_%H%M%S')
        super(H103img, self).__init__(path, month_path_str=month_path_str,
                                      day_search_str=day_search_str,
                                      file_search_str=file_search_str,
                                      filename_datetime_format=filename_datetime_format)


class H14Single(ImageBase):
    """

    Parameters
    ----------
    expand_grid : boolean, optional
        if set the images will be expanded to a 2D image during reading
        if false the images will be returned as 1D arrays on the
        reduced gaussian grid
        Default: True
    metadata_fields: list, optional
        fields of the message to put into the metadata dictionary.
    """

    def __init__(self, filename, mode='r', expand_grid=True,
                 metadata_fields=['units',
                                  'name']):
        self.expand_grid = expand_grid
        self.metadata_fields = metadata_fields
        self.pygrib1 = True
        if int(pygrib.__version__[0]) > 1:
            self.pygrib1 = False

        super(H14Single, self).__init__(filename, mode=mode)

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

        if self.pygrib1:
            param_names = {'40': 'SM_layer1_0-7cm',
                           '41': 'SM_layer2_7-28cm',
                           '42': 'SM_layer3_28-100cm',
                           '43': 'SM_layer4_100-289cm'}

        else:
            param_names = {'SWI1 Soil wetness index in layer 1': 'SM_layer1_0-7cm',
                           'SWI2 Soil wetness index in layer 2': 'SM_layer2_7-28cm',
                           'SWI3 Soil wetness index in layer 3': 'SM_layer3_28-100cm',
                           'SWI4 Soil wetness index in layer 4': 'SM_layer4_100-289cm'}
        data = {}
        metadata = {}

        with pygrib.open(self.filename) as grb:
            for i, message in enumerate(grb):
                message.expand_grid(self.expand_grid)
                if i == 1:
                    lats, lons = message.latlons()
                data[param_names[message['parameterName']]] = message.values

                # read and store metadata
                md = {}
                for k in self.metadata_fields:
                    if message.valid_key(k):
                        md[k] = message[k]
                metadata[param_names[message['parameterName']]] = md

        return Image(lons, lats, data, metadata, timestamp)

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class H14img(MultiTemporalImageBase):

    """
    Class for reading HSAF H14 SM DAS 2 products in grib format
    The images have to be uncompressed in the following folder structure
    path - month_path_str (default 'h14_%Y%m_grib')

    For example if path is set to /home/user/hsaf14 and month_path_str is left to the default 'h14_%Y%m_grib'
    then the images for March 2012 have to be in
    the folder /home/user/hsaf14/h14_201203_grib/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h14_%Y%m_grib'
    file_str: string, optional
        this string is used in datetime.datetime.strftime to get the filename of a H14 daily grib file
        Default: 'H14_%Y%m%d00.grib'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_str
        Default: %Y%m%d
    """

    def __init__(self, path, month_path_str='h14_%Y%m_grib',
                 file_str='H14_{datetime}00.grib',
                 datetime_format='%Y%m%d',
                 expand_grid=True):
        self.path = path
        self.month_path_str = month_path_str
        self.file_search_str = file_str
        super(H14img, self).__init__(path, H14Single,
                                     subpath_templ=[month_path_str],
                                     fname_templ=file_str,
                                     datetime_format=datetime_format,
                                     ioclass_kws={'expand_grid': expand_grid})
