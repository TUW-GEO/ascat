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

"""
Readers for H SAF soil moisture products.
"""

import os
import glob
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import pygrib
except ImportError:
    warnings.warn(
        'pygrib can not be imported GRIB files (H14) can not be read.')

from ascat.utils import tmp_unzip
from ascat.file_handling import ChronFiles
from ascat.read_native.bufr import BUFRReader
from ascat.eumetsat.level2 import AscatL2File
from ascat.read_native.cdr import AscatGriddedNcTs


class H08Bufr:

    def __init__(self, filename):
        """
        Initialize H08Bufr.

        Parameters
        ----------
        filename : str
            Filename.
        """
        if os.path.splitext(filename)[1] == '.gz':
            self.filename = tmp_unzip(filename)
        else:
            self.filename = filename

    def read(self):
        """
        Read file.

        Returns
        -------
        data : numpy.ndarray
            H08 data.
        """
        data = defaultdict(list)

        with BUFRReader(self.filename) as bufr:

            lons = []
            ssm = []
            ssm_noise = []
            ssm_corr_flag = []
            ssm_proc_flag = []

            for i, message in enumerate(bufr.messages()):
                if i == 0:
                    # first message is just lat, lon extent
                    # check if any data in bbox
                    # lon_min, lon_max = message[0, 2], message[0, 3]
                    lat_min, lat_max = message[0, 4], message[0, 5]
                else:
                    # first 5 elements are there only once, after that,
                    # 4 elements are repeated till the end of the array
                    # these 4 are ssm, ssm_noise, ssm_corr_flag and
                    # ssm_proc_flag each message contains the values for
                    # 120 lons between lat_min and lat_max the grid spacing
                    # is 0.00416667 degrees
                    lons.append(message[:, 0])
                    lat_min = message[0, 1]
                    lat_max = message[0, 2]
                    ssm.append(message[:, 4::4])
                    ssm_noise.append(message[:, 5::4])
                    ssm_corr_flag.append(message[:, 6::4])
                    ssm_proc_flag.append(message[:, 7::4])

        ssm = np.rot90(np.vstack(ssm)).astype(np.float32)
        ssm_noise = np.rot90(np.vstack(ssm_noise)).astype(np.float32)
        ssm_corr_flag = np.rot90(
            np.vstack(ssm_corr_flag)).astype(np.float32)
        ssm_proc_flag = np.rot90(
            np.vstack(ssm_proc_flag)).astype(np.float32)
        lats_dim = np.linspace(lat_max, lat_min, ssm.shape[0])
        lons_dim = np.concatenate(lons)

        data = {'ssm': ssm, 'ssm_noise': ssm_noise,
                'proc_flag': ssm_proc_flag, 'corr_flag': ssm_corr_flag}

        # if there are is a gap in the image it is not a 2D array in
        # lon, lat space but has a jump in latitude or longitude
        # detect a jump in lon or lat spacing

        lon_jump_ind = np.where(np.diff(lons_dim) > 0.00418)[0]

        if lon_jump_ind.size > 1:
            print("More than one jump in longitude")

        if lon_jump_ind.size == 1:
            lon_jump_ind = lon_jump_ind[0]
            diff_lon_jump = np.abs(
                lons_dim[lon_jump_ind] - lons_dim[lon_jump_ind + 1])
            missing_elements = int(np.round(diff_lon_jump / 0.00416666))
            missing_lons = np.linspace(lons_dim[lon_jump_ind],
                                       lons_dim[lon_jump_ind + 1],
                                       missing_elements,
                                       endpoint=False)

            # fill up longitude dimension to full grid
            lons_dim = np.concatenate([lons_dim[:lon_jump_ind],
                                       missing_lons,
                                       lons_dim[lon_jump_ind + 1:]])

            # fill data with NaN values
            empty = np.empty((lats_dim.shape[0], missing_elements))
            empty.fill(1e38)
            for key in data:
                data[key] = np.concatenate(
                    [data[key][:, :lon_jump_ind],
                        empty, data[key][:, lon_jump_ind + 1:]], axis=1)

        lat_jump_ind = np.where(np.diff(lats_dim) > 0.00418)[0]

        if lat_jump_ind.size > 1:
            print("More than one jump in latitude")

        if lat_jump_ind.size == 1:
            diff_lat_jump = np.abs(
                lats_dim[lat_jump_ind] - lats_dim[lat_jump_ind + 1])
            missing_elements = np.round(diff_lat_jump / 0.00416666)
            missing_lats = np.linspace(lats_dim[lat_jump_ind],
                                       lats_dim[lat_jump_ind + 1],
                                       missing_elements,
                                       endpoint=False)

            # fill up longitude dimension to full grid
            lats_dim = np.concatenate(
                [lats_dim[:lat_jump_ind], missing_lats,
                    lats_dim[lat_jump_ind + 1:]])
            # fill data with NaN values
            empty = np.empty((missing_elements, lons_dim.shape[0]))
            empty.fill(1e38)
            for key in data:
                data[key] = np.concatenate(
                    [data[key][:lat_jump_ind, :], empty,
                        data[key][lat_jump_ind + 1:, :]], axis=0)

        data['lon'], data['lat'] = np.meshgrid(lons_dim, lats_dim)

        return data

    def close(self):
        """
        Close file.
        """
        pass


class H08BufrFileList(ChronFiles):

    """
    Reads H SAF H08 data.
    """

    def __init__(self, path):
        """
        Initialize.
        """
        fn_templ = 'h08_{date}*.buf'
        sf_templ = {'month': 'h08_{date}_buf'}

        super().__init__(path, H08Bufr, fn_templ, sf_templ=sf_templ)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {'date': timestamp.strftime('%Y%m%d_%H%M%S')}
        sf_read_fmt = {'month': {'date': timestamp.strftime('%Y%m')}}

        fn_write_fmt = None
        sf_write_fmt = None

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(os.path.basename(filename)[4:19],
                                 '%Y%m%d_%H%M%S')

    def read_period(dt_start, dt_end, delta):
        """
        Read period not implemented.
        """
        raise NotImplementedError


class AscatNrtBufrFileList(ChronFiles):

    def __init__(self, root_path, product_id='*'):
        """
        Initialize.
        """
        fn_templ = '{product_id}_{date}*.buf'
        sf_templ = None

        self.product_id = product_id

        super().__init__(root_path, AscatL2File, fn_templ, sf_templ=sf_templ)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {'date': timestamp.strftime('%Y%m%d_%H%M%S'),
                       'product_id': self.product_id}
        sf_read_fmt = None
        fn_write_fmt = None
        sf_write_fmt = None

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(os.path.basename(filename)[4:19],
                                 '%Y%m%d%_H%M%S')

    def _merge_data(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        return np.hstack(data)


class H14Grib:

    """
    Class reading H14 soil moisture in GRIB format.
    """

    def __init__(self, filename, expand_grid=True,
                 metadata_fields=['units', 'name']):
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
        self.filename = filename
        self.expand_grid = expand_grid
        self.metadata_fields = metadata_fields
        self.pygrib1 = True

        if int(pygrib.__version__[0]) > 1:
            self.pygrib1 = False

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
        """
        if self.pygrib1:
            param_names = {'40': 'SM_layer1_0-7cm',
                           '41': 'SM_layer2_7-28cm',
                           '42': 'SM_layer3_28-100cm',
                           '43': 'SM_layer4_100-289cm'}
        else:
            param_names = {
                'SWI Soil wetness index in layer 1': 'SM_layer1_0-7cm',
                'SWI Soil wetness index in layer 2': 'SM_layer2_7-28cm',
                'SWI Soil wetness index in layer 3': 'SM_layer3_28-100cm',
                'SWI Soil wetness index in layer 4': 'SM_layer4_100-289cm',
                'Soil wetness index in layer 1': 'SM_layer1_0-7cm',
                'Soil wetness index in layer 2': 'SM_layer2_7-28cm',
                'Soil wetness index in layer 3': 'SM_layer3_28-100cm',
                'Soil wetness index in layer 4': 'SM_layer4_100-289cm'}
        data = {}
        metadata = {}

        with pygrib.open(self.filename) as grb:
            for i, message in enumerate(grb):
                message.expand_grid(self.expand_grid)
                if i == 1:
                    data['lat'], data['lon'] = message.latlons()

                data[param_names[message['parameterName']]] = message.values

                # read and store metadata
                md = {}
                for k in self.metadata_fields:
                    if message.valid_key(k):
                        md[k] = message[k]

                metadata[param_names[message['parameterName']]] = md

        return data

    def close(self):
        pass


class H14GribFileList(ChronFiles):

    """
    Reads H SAF H08 data.
    """

    def __init__(self, path):
        """
        Initialize.
        """
        fn_templ = 'H14_{date}.grib'
        sf_templ = {'month': 'h14_{date}_grib'}

        super().__init__(path, H14Grib, fn_templ, sf_templ=sf_templ)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {'date': timestamp.strftime('%Y%m%d%H')}
        sf_read_fmt = {'month': {'date': timestamp.strftime('%Y%m')}}
        fn_write_fmt = None
        sf_write_fmt = None

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(os.path.basename(filename)[4:15], '%Y%m%d%H')

    def read_period(dt_start, dt_end, delta):
        """
        Read period not implemented.
        """
        raise NotImplementedError()


class AscatSsmDataRecord(AscatGriddedNcTs):

    """
    Class reading Metop ASCAT soil moisture data record.
    """

    def __init__(self, cdr_path, grid_path, fn_format=None,
                 grid_filename='TUW_WARP5_grid_info_2_2.nc',
                 static_layer_path=None, **kwargs):
        """

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
        if fn_format is None:
            first_file = glob.glob(os.path.join(cdr_path, '*.nc'))

            if len(first_file) == 0:
                raise RuntimeError('No files found')

            version = os.path.basename(first_file[0]).rsplit('_', 1)[0]
            fn_format = '{:}_{{:04d}}'.format(version)

        grid_filename = os.path.join(grid_path, grid_filename)

        super().__init__(cdr_path, fn_format, grid_filename,
                         static_layer_path, **kwargs)
