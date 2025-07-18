# Copyright (c) 2025, TU Wien
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

import zarr
import numpy as np
import pandas as pd
from fibgrid.realization import FibGrid

try:
    import pygrib
except ImportError:
    warnings.warn(
        'pygrib can not be imported GRIB files (H14) can not be read.')

from ascat.file_handling import ChronFiles
from ascat.file_handling import Filenames
from ascat.eumetsat.level2 import AscatL2File
from ascat.read_native.cdr import AscatGriddedNcTs


class AscatNrtBufrFileList(ChronFiles):
    """
    Class reading ASCAT NRT BUFR files.
    """

    def __init__(self,
                 root_path,
                 product_id='*',
                 filename_template=None,
                 subfolder_template=None):
        """
        Initialize.
        """
        if filename_template is None:
            filename_template = '{product_id}_{date}*.buf'

        self.product_id = product_id

        super().__init__(root_path,
                         AscatL2File,
                         filename_template,
                         sf_templ=subfolder_template)

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
        fn_read_fmt = {
            'date': timestamp.strftime('%Y%m%d_%H%M%S'),
            'product_id': self.product_id
        }
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
        return datetime.strptime(
            os.path.basename(filename)[4:19], '%Y%m%d%_H%M%S')

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


class H14Grib(Filenames):
    """
    Class reading H14 soil moisture in GRIB format.
    """

    def __init__(self,
                 filename,
                 expand_grid=True,
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
        super().__init__(filename)
        self.expand_grid = expand_grid
        self.metadata_fields = metadata_fields
        self.pygrib1 = True

        if int(pygrib.__version__[0]) > 1:
            self.pygrib1 = False

    def _read(self, filename, timestamp=None):
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
            param_names = {
                '40': 'SM_layer1_0-7cm',
                '41': 'SM_layer2_7-28cm',
                '42': 'SM_layer3_28-100cm',
                '43': 'SM_layer4_100-289cm'
            }
        else:
            param_names = {
                'SWI1 Soil wetness index in layer 1': 'SM_layer1_0-7cm',
                'SWI2 Soil wetness index in layer 2': 'SM_layer2_7-28cm',
                'SWI3 Soil wetness index in layer 3': 'SM_layer3_28-100cm',
                'SWI4 Soil wetness index in layer 4': 'SM_layer4_100-289cm',
                'Soil wetness index in layer 1': 'SM_layer1_0-7cm',
                'Soil wetness index in layer 2': 'SM_layer2_7-28cm',
                'Soil wetness index in layer 3': 'SM_layer3_28-100cm',
                'Soil wetness index in layer 4': 'SM_layer4_100-289cm'
            }
        data = {}
        metadata = {}

        with pygrib.open(filename) as grb:
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

    def __init__(self,
                 cdr_path,
                 grid_path,
                 fn_format=None,
                 grid_filename='TUW_WARP5_grid_info_2_2.nc',
                 static_layer_path=None,
                 **kwargs):
        """
        Initialize.

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

        super().__init__(cdr_path, fn_format, grid_filename, static_layer_path,
                         **kwargs)


class H121Zarr:
    """
    Class reading ASCAT SSM CDR v8 12.5 km (H121) in zarr data format
    stored as incomplete multidimensional array representation.

    This class is for testing purpose only.
    """

    def __init__(self):
        """Initialize."""
        self.path = "https://www.geo.tuwien.ac.at/shahn/h121/"
        self.lut = zarr.open(self.path, mode="r", path="lut")[:]
        self.data = zarr.open(self.path, mode="r")
        self.grid = FibGrid(12.5)

    def read(self, *args):
        """
        Read time series either by GPI (1 argument) or lon/lat (2 arguments).

        Parameters
        ----------
        gpi : int
            Grid point index.

        or

        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.

        Returns
        -------
        pandas.DataFrame
            Time series data.
        """
        if len(args) == 1:
            gpi = args[0]
            return self.read_gpi(gpi)
        elif len(args) == 2:
            lon, lat = args
            return self.read_lonlat(lon, lat)
        else:
            raise ValueError("Pass either (gpi) or (lon, lat)")

    def read_gpi(self, gpi):
        """
        Read time series for given grid point (Fibonacci 12.5 km).

        Parameters
        ----------
        gpi : int32
            Grid point index.

        Returns
        -------
        df : pandas.DataFrame
            Time series data.
        """
        return self._read_by_gpi(gpi)

    def read_lonlat(self, lon, lat, max_dist=15000.):
        """
        Read the time series data for the grid point closest
        to the given lon/lat coordinates.

        Parameters
        ----------
        lon : float32
            Longitude coordinate.
        lat : float32
            Latitude coordinate.
        max_dist : float32
            Maximum searching distance.

        Returns
        -------
        df : pandas.DataFrame
            Time series data.
        """
        gpi, distance = self.grid.find_nearest_gpi(lon, lat, max_dist)
        return self._read_by_gpi(gpi)

    def _read_by_gpi(self, gpi):
        """
        Read data from grid point index.
        """
        i = self.lut[gpi]
        if i == 861789:
            raise RuntimeError(f"Grid point {gpi} not found in data.")

        dt = self.data["time"][i].astype(np.dtype("<M8[ns]"))

        df = pd.DataFrame(
            {
                "as_des_pass": self.data["as_des_pass"][i],
                "swath_indicator": self.data["swath_indicator"][i],
                "ssm": self.data["surface_soil_moisture"][i],
                "ssm_noise": self.data["surface_soil_moisture_noise"][i],
                "backscatter40": self.data["backscatter40"][i],
                "slope40": self.data["slope40"][i],
                "curvature40": self.data["curvature40"][i],
            },
            index=dt)

        df = df[df.index != np.datetime64("1970-01-01")]
        df.replace(-2**31, np.nan, inplace=True)

        return df
