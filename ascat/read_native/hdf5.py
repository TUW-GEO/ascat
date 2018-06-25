# Copyright (c) 2018, TU Wien, Department of Geodesy and Geoinformation
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
Readers for SZF data in h5 format.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import netCDF4
import h5py

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

class AscatL1H5File(ImageBase):
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

    def __init__(self, filename, mode='r', h5_keys=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL1H5File, self).__init__(filename, mode=mode,
                                               **kwargs)
        self.h5_keys = h5_keys
        self.ds = None

    def read(self, timestamp=None):
        """
        reads from the netCDF file given by the filename

        Returns
        -------
        data : pygeobase.object_base.Image
        """

        if self.ds is None:
            self.ds = h5py.File(self.filename)['U-MARF/EPS/ASCA_SZF_1B']
        raw_data = self.ds['DATA']
        raw_metadata = self.ds['METADATA']

        # store data in dictionary
        dd = {}
        # dd = raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'].value

        if self.h5_keys is None:
            var_to_read = list(raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'].dtype.names)
        else:
            var_to_read = self.h5_keys

        # make sure that essential variables are read always:
        if 'LATITUDE_FULL' not in var_to_read:
            var_to_read.append('LATITUDE_FULL')
        if 'LONGITUDE_FULL' not in var_to_read:
            var_to_read.append('LONGITUDE_FULL')

        for name in var_to_read:
            variable = raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'][name]
            if name in raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'].value['EntryName']:
                var_index = np.where(raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR']['EntryName']==name)[0][0]
                if raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'].value['Scale Factor'][var_index] != "n/a":
                    sf = 10**float(raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'].value['Scale Factor'][var_index])
                    variable = variable / sf
            dd[name] = variable[:].flatten()


        # num_cells = self.ds.dimensions['numCells'].size
        # for name in var_to_read:
        #     variable = self.ds.variables[name]
        #     dd[name] = variable[:].flatten()
        #     if len(variable.shape) == 1:
        #         # If the data is 1D then we repeat it for each cell
        #         dd[name] = np.repeat(dd[name], num_cells)

            # if name == 'utc_line_nodes':
            #     utc_dates = netCDF4.num2date(dd[name], variable.units)
            #     dd['jd'] = netCDF4.netcdftime.JulianDayFromDate(utc_dates)

        dd['AS_DES_PASS'] = (dd['SAT_TRACK_AZI'] < 270).astype(np.uint8)

        longitude = dd.pop('LONGITUDE_FULL')
        latitude = dd.pop('LATITUDE_FULL')

        return Image(longitude, latitude, dd, {}, timestamp, timekey='jd')

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
