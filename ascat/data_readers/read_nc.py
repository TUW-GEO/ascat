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
General Level 2 data readers for ASCAT data in all formats. Not specific to distributor.
"""

import numpy as np
import pandas as pd
import netCDF4

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.io_base import IntervalReadingMixin
from pygeobase.object_base import Image

class AscatL1SsmNcFile(ImageBase):
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
        super(AscatL1SsmNcFile, self).__init__(filename, mode=mode,
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

