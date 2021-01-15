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
Readers for ASCAT Level 1b and Level 2 data in NetCDF format.
"""

import netCDF4
import numpy as np
import xarray as xr

from ascat.utils import get_toi_subset, get_roi_subset

class AscatL1NcFile():

    def __init__(self, filename, mode='r'):
        """
        Initialize AscatL1NcFile.

        Parameters
        ----------
        filename : str
            Filename.
        mode : str, optional
            File mode (default: 'r')
        """
        self.filename = filename
        self.mode = mode

    def read(self, toi=None, roi=None):
        """
        Read ASCAT Level 1 data.

        Parameters
        ----------
        toi : tuple of datetime, optional
            Filter data for given time of interest (default: None).
        roi : tuple of 4 float, optional
            Filter data for region of interest (default: None).
            e.g. latmin, lonmin, latmax, lonmax

        Returns
        -------
        ds : dict, xarray.Dataset
            ASCAT Level 1 data.
        """
        data_var = {}
        metadata = {}

        with netCDF4.Dataset(self.filename) as fid:

            metadata['sat_id'] = fid.platform[-1]
            metadata['orbit_start'] = fid.start_orbit_number
            metadata['processor_major_version'] = fid.processor_major_version
            metadata['product_minor_version'] = fid.product_minor_version
            metadata['format_major_version'] = fid.format_major_version
            metadata['format_minor_version'] = fid.format_minor_version

            num_cells = fid.dimensions['numCells'].size
            num_rows = fid.dimensions['numRows'].size

            for var_name in fid.variables.keys():

                if var_name in ['sigma0']:
                    continue

                if len(fid.variables[var_name].shape) == 1:
                    dim = ['num_rows']
                elif len(fid.variables[var_name].shape) == 2:
                    dim = ['num_rows', 'num_cells']
                elif len(fid.variables[var_name].shape) == 3:
                    dim = ['num_rows', 'num_cells', 'num_sigma0']
                else:
                    raise RuntimeError('Unknown dimension')

                if var_name == 'utc_line_nodes':
                    data_var[var_name] = (dim, fid.variables[
                        var_name][:].filled().astype(
                            'timedelta64[s]') + np.datetime64('2000-01-01'))
                else:
                    data_var[var_name] = (
                        dim, fid.variables[var_name][:].filled())

            data_var['as_des_pass'] = (
                ['num_rows'], (data_var[
                    'sat_track_azi'][1] < 270).astype(np.uint8))

        coords = {"lon": data_var.pop('longitude'),
                  "lat": data_var.pop('latitude'),
                  "time": data_var.pop('utc_line_nodes')}

        data_var['node_num'] = (['num_rows', 'num_cells'], np.tile(np.arange(
            1, num_cells+1), (num_rows, 1)))
        data_var['line_num'] = (['num_rows', 'num_cells'], np.arange(
            1, num_rows+1).repeat(num_cells).reshape(-1, num_cells))

        ds = xr.Dataset(data_var, coords=coords, attrs=metadata)

        if toi:
            ds = get_toi_subset(ds, toi)

        if roi:
            ds = get_roi_subset(ds, roi)

        return ds

    def close(self):
        """
        Close file.
        """
        pass


class AscatL2NcFile:

    """
    Read ASCAT Level 2 file in NetCDF format.
    """

    def __init__(self, filename, mode='r'):
        """
        Initialize AscatL1NcFile.

        Parameters
        ----------
        filename : str
            Filename.
        mode : str, optional
            File mode (default: 'r')
        """
        self.filename = filename
        self.mode = mode

    def read(self, toi=None, roi=None):
        """
        Read ASCAT Level 2 data.

        Parameters
        ----------
        toi : tuple of datetime, optional
            Filter data for given time of interest (default: None).
        roi : tuple of 4 float, optional
            Filter data for region of interest (default: None).
            latmin, lonmin, latmax, lonmax

        Returns
        -------
        ds : dict, xarray.Dataset
            ASCAT Level 2 data.
        """
        data_var = {}
        metadata = {}

        with netCDF4.Dataset(self.filename) as fid:

            metadata['sat_id'] = fid.platform_long_name[-1]
            metadata['orbit_start'] = fid.start_orbit_number
            metadata['processor_major_version'] = fid.processor_major_version
            metadata['product_minor_version'] = fid.product_minor_version
            metadata['format_major_version'] = fid.format_major_version
            metadata['format_minor_version'] = fid.format_minor_version

            num_cells = fid.dimensions['numCells'].size
            num_rows = fid.dimensions['numRows'].size

            for var_name in fid.variables.keys():

                if len(fid.variables[var_name].shape) == 1:
                    dim = ['num_rows']
                elif len(fid.variables[var_name].shape) == 2:
                    dim = ['num_rows', 'num_cells']
                elif len(fid.variables[var_name].shape) == 3:
                    dim = ['num_rows', 'num_cells', 'num_sigma0']
                else:
                    raise RuntimeError('Unknown dimension')

                if var_name == 'utc_line_nodes':
                    data_var[var_name] = (dim, fid.variables[
                        var_name][:].filled().astype(
                            'timedelta64[s]') + np.datetime64('2000-01-01'))
                else:
                    data_var[var_name] = (
                        dim, fid.variables[var_name][:].filled())

        data_var['as_des_pass'] = (
            ['num_rows'], (data_var[
                'sat_track_azi'][1] < 270).astype(np.uint8))

        coords = {"lon": data_var.pop('longitude'),
                  "lat": data_var.pop('latitude'),
                  "time": data_var.pop('utc_line_nodes')}

        data_var['node_num'] = (['num_rows', 'num_cells'], np.tile(np.arange(
            1, num_cells+1), (num_rows, 1)))
        data_var['line_num'] = (['num_rows', 'num_cells'], np.arange(
            1, num_rows+1).repeat(num_cells).reshape(-1, num_cells))

        ds = xr.Dataset(data_var, coords=coords, attrs=metadata)

        if toi:
            ds = get_toi_subset(ds, toi)

        if roi:
            ds = get_roi_subset(ds, roi)

        return ds

    def close(self):
        """
        Close file.
        """
        pass
