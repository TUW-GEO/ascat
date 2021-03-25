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
Readers for ASCAT Level 1b and Level 2 data in NetCDF format.
"""

import os

import netCDF4
import numpy as np
import xarray as xr

from ascat.utils import tmp_unzip


def read_nc(filename, generic, to_xarray, skip_fields, gen_fields_lut):
    """
    Read NetCDF file.

    Parameters
    ----------
    filename : str
        Filename.
    generic : bool
        'True' reading and converting into generic format or
        'False' reading original field names.
    to_xarray : bool
        'True' return data as xarray.Dataset
        'False' return data as numpy.ndarray.
    skip_fields : list
        Variables to skip.
    gen_fields_lut : dict
        Conversion look-up table for generic names.

    Returns
    -------
    ds : xarray.Dataset, numpy.ndarray
        ASCAT data.
    """
    data = {}
    metadata = {}

    coords_fields = {'utc_line_nodes': 'time',
                     'longitude': 'lon',
                     'latitude': 'lat'}

    with netCDF4.Dataset(filename) as fid:

        if hasattr(fid, 'platform'):
            metadata['platform_id'] = fid.platform[2:]
        elif hasattr(fid, 'platform_long_name'):
            metadata['platform_id'] = fid.platform_long_name[2:]

        metadata['orbit_start'] = fid.start_orbit_number
        metadata['processor_major_version'] = fid.processor_major_version
        metadata['product_minor_version'] = fid.product_minor_version
        metadata['format_major_version'] = fid.format_major_version
        metadata['format_minor_version'] = fid.format_minor_version
        metadata['filename'] = os.path.basename(filename)

        num_rows = fid.dimensions['numRows'].size
        num_cells = fid.dimensions['numCells'].size

        dtype = []
        for var_name in fid.variables.keys():

            if var_name in ['sigma0']:
                continue

            new_var_name = var_name
            var_data = fid.variables[var_name][:].filled()

            if generic:
                if var_name in skip_fields:
                    continue

                if var_name in gen_fields_lut:
                    new_var_name = gen_fields_lut[var_name]

            if len(fid.variables[var_name].shape) == 1:
                var_data = var_data.repeat(num_cells)
            elif len(fid.variables[var_name].shape) == 2:
                var_data = var_data.flatten()
            elif len(fid.variables[var_name].shape) == 3:
                var_data = var_data.reshape(-1, 3)
            else:
                raise RuntimeError('Unknown dimension')

            if var_name in coords_fields:
                new_var_name = coords_fields[var_name]

            if var_name == 'utc_line_nodes':
                var_data = var_data.astype(
                    'timedelta64[s]') + np.datetime64('2000-01-01')

            data[new_var_name] = var_data

            if len(var_data.shape) == 1:
                dtype.append((new_var_name, var_data.dtype.str))
            elif len(var_data.shape) > 1:
                dtype.append((new_var_name, var_data.dtype.str,
                              var_data.shape[1:]))

    num_records = num_rows * num_cells
    coords_fields = ['lon', 'lat', 'time']

    if generic:
        sat_id = np.array([0, 4, 3, 5], dtype=np.uint8)
        data['sat_id'] = np.zeros(
            num_records, dtype=np.uint8) + sat_id[
            int(metadata['platform_id'])]
        dtype.append(('sat_id', np.uint8))

    if to_xarray:
        for k in data.keys():
            if len(data[k].shape) == 1:
                dim = ['obs']
            elif len(data[k].shape) == 2:
                dim = ['obs', 'beam']

            data[k] = (dim, data[k])

        coords = {}
        for cf in coords_fields:
            coords[cf] = data.pop(cf)

        ds = xr.Dataset(data, coords=coords, attrs=metadata)
    else:
        ds = np.empty(num_records, dtype=np.dtype(dtype))
        for k, v in data.items():
            ds[k] = v

    return ds


class AscatL1bNcFile():

    """
    Read ASCAT Level 1b file in NetCDF format.
    """

    def __init__(self, filename):
        """
        Initialize AscatL1bNcFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        if os.path.splitext(filename)[1] == '.gz':
            self.filename = tmp_unzip(filename)
        else:
            self.filename = filename

    def read(self, generic=False, to_xarray=False):
        """
        Read ASCAT Level 1b data.

        Parameters
        ----------
        generic : bool, optional
            'True' reading and converting into generic format or
            'False' reading original field names (default: False).
        to_xarray : bool, optional
            'True' return data as xarray.Dataset
            'False' return data as numpy.ndarray (default: False).

        Returns
        -------
        ds : xarray.Dataset, numpy.ndarray
            ASCAT Level 1b data.
        """
        gen_fields_lut = {'longitude': 'lon', 'latitude': 'lat',
                          'inc_angle_trip': 'inc', 'azi_angle_trip': 'azi',
                          'sigma0_trip': 'sig', 'num_val_trip': 'num_val'}

        skip_fields = ['sat_track_azi', 'f_f', 'f_v', 'f_oa', 'f_sa',
                       'f_tel', 'f_ref', 'abs_line_number']

        ds = read_nc(self.filename, generic, to_xarray,
                     skip_fields, gen_fields_lut)

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

    def __init__(self, filename):
        """
        Initialize AscatL2NcFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        if os.path.splitext(filename)[1] == '.gz':
            self.filename = tmp_unzip(filename)
        else:
            self.filename = filename

    def read(self, generic=False, to_xarray=False):
        """
        Read ASCAT Level 2 data.

        Parameters
        ----------
        generic : bool, optional
            'True' reading and converting into generic format or
            'False' reading original field names (default: False).
        to_xarray : bool, optional
            'True' return data as xarray.Dataset
            'False' return data as numpy.ndarray (default: False).

        Returns
        -------
        ds : dict, xarray.Dataset
            ASCAT Level 2 data.
        """
        gen_fields_lut = {'longitude': 'lon', 'latitude': 'lat',
                          'soil_moisture': 'sm',
                          'soil_moisture_error': 'sm_noise',
                          'sigma40': 'sig40',
                          'sigma40_error': 'sig40_noise',
                          'slope40': 'slop40',
                          'slope40_error': 'slop40_noise',
                          'soil_moisture_sensitivity': 'sens',
                          'dry_backscatter': 'dry_ref',
                          'wet_backscatter': 'wet_ref',
                          'mean_soil_moisture': 'sm_mean',
                          'snow_cover_probability': 'snow_prob',
                          'frozen_soil_probability': 'frozen_prob',
                          'topography_flag': 'topo_flag'}

        skip_fields = ['sat_track_azi', 'abs_line_number']

        ds = read_nc(self.filename, generic, to_xarray,
                     skip_fields, gen_fields_lut)

        return ds

    def close(self):
        """
        Close file.
        """
        pass
