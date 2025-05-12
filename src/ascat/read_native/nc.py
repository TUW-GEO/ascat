# Copyright (c) 2025, TU Wien, Department of Geodesy and Geoinformation
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
from datetime import datetime
from datetime import timedelta

import netCDF4
import numpy as np
import xarray as xr

from ascat.utils import tmp_unzip
from ascat.utils import daterange
from ascat.utils import mask_dtype_nans
from ascat.utils import uint8_nan
from ascat.utils import float32_nan
from ascat.file_handling import ChronFiles
from ascat.read_native import AscatFile


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
    data : xarray.Dataset or numpy.ndarray
        ASCAT data.
    metadata : dict
        Metadata.
    """
    data = {}
    metadata = {}

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

            if generic and var_name in skip_fields:
                continue

            if generic and var_name in gen_fields_lut:
                new_var_name = gen_fields_lut[var_name][0]
                fill_value = gen_fields_lut[var_name][2]
            else:
                new_var_name = var_name
                fill_value = None

            var_data = fid.variables[var_name][:].filled(fill_value)

            if var_name == 'azi_angle_trip':
                var_data[(var_data < 0) & (var_data != fill_value)] += 360

            if len(fid.variables[var_name].shape) == 1:
                var_data = var_data.repeat(num_cells)
            elif len(fid.variables[var_name].shape) == 2:
                var_data = var_data.flatten()
            elif len(fid.variables[var_name].shape) == 3:
                var_data = var_data.reshape(-1, 3)
            else:
                raise RuntimeError('Unknown dimension')

            if var_name == 'utc_line_nodes':
                var_data = var_data.astype('timedelta64[s]') + np.datetime64(
                    '2000-01-01')

            data[new_var_name] = var_data

            if len(var_data.shape) == 1:
                dtype.append((new_var_name, var_data.dtype.str))
            elif len(var_data.shape) > 1:
                dtype.append(
                    (new_var_name, var_data.dtype.str, var_data.shape[1:]))

    num_records = num_rows * num_cells
    coords_fields = ['lon', 'lat', 'time']

    if generic:
        sat_id = np.array([0, 4, 3, 5], dtype=np.uint8)
        data['sat_id'] = np.zeros(num_records, dtype=np.uint8) + sat_id[int(
            metadata['platform_id'])]
        dtype.append(('sat_id', np.uint8))

        n_records = data['lat'].shape[0]
        n_lines = n_records // num_cells

        data['node_num'] = np.tile((np.arange(num_cells) + 1), n_lines)
        dtype.append(('node_num', np.uint8))

        data['line_num'] = np.arange(n_lines).repeat(num_cells)
        dtype.append(('line_num', np.int32))

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

        data = xr.Dataset(data, coords=coords, attrs=metadata)
        if generic:
            data = mask_dtype_nans(data)
    else:
        ds = np.empty(num_records, dtype=np.dtype(dtype))
        for k, v in data.items():
            ds[k] = v
        data = ds

    return data, metadata

class AscatL1bNcFile(AscatFile):
    """
    Read ASCAT Level 1b file in NetCDF format.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize AscatL1bNcFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        super().__init__(filename, **kwargs)
        for i, fname in enumerate(self.filenames):
            if os.path.splitext(fname)[1] == '.gz':
                self.filenames[i] = tmp_unzip(fname)

    def _read(self, filename, generic=False, to_xarray=False):
        """
        Read one ASCAT Level 1b NetCDF4 file.

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
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.
        """
        gen_fields_lut = {
            'longitude': ('lon', np.float32, None),
            'latitude': ('lat', np.float32, None),
            'utc_line_nodes': ('time', np.float32, None),
            'inc_angle_trip': ('inc', np.float32, float32_nan),
            'azi_angle_trip': ('azi', np.float32, float32_nan),
            'sigma0_trip': ('sig', np.float32, float32_nan),
            'kp': ('kp', np.float32, float32_nan),
            'f_kp': ('kp_quality', np.float32, uint8_nan),
            'num_val_trip': ('num_val', np.float32, None)
        }

        skip_fields = [
            'f_f', 'f_v', 'f_oa', 'f_sa', 'f_tel', 'f_ref', 'abs_line_number'
        ]

        data, metadata = read_nc(filename, generic, to_xarray,
                                 skip_fields, gen_fields_lut)

        return data, metadata

    def _merge(self, data):
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
        if isinstance(data[0], tuple):
            data, metadata = zip(*data)
            if isinstance(data[0], xr.Dataset):
                data = xr.concat(data,
                                 dim="obs",
                                 combine_attrs="drop_conflicts")
            else:
                data = np.hstack(data)
            data = (data, metadata)
        else:
            data = np.hstack(data)

        return data

class AscatL1bNcFileGeneric(AscatL1bNcFile):
    """
    The same as AscatL1bNcFile but with generic=True by default.
    """
    def _read(self, filename, generic=True, to_xarray=False, **kwargs):
        return super()._read(filename, generic=generic, to_xarray=to_xarray, **kwargs)


class AscatL2NcFile(AscatFile):
    """
    Read ASCAT Level 2 file in NetCDF format.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize AscatL2NcFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        super().__init__(filename, **kwargs)
        for i, fname in enumerate(self.filenames):
            if os.path.splitext(fname)[1] == '.gz':
                self.filenames[i] = tmp_unzip(fname)

    def _read(self, filename, generic=False, to_xarray=False):
        """
        Read one ASCAT Level 2 NetCDF4 file.

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
        gen_fields_lut = {
            'longitude': ('lon', np.float32, None),
            'latitude': ('lat', np.float32, None),
            'utc_line_nodes': ('time', np.float32, None),
            'inc_angle_trip': ('inc', np.float32, float32_nan),
            'azi_angle_trip': ('azi', np.float32, float32_nan),
            'sigma0_trip': ('sig', np.float32, float32_nan),
            'kp': ('kp', np.float32, float32_nan),
            'soil_moisture': ('sm', np.float32, float32_nan),
            'soil_moisture_error': ('sm_noise', np.float32, float32_nan),
            'sigma40': ('sig40', np.float32, float32_nan),
            'sigma40_error': ('sig40_noise', np.float32, float32_nan),
            'slope40': ('slope40', np.float32, float32_nan),
            'slope40_error': ('slope40_noise', np.float32, float32_nan),
            'soil_moisture_sensitivity': ('sm_sens', np.float32, float32_nan),
            'dry_backscatter': ('dry_sig40', np.float32, float32_nan),
            'wet_backscatter': ('wet_sig40', np.float32, float32_nan),
            'mean_soil_moisture': ('sm_mean', np.float32, float32_nan),
            'proc_flag1': ('corr_flag', np.uint8, None),
            'proc_flag2': ('proc_flag', np.uint8, None),
            'aggregated_quality_flag': ('agg_flag', np.uint8, None),
            'snow_cover_probability': ('snow_prob', np.uint8, None),
            'frozen_soil_probability': ('frozen_prob', np.uint8, None),
            'wetland_flag': ('wetland', np.uint8, None),
            'topography_flag': ('topo', np.uint8, None)
        }

        skip_fields = ['abs_line_number']

        data, metadata = read_nc(filename, generic, to_xarray,
                                 skip_fields, gen_fields_lut)

        return data, metadata

    def _merge(self, data):
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
        if isinstance(data[0], tuple):
            data, metadata = zip(*data)
            if isinstance(data[0], xr.Dataset):
                data = xr.concat(data,
                                 dim="obs",
                                 combine_attrs="drop_conflicts")
            else:
                data = np.hstack(data)
            data = (data, metadata)
        else:
            data = np.hstack(data)

        return data

class AscatL2NcFileGeneric(AscatL2NcFile):
    """
    The same as AscatL1bNcFile but with generic=True by default.
    """
    def _read(self, filename, generic=True, to_xarray=False, **kwargs):
        return super()._read(filename, generic=generic, to_xarray=to_xarray, **kwargs)


class AscatSsmNcSwathFile(AscatFile):
    """
    Class reading ASCAT Surface Soil Moisture Netcdf swath file.
    """

    def _read(self, filename, mask_and_scale=None, sel_dt=None):
        """
        Read/load data from NetCDF file.

        Parameters
        ----------
        mask_and_scale : boolean, optional
            Mask and scale data using _FillValue.

        Returns
        -------
        data : xarray.Dataset
            Data.
        """
        with xr.open_dataset(self.filename,
                             mask_and_scale=mask_and_scale) as ds:
            data = ds.load()

        if sel_dt is not None:
            data = data.where((data.time >= sel_dt[0])
                              & (data.time <= sel_dt[1]),
                              drop=True)

        return data


class AscatSsmNcSwathFileList(ChronFiles):
    """
    Class reading ASCAT Surface Soil Moisture Netcdf swath file list.
    """

    def __init__(self,
                 path,
                 filename_template=None,
                 subfolder_template=None,
                 sat="?",
                 cls_kwargs=None):
        """
        Initialize object.

        Parameters
        ----------
        path : str
            Root path to data.
        filename_template : str, optional
            Filename template (default: "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H???_C_LIIB_{date}_*_*____.nc")
        subfolder_template : str, optional
            Subfolder template (default: /path/metop_{sat}/{year}/)
        sat : str, optional
            Satellite Metop-A: "A", Metop-B: "B", Metop-C: "C" or all: "?"
            (default: "?")
        cls_kwargs : dict, optional
            Keyword arguments passed to file IO class (default: None).
        """
        if filename_template is None:
            filename_template = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H???_C_LIIB_{date}_*_*____.nc"

        if subfolder_template is None:
            subfolder_template = {"satellite": "metop_{sat}", "date": "{year}"}

        if cls_kwargs is None:
            cls_kwargs = {}

        self.sat = sat

        super().__init__(path,
                         AscatSsmNcSwathFile,
                         filename_template,
                         sf_templ=subfolder_template,
                         cls_kwargs=cls_kwargs)

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
            "date": timestamp.strftime("%Y%m%d%H%M%S"),
            "sat": self.sat.upper()
        }
        fn_write_fmt = None

        sf_read_fmt = {
            "satellite": {
                "sat": self.sat
            },
            "date": {
                "year": timestamp.strftime("%Y"),
                "month": timestamp.strftime("%m"),
                "day": timestamp.strftime("%d")
            }
        }
        sf_write_fmt = None

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename, start=53, end=64):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.
        start : int, optional
            Start position of date field (default: 58).
        start : int, optional
            End position  of date field (default: 72).

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(
            os.path.basename(filename)[start:end], "%Y%m%d%H%M%S")

    def search_date(self, timestamp, **kwargs):
        """
        Search date.

        Parameters
        ----------
        timestamp : datetime
            Date.

        Returns
        -------
        filenames : list
            Filenames.
        """
        return super().search_date(timestamp, date_str="%Y%m%d%H*", **kwargs)

    def iter_daterange(self, start_date, end_date):
        """
        Generator returning filenames between start and end date.

        Parameters
        ----------
        start_date : datetime
            Start date.
        end_date : datetime
            End date.

        Yields
        ------
        filename : str
            Filename.
        """
        for single_date in daterange(start_date, end_date):
            filenames = self.search_date(single_date)
            for filename in filenames:
                yield filename

    def read_date(self, timestamp):
        """
        Read data for given timestamp.

        Parameters
        ----------
        timestamp : datetime
            Date.

        Returns
        -------
        data : xarray.Dataset
            Data.
        """
        filenames = self.search_date(timestamp)

        if len(filenames) > 1:
            raise RuntimeError(
                f"Multiple files found for timestamp {timestamp}")
        elif len(filenames) == 0:
            print(f"No file found for timestamp {timestamp}")
            data = None
        else:
            self._open(filenames[0])
            data = self.fid.read()

        return data

    def read_period(self,
                    start_dt,
                    end_dt,
                    delta_dt=timedelta(hours=1),
                    buffer_dt=timedelta(hours=1),
                    **kwargs):
        """
        Read data for given interval.

        Parameters
        ----------
        start_dt : datetime
            Start datetime.
        end_dt : datetime
            End datetime.
        delta_dt : timedelta, optional
            Time delta used to jump through search date.
        buffer_dt : timedelta, optional
            Search buffer used to find files which could possibly contain
            data but would be left out because of dt_start.

        Returns
        -------
        data : dict, numpy.ndarray
            Data stored in file.
        """
        filenames = self.search_period(start_dt - buffer_dt,
                                       end_dt + buffer_dt, delta_dt)

        merged_data = []

        sel_dt = (np.datetime64(start_dt - timedelta(minutes=15)),
                  np.datetime64(end_dt + timedelta(minutes=15)))

        for filename in filenames:
            self._open(filename)

            try:
                data = self.fid.read(sel_dt=sel_dt)

                if data is not None:
                    merged_data.append(data)
            except:
                print(f"Error reading: {self.fid.filename}")

        if merged_data:
            merged_data = xr.concat(merged_data,
                                    dim="obs",
                                    combine_attrs="drop_conflicts")
        else:
            merged_data = None

        return merged_data
