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
Readers for ASCAT Level 1b and Level 2 data in BUFR format.
"""

import os

import numpy as np
import xarray as xr
from cadati.cal_date import cal2dt

try:
    import pdbufr
except ImportError:
    pass

from ascat.utils import tmp_unzip
from ascat.utils import mask_dtype_nans
from ascat.utils import uint8_nan
from ascat.utils import uint16_nan
from ascat.utils import int32_nan
from ascat.utils import float32_nan
from ascat.read_native import AscatFile

bufr_nan = 1.7e+38

nan_val_dict = {
    np.float32: float32_nan,
    np.uint8: uint8_nan,
    np.uint16: uint16_nan,
    np.int32: int32_nan
}

class AscatL1bBufrFile(AscatFile):
    """
    Read ASCAT Level 1b file in BUFR format.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize AscatL1bBufrFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        super().__init__(filename, **kwargs)

        for i, fname in enumerate(self.filenames):
            if os.path.splitext(fname)[1] == '.gz':
                self.filenames[i] = tmp_unzip(fname)

        self.msg_name_lookup = {
                4: "Satellite Identifier",
                6: "Direction Of Motion Of Moving Observing Platform",
                16: "Orbit Number",
                17: "Cross-Track Cell Number",
                21: "f_Beam Identifier",
                22: "f_Radar Incidence Angle",
                23: "f_Antenna Beam Azimuth",
                24: "f_Backscatter",
                25: "f_Radiometric Resolution (Noise Value)",
                26: "f_ASCAT KP Estimate Quality",
                27: "f_ASCAT Sigma-0 Usability",
                34: "f_ASCAT Land Fraction",
                35: "m_Beam Identifier",
                36: "m_Radar Incidence Angle",
                37: "m_Antenna Beam Azimuth",
                38: "m_Backscatter",
                39: "m_Radiometric Resolution (Noise Value)",
                40: "m_ASCAT KP Estimate Quality",
                41: "m_ASCAT Sigma-0 Usability",
                48: "m_ASCAT Land Fraction",
                49: "a_Beam Identifier",
                50: "a_Radar Incidence Angle",
                51: "a_Antenna Beam Azimuth",
                52: "a_Backscatter",
                53: "a_Radiometric Resolution (Noise Value)",
                54: "a_ASCAT KP Estimate Quality",
                55: "a_ASCAT Sigma-0 Usability",
                62: "a_ASCAT Land Fraction"
            }

    def _read(self, filename, generic=False, to_xarray=False):
        """
        Read one ASCAT Level 1b BUFR file.

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
        df = pdbufr.read_bufr(filename, columns="data", flat=True)

        col_rename = {}
        for i, col in enumerate(df.columns.to_list()):
            name = self.msg_name_lookup.get(i + 1, None)
            if name is not None:
                col_rename[col] = name

        data = df.rename(columns=col_rename)[col_rename.values()]

        data["lat"] = df["#1#latitude"].values.astype(np.float32)
        data["lon"] = df["#1#longitude"].values.astype(np.float32)

        year = df["#1#year"].values.astype(int)
        month = df["#1#month"].values.astype(int)
        day = df["#1#day"].values.astype(int)
        hour = df["#1#hour"].values.astype(int)
        minute = df["#1#minute"].values.astype(int)
        seconds = df["#1#second"].values.astype(int)
        milliseconds = np.zeros(seconds.size)
        cal_dates = np.vstack(
            (year, month, day, hour, minute, seconds, milliseconds)).T

        data['time'] = cal2dt(cal_dates)
        data = data.to_records(index=False)
        data = {name:data[name] for name in data.dtype.names}

        metadata = {}
        metadata['platform_id'] = data['Satellite Identifier'][0].astype(int)
        metadata['orbit_start'] = np.uint32(data['Orbit Number'][0])
        metadata['filename'] = os.path.basename(filename)

        # add/rename/remove fields according to generic format
        if generic:
            data = conv_bufrl1b_generic(data, metadata)

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data.keys():
                if len(data[k].shape) == 1:
                    dim = ['obs']
                elif len(data[k].shape) == 2:
                    dim = ['obs', 'beam']

                data[k] = (dim, data[k])

            coords = {}
            coords_fields = ['lon', 'lat', 'time']
            for cf in coords_fields:
                coords[cf] = data.pop(cf)

            data = xr.Dataset(data, coords=coords, attrs=metadata)
            if generic:
                data = mask_dtype_nans(data)
        else:
            # collect dtype info
            dtype = []
            for var_name in data.keys():
                if len(data[var_name].shape) == 1:
                    dtype.append((var_name, data[var_name].dtype.str))
                elif len(data[var_name].shape) > 1:
                    dtype.append((var_name, data[var_name].dtype.str,
                                  data[var_name].shape[1:]))

            ds = np.empty(data['time'].size, dtype=np.dtype(dtype))
            for k, v in data.items():
                ds[k] = v

            data = ds

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
        data : numpy.ndarray or xarray.Dataset
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

class AscatL1bBufrFileGeneric(AscatL1bBufrFile):
    """
    The same as AscatL1bBufrFile but with generic=True by default.
    """
    def _read(self, filename, generic=True, to_xarray=False, **kwargs):
        return super()._read(filename, generic=generic, to_xarray=to_xarray, **kwargs)


def conv_bufrl1b_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Spacecraft_id vs sat_id encoding

    BUFR encoding - Spacecraft_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-1 (Metop-B)
    - 4 Metop-2 (Metop-A)
    - 5 Metop-3 (Metop-C)

    Internal encoding - sat_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-2 (Metop-A)
    - 4 Metop-1 (Metop-B)
    - 5 Metop-3 (Metop-C)

    Parameters
    ----------
    data: dict of numpy.ndarray
        Original dataset.
    metadata: dict
        Metadata.

    Returns
    -------
    data: dict of numpy.ndarray
        Converted dataset.
    """
    skip_fields = ['Satellite Identifier']

    gen_fields_beam = {
        'Radar Incidence Angle': ('inc', np.float32, bufr_nan, 1),
        'Backscatter': ('sig', np.float32, bufr_nan, 1),
        'Antenna Beam Azimuth': ('azi', np.float32, bufr_nan, 1),
        'ASCAT Sigma-0 Usability': ('f_usable', np.uint8, None, 1),
        'Beam Identifier': ('beam_num', np.uint8, None, 1),
        'Radiometric Resolution (Noise Value)':
        ('kp', np.float32, bufr_nan, 0.01),
        'ASCAT KP Estimate Quality': ('kp_quality', np.uint8, bufr_nan, 1),
        'ASCAT Land Fraction': ('f_land', np.float32, None, 1)
    }

    gen_fields_lut = {
        'Orbit Number': ('abs_orbit_nr', np.int32),
        'Cross-Track Cell Number': ('node_num', np.uint8),
        'Direction Of Motion Of Moving Observing Platform':
        ('sat_track_azi', np.float32)
    }

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name, (new_name, new_dtype) in gen_fields_lut.items():
        data[new_name] = data.pop(var_name).astype(new_dtype)

    for var_name, (new_name, new_dtype, nan_val, s) in gen_fields_beam.items():
        f = ['{}_{}'.format(b, var_name) for b in ['f', 'm', 'a']]
        data[new_name] = np.vstack((data.pop(f[0]), data.pop(f[1]),
                                    data.pop(f[2]))).T.astype(new_dtype)
        if nan_val is not None:
            valid = data[new_name] != nan_val
            data[new_name][~valid] = nan_val_dict[new_dtype]
            data[new_name][valid] *= s

    if data['node_num'].max() == 82:
        data['swath_indicator'] = 1 * (data['node_num'] > 41)
    elif data['node_num'].max() == 42:
        data['swath_indicator'] = 1 * (data['node_num'] > 21)
    else:
        raise ValueError('Cross-track cell number size unknown')

    n_lines = data['lat'].shape[0] / data['node_num'].max()
    data['line_num'] = np.arange(n_lines).repeat(data['node_num'].max())

    sat_id = np.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
    data['sat_id'] = np.zeros(data['time'].size, dtype=np.uint8) + sat_id[int(
        metadata['platform_id'])]

    # compute ascending/descending direction
    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data


class AscatL2BufrFile(AscatFile):
    """
    Read ASCAT Level 2 file in BUFR format.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize AscatL2BufrFile.

        Parameters
        ----------
        filename: str
            Filename.
        """
        super().__init__(filename, **kwargs)

        for i, fname in enumerate(self.filenames):
            if os.path.splitext(fname)[1] == '.gz':
                self.filenames[i] = tmp_unzip(fname)

        self.msg_name_lookup = {
            4: "Satellite Identifier",
            6: "Direction Of Motion Of Moving Observing Platform",
            16: "Orbit Number",
            17: "Cross-Track Cell Number",
            21: "f_Beam Identifier",
            22: "f_Radar Incidence Angle",
            23: "f_Antenna Beam Azimuth",
            24: "f_Backscatter",
            25: "f_Radiometric Resolution (Noise Value)",
            26: "f_ASCAT KP Estimate Quality",
            27: "f_ASCAT Sigma-0 Usability",
            34: "f_ASCAT Land Fraction",
            35: "m_Beam Identifier",
            36: "m_Radar Incidence Angle",
            37: "m_Antenna Beam Azimuth",
            38: "m_Backscatter",
            39: "m_Radiometric Resolution (Noise Value)",
            40: "m_ASCAT KP Estimate Quality",
            41: "m_ASCAT Sigma-0 Usability",
            48: "m_ASCAT Land Fraction",
            49: "a_Beam Identifier",
            50: "a_Radar Incidence Angle",
            51: "a_Antenna Beam Azimuth",
            52: "a_Backscatter",
            53: "a_Radiometric Resolution (Noise Value)",
            54: "a_ASCAT KP Estimate Quality",
            55: "a_ASCAT Sigma-0 Usability",
            62: "a_ASCAT Land Fraction",
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
            # 75: "Rain Fall Detection",
            76: "Soil Moisture Correction Flag",
            77: "Soil Moisture Processing Flag",
            78: "Soil Moisture Quality",
            79: "Snow Cover",
            80: "Frozen Land Surface Fraction",
            81: "Inundation And Wetland Fraction",
            82: "Topographic Complexity",
        }

    def _read(self, filename, generic=False, to_xarray=False):
        """
        Read one ASCAT Level 2 BUFR file.

        Parameters
        ----------
        generic : bool, optional
            'True' reading and converting into generic format or
            'False' reading original field names(default: False).
        to_xarray : bool, optional
            'True' return data as xarray.Dataset
            'False' return data as numpy.ndarray(default: False).

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.
        """
        df = pdbufr.read_bufr(filename, columns="data", flat=True)

        col_rename = {}
        for i, col in enumerate(df.columns.to_list()):
            name = self.msg_name_lookup.get(i + 1, None)
            if name is not None:
                col_rename[col] = name

        data = df.rename(columns=col_rename)[col_rename.values()]

        data["lat"] = df["#1#latitude"].values.astype(np.float32)
        data["lon"] = df["#1#longitude"].values.astype(np.float32)

        year = df["#1#year"].values.astype(int)
        month = df["#1#month"].values.astype(int)
        day = df["#1#day"].values.astype(int)
        hour = df["#1#hour"].values.astype(int)
        minute = df["#1#minute"].values.astype(int)
        seconds = df["#1#second"].values.astype(int)
        milliseconds = np.zeros(seconds.size)
        cal_dates = np.vstack(
            (year, month, day, hour, minute, seconds, milliseconds)).T

        data['time'] = cal2dt(cal_dates)
        data = data.to_records(index=False)
        data = {name:data[name] for name in data.dtype.names}

        data["Mean Surface Soil Moisture"] *= 100.

        metadata = {}
        metadata['platform_id'] = data['Satellite Identifier'][0].astype(int)
        metadata['orbit_start'] = np.uint32(data['Orbit Number'][0])
        metadata['filename'] = os.path.basename(filename)

        # add/rename/remove fields according to generic format
        if generic:
            data = conv_bufrl2_generic(data, metadata)

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data.keys():
                if len(data[k].shape) == 1:
                    dim = ['obs']
                elif len(data[k].shape) == 2:
                    dim = ['obs', 'beam']

                data[k] = (dim, data[k])

            coords = {}
            coords_fields = ['lon', 'lat', 'time']
            for cf in coords_fields:
                coords[cf] = data.pop(cf)

            data = xr.Dataset(data, coords=coords, attrs=metadata)
            if generic:
                data = mask_dtype_nans(data)
        else:
            # collect dtype info
            dtype = []
            # fill_value = []

            for var_name in data.keys():

                if len(data[var_name].shape) == 1:
                    dtype.append((var_name, data[var_name].dtype.str))
                    # fill_value.append(data[var_name].fill_value)

                elif len(data[var_name].shape) > 1:
                    dtype.append((var_name, data[var_name].dtype.str,
                                  data[var_name].shape[1:]))
                    # fill_value.append(data[var_name].shape[1] *
                    #                   [data[var_name].fill_value])

            ds = np.ma.empty(data['time'].size, dtype=np.dtype(dtype))
            # fill_value_arr = np.array((*fill_value, ), dtype=np.dtype(dtype))

            for k, v in data.items():
                ds[k] = v

            # ds.fill_value = fill_value_arr
            data = ds

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
        data : numpy.ndarray or xarray.Dataset
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

class AscatL2BufrFileGeneric(AscatL2BufrFile):
    """
    The same as AscatL1bBufrFile but with generic=True by default.
    """
    def _read(self, filename, generic=True, to_xarray=False, **kwargs):
        return super()._read(filename, generic=generic, to_xarray=to_xarray, **kwargs)

def conv_bufrl2_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Spacecraft_id vs sat_id encoding

    BUFR encoding - Spacecraft_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-1 (Metop-B)
    - 4 Metop-2 (Metop-A)
    - 5 Metop-3 (Metop-C)

    Internal encoding - sat_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-2 (Metop-A)
    - 4 Metop-1 (Metop-B)
    - 5 Metop-3 (Metop-C)

    Parameters
    ----------
    data: dict of numpy.ndarray
        Original dataset.
    metadata: dict
        Metadata.

    Returns
    -------
    data: dict of numpy.ndarray
        Converted dataset.
    """
    skip_fields = ['Satellite Identifier']

    gen_fields_beam = {
        'Radar Incidence Angle': ('inc', np.float32),
        'Backscatter': ('sig', np.float32),
        'Antenna Beam Azimuth': ('azi', np.float32),
        'ASCAT Sigma-0 Usability': ('f_usable', np.uint8),
        'Beam Identifier': ('beam_num', np.uint8),
        'Radiometric Resolution (Noise Value)': ('kp_noise', np.float32),
        'ASCAT KP Estimate Quality': ('kp', np.float32),
        'ASCAT Land Fraction': ('f_land', np.float32)
    }

    gen_fields_lut = {
        'Orbit Number': ('abs_orbit_nr', np.int32),
        'Cross-Track Cell Number': ('node_num', np.uint8),
        'Direction Of Motion Of Moving Observing Platform':
        ('sat_track_azi', np.float32),
        'Surface Soil Moisture (Ms)': ('sm', np.float32),
        'Estimated Error In Surface Soil Moisture': ('sm_noise', np.float32),
        'Backscatter': ('sig40', np.float32),
        'Estimated Error In Sigma0 At 40 Deg Incidence Angle':
        ('sig40_noise', np.float32),
        'Slope At 40 Deg Incidence Angle': ('slope40', np.float32),
        'Estimated Error In Slope At 40 Deg Incidence Angle':
        ('slope40_noise', np.float32),
        'Soil Moisture Sensitivity': ('sm_sens', np.float32),
        'Dry Backscatter': ('dry_sig40', np.float32),
        'Wet Backscatter': ('wet_sig40', np.float32),
        'Mean Surface Soil Moisture': ('sm_mean', np.float32),
        # 'Rain Fall Detection': ('rf', np.float32),
        'Soil Moisture Correction Flag': ('corr_flag', np.uint8),
        'Soil Moisture Processing Flag': ('proc_flag', np.uint8),
        'Soil Moisture Quality': ('agg_flag', np.uint8),
        'Snow Cover': ('snow_prob', np.uint8),
        'Frozen Land Surface Fraction': ('frozen_prob', np.uint8),
        'Inundation And Wetland Fraction': ('wetland', np.uint8),
        'Topographic Complexity': ('topo', np.uint8)
    }

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name, (new_name, new_dtype) in gen_fields_lut.items():
        mask = (data[var_name] == bufr_nan) |  (np.isnan(data[var_name]))
        data[var_name][mask] = nan_val_dict[new_dtype]

        data[new_name] = np.ma.array(data.pop(var_name).astype(new_dtype), mask=mask)
        data[new_name].fill_value = nan_val_dict[new_dtype]

    for var_name, (new_name, new_dtype) in gen_fields_beam.items():

        f = ['{}_{}'.format(b, var_name) for b in ['f', 'm', 'a']]

        mask = np.vstack((data[f[0]] == bufr_nan, data[f[1]] == bufr_nan,
                          data[f[2]] == bufr_nan)).T

        data[new_name] = np.ma.vstack((data.pop(f[0]), data.pop(f[1]),
                                       data.pop(f[2]))).T.astype(new_dtype)

        data[new_name].mask = mask
        data[new_name][mask] = nan_val_dict[new_dtype]

        data[new_name].fill_value = nan_val_dict[new_dtype]

    if data['node_num'].max() == 82:
        data['swath_indicator'] = np.ma.array(1 * (data['node_num'] > 41),
                                              dtype=np.uint8,
                                              mask=data['node_num'] > 82)
    elif data['node_num'].max() == 42:
        data['swath_indicator'] = np.ma.array(1 * (data['node_num'] > 21),
                                              dtype=np.uint8,
                                              mask=data['node_num'] > 42)
    else:
        raise ValueError('Cross-track cell number size unknown')

    n_lines = data['lat'].shape[0] / data['node_num'].max()
    line_num = np.arange(n_lines).repeat(data['node_num'].max())
    data['line_num'] = np.ma.array(line_num,
                                   dtype=np.uint16,
                                   mask=np.zeros_like(line_num),
                                   fill_value=uint16_nan)

    sat_id = np.ma.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
    data['sat_id'] = np.ma.zeros(data['time'].size,
                                 dtype=np.uint8) + sat_id[int(
                                     metadata['platform_id'])]
    data['sat_id'].mask = np.zeros(data['time'].size)
    data['sat_id'].fill_value = uint8_nan

    # compute ascending/descending direction
    data['as_des_pass'] = np.ma.array(data['sat_track_azi'] < 270,
                                      dtype=np.uint8,
                                      mask=np.zeros(data['time'].size),
                                      fill_value=uint8_nan)

    mask = data['lat'] == bufr_nan
    data['lat'] = np.ma.array(data['lat'], mask=mask, fill_value=float32_nan)

    mask = data['lon'] == bufr_nan
    data['lon'] = np.ma.array(data['lon'], mask=mask, fill_value=float32_nan)

    data['time'] = np.ma.array(data['time'], mask=mask, fill_value=0)

    return data
