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
Readers for ASCAT Level 1b and Level 2 data in BUFR format.
"""

import os
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr
from cadati.cal_date import cal2dt

from ascat.utils import tmp_unzip

try:
    from pybufr_ecmwf import raw_bufr_file
    from pybufr_ecmwf import ecmwfbufr
    from pybufr_ecmwf import ecmwfbufr_parameters
except ImportError:
    warnings.warn(
        'pybufr-ecmwf can not be imported, BUFR data cannot be read.')


class AscatL1bBufrFile():

    """
    Read ASCAT Level 1b file in BUFR format.
    """

    def __init__(self, filename, msg_name_lookup=None):
        """
        Initialize AscatL1bBufrFile.

        Parameters
        ----------
        filename : str
            Filename.
        msg_name_lookup: dict, optional
            Dictionary mapping bufr msg number to parameter name.
            See :ref:`ascatformattable`.
        """
        if os.path.splitext(filename)[1] == '.gz':
            self.filename = tmp_unzip(filename)
        else:
            self.filename = filename

        if msg_name_lookup is None:
            msg_name_lookup = {
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
                62: "a_ASCAT Land Fraction"}

        self.msg_name_lookup = msg_name_lookup

    def read(self, generic=False, to_xarray=False):
        """
        Read ASCAT Level 1 data.

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
        ds : xarray.Dataset
            ASCAT Level 1 data.
        """
        skip_fields = ['Satellite Identifier']

        gen_fields_beam = [
            ('Radar Incidence Angle', 'inc', np.float32),
            ('Backscatter', 'sig', np.float32),
            ('Antenna Beam Azimuth', 'azi', np.float32),
            ('ASCAT Sigma-0 Usability', 'f_usable', np.uint8),
            ('Beam Identifier', 'beam_num', np.uint8),
            ('Radiometric Resolution (Noise Value)', 'kp_noise', np.float32),
            ('ASCAT KP Estimate Quality', 'kp', np.float32),
            ('ASCAT Land Fraction', 'f_land', np.float32)]

        gen_fields_lut = {
            'Orbit Number': ('abs_orbit_nr', np.int32),
            'Cross-Track Cell Number': ('ctcn', np.uint8),
            'Direction Of Motion Of Moving Observing Platform':
            ('sat_azi_track', np.float32)}

        data_var = defaultdict(list)

        with BUFRReader(self.filename) as bufr:
            for message in bufr.messages():
                # read lon/lat
                data_var['lat'].append(message[:, 12])
                data_var['lon'].append(message[:, 13])

                # read time
                year = message[:, 6].astype(int)
                month = message[:, 7].astype(int)
                day = message[:, 8].astype(int)
                hour = message[:, 9].astype(int)
                minute = message[:, 10].astype(int)
                seconds = message[:, 11].astype(int)
                milliseconds = np.zeros(seconds.size)
                cal_dates = np.vstack((
                    year, month, day, hour, minute, seconds, milliseconds)).T
                data_var['time'].append(cal2dt(cal_dates))

                # read data fields
                for num, var_name in self.msg_name_lookup.items():
                    data_var[var_name].append(message[:, num-1])

        # concatenate lists to array
        for var_name in data_var.keys():
            data_var[var_name] = np.concatenate(data_var[var_name])

        # define metadata
        metadata = {}
        metadata['platform_id'] = data_var[
            'Satellite Identifier'][0].astype(int)
        metadata['orbit_start'] = np.uint32(data_var['Orbit Number'][0])
        metadata['filename'] = os.path.basename(self.filename)

        num_records = data_var['time'].size

        # add/rename/remove fields according to generic format
        if generic:
            for var_name in skip_fields:
                if var_name in data_var:
                    data_var.pop(var_name)

            for var_name in data_var.keys():

                if var_name in gen_fields_lut:
                    new_name = gen_fields_lut[var_name][0]
                    new_dtype = gen_fields_lut[var_name][1]
                    data_var[new_name] = data_var.pop(
                        var_name).astype(new_dtype)

            for var_name, new_name, new_dtype in gen_fields_beam:
                f = ['{}_{}'.format(b, var_name) for b in ['f', 'm', 'a']]
                data_var[new_name] = np.vstack(
                    (data_var.pop(f[0]), data_var.pop(f[1]),
                     data_var.pop(f[2]))).T.astype(new_dtype)

            sat_id = np.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
            data_var['sat_id'] = np.zeros(
                num_records, dtype=np.uint8) + sat_id[
                int(metadata['platform_id'])]

        # collect dtype info
        dtype = []
        for var_name in data_var.keys():
            if len(data_var[var_name].shape) == 1:
                dtype.append((var_name, data_var[var_name].dtype.str))
            elif len(data_var[var_name].shape) > 1:
                dtype.append((var_name, data_var[var_name].dtype.str,
                              data_var[var_name].shape[1:]))

        coords_fields = ['lon', 'lat', 'time']

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data_var.keys():
                if len(data_var[k].shape) == 1:
                    dim = ['obs']
                elif len(data_var[k].shape) == 2:
                    dim = ['obs', 'beam']

                data_var[k] = (dim, data_var[k])

            coords = {}
            for cf in coords_fields:
                coords[cf] = data_var.pop(cf)

            ds = xr.Dataset(data_var, coords=coords, attrs=metadata)
        else:
            ds = np.empty(num_records, dtype=np.dtype(dtype))
            for k, v in data_var.items():
                ds[k] = v

        # There can be suspicious values (32.32) instead of normal nan_values
        # Since some elements rly have this value we check the other triplet
        # data of that beam to filter the nan_values out
        for beam in ['f', 'm', 'a']:
            azi = beam + '_Antenna Beam Azimuth'
            sig = beam + '_Backscatter'
            inc = beam + '_Radar Incidence Angle'
            if azi in ds:
                mask_azi = ds[azi] == 32.32
                mask_sig = ds[sig] == 1.7e+38
                mask_inc = ds[inc] == 1.7e+38
                mask = np.all([mask_azi, mask_sig, mask_inc], axis=0)
                ds[azi][mask] = 1.7e+38

        return ds

    def close(self):
        """
        Close file.
        """
        pass


def conv_bufrl1b_generic(ds):
    """
    Rename and convert data types of dataset.

    Spacecraft_id vs sat_id encoding

    BUFR encoding - Spacecraft_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 METOP-1 (Metop-B)
    - 4 METOP-2 (Metop-A)
    - 5 METOP-3 (Metop-C)

    Internal encoding - sat_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 METOP-2 (Metop-A)
    - 4 METOP-1 (Metop-B)
    - 5 METOP-3 (Metop-C)

    Parameters
    ----------
    ds : xarray.Dataset
        Original dataset.

    Returns
    -------
    ds : xarray.Dataset
        Converted dataset.
    """
    obs_dim = ['obs']

    new_names = [('Orbit Number', 'abs_orbit_nr', np.int32),
                 ('Cross-Track Cell Number', 'ctcn', np.uint8),
                 ('Direction Of Motion Of Moving Observing Platform',
                  'sat_azi_track', np.float32)]

    for old_name, new_name, dtype in new_names:
        ds[new_name] = (['obs'], ds[old_name].astype(dtype))
        ds = ds.drop_vars([old_name])

    fields = [('inc', 'Radar Incidence Angle', np.float32),
              ('sig', 'Backscatter', np.float32),
              ('azi', 'Antenna Beam Azimuth', np.float32),
              ('f_usable', 'ASCAT Sigma-0 Usability', np.uint8),
              ('beam_num', 'Beam Identifier', np.uint8),
              ('kp_noise', 'Radiometric Resolution (Noise Value)', np.float32),
              ('kp', 'ASCAT KP Estimate Quality', np.float32),
              ('f_land', 'ASCAT Land Fraction', np.float32)]

    for name, bufr_name, dtype in fields:
        all_fields = ['{}_{}'.format(b, bufr_name) for b in ['f', 'm', 'a']]
        ds[name] = (['obs', 'beam'], ds[
            all_fields].to_array().data.T.astype(dtype))
        ds = ds.drop_vars(all_fields)

    ds['as_des_pass'] = (obs_dim, (ds['sat_azi_track'] < 270).astype(np.uint8))

    if ds['ctcn'].max() == 82:
        val = 41
    elif ds['ctcn'].max() == 42:
        val = 21
    else:
        raise ValueError('Unsuspected node number')

    ds['swath_indicator'] = (obs_dim, (ds['ctcn'] > val).astype(np.uint8))

    sat_id = np.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
    ds['sat_id'] = (
        obs_dim, sat_id[ds['Satellite Identifier'].data.astype(int)])
    ds = ds.drop_vars(['Satellite Identifier'])

    return ds


class AscatL2BufrFile():

    """
    Read ASCAT Level 2 file in BUFR format.
    """

    def __init__(self, filename, msg_name_lookup=None):
        """
        Initialize AscatL2BufrFile.

        Parameters
        ----------
        filename : str
            Filename path.
        msg_name_lookup: dict, optional
            Dictionary mapping bufr msg number to parameter name.
            See :ref:`ascatformattable`.
        """
        if os.path.splitext(filename)[1] == '.gz':
            self.filename = tmp_unzip(filename)
        else:
            self.filename = filename

        if msg_name_lookup is None:
            msg_name_lookup = {
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
                75: "Rain Fall Detection",
                76: "Soil Moisture Correction Flag",
                77: "Soil Moisture Processing Flag",
                78: "Soil Moisture Quality",
                79: "Snow Cover",
                80: "Frozen Land Surface Fraction",
                81: "Inundation And Wetland Fraction",
                82: "Topographic Complexity"}

        self.msg_name_lookup = msg_name_lookup

    def read(self):
        """
        Read ASCAT Level 2 data.

        Returns
        -------
        ds : xarray.Dataset
            ASCAT Level 2 data.
        """
        dates = []
        latitude = []
        longitude = []

        data_var = defaultdict(list)

        with BUFRReader(self.filename) as bufr:
            for message in bufr.messages():

                # read lon/lat
                latitude.append(message[:, 12])
                longitude.append(message[:, 13])

                # read time
                year = message[:, 6].astype(int)
                month = message[:, 7].astype(int)
                day = message[:, 8].astype(int)
                hour = message[:, 9].astype(int)
                minute = message[:, 10].astype(int)
                seconds = message[:, 11].astype(int)
                milliseconds = np.zeros(seconds.size)
                cal_dates = np.vstack((year, month, day,
                                       hour, minute, seconds, milliseconds)).T
                dates.append(cal2dt(cal_dates))

                # read data fields
                for num, name in self.msg_name_lookup.items():
                    data_var[name].append(message[:, num - 1])

        dates = np.concatenate(dates)
        latitude = np.concatenate(latitude).astype(np.float32)
        longitude = np.concatenate(longitude).astype(np.float32)

        for num, name in self.msg_name_lookup.items():
            if name not in data_var:
                continue

            arr = np.concatenate(data_var[name])

            if num == 74:
                # ssm mean is encoded differently
                valid = arr != 1.7e+38
                arr[valid] = arr[valid] * 100

            if len(arr.shape) == 1:
                dim = ['obs']
            else:
                raise RuntimeError('Unsuspected dimension shape')

            data_var[name] = (dim, arr)

        metadata = {}
        metadata['spacecraft_id'] = data_var[
            'Satellite Identifier'][1][0].astype(int)
        metadata['orbit_start'] = np.uint32(data_var['Orbit Number'][1][0])
        metadata['filename'] = os.path.basename(self.filename)

        coords = {"lon": (['obs'], longitude), "lat": (['obs'], latitude),
                  "time": (['obs'], dates)}

        ds = xr.Dataset(data_var, coords=coords, attrs=metadata)

        # There can be suspicious values (32.32) instead of normal nan_values
        # Since some elements rly have this value we check the other triplet
        # data of that beam to filter the nan_values out
        for beam in ['f', 'm', 'a']:
            azi = beam + '_Antenna Beam Azimuth'
            sig = beam + '_Backscatter'
            inc = beam + '_Radar Incidence Angle'
            if azi in ds:
                mask_azi = ds[azi] == 32.32
                mask_sig = ds[sig] == 1.7e+38
                mask_inc = ds[inc] == 1.7e+38
                mask = np.all([mask_azi, mask_sig, mask_inc], axis=0)
                ds[azi][mask] = 1.7e+38

        if self.generic:
            ds = conv_bufrl2_generic(ds)

        return ds

    def close(self):
        pass


def conv_bufrl2_generic(ds):
    """
    Rename and convert data types of dataset.

    Spacecraft_id vs sat_id encoding

    BUFR encoding - Spacecraft_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 METOP-1 (Metop-B)
    - 4 METOP-2 (Metop-A)
    - 5 METOP-3 (Metop-C)

    Internal encoding - sat_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 METOP-2 (Metop-A)
    - 4 METOP-1 (Metop-B)
    - 5 METOP-3 (Metop-C)

    Parameters
    ----------
    ds : xarray.Dataset
        Original dataset.

    Returns
    -------
    ds : xarray.Dataset
        Converted dataset.
    """
    obs_dim = ['obs']

    new_names = [('Orbit Number', 'abs_orbit_nr', np.int32),
                 ('Cross-Track Cell Number', 'ctcn', np.uint8),
                 ('Direction Of Motion Of Moving Observing Platform',
                  'sat_azi_track', np.float32),
                 ('Surface Soil Moisture (Ms)', 'sm', np.float32),
                 ('Estimated Error In Surface Soil Moisture', 'sm_noise', np.float32),
                 ('Backscatter', 'sigma40', np.float32),
                 ('Estimated Error In Sigma0 At 40 Deg Incidence Angle',
                  'sigma40_noise', np.float32),
                 ('Slope At 40 Deg Incidence Angle', 'slope40', np.float32),
                 ('Estimated Error In Slope At 40 Deg Incidence Angle',
                  'slope40_noise', np.float32),
                 ('Soil Moisture Sensitivity', 'sens', np.float32),
                 ('Dry Backscatter', 'dry', np.float32),
                 ('Wet Backscatter', 'wet', np.float32),
                 ('Mean Surface Soil Moisture', 'sm_mean', np.float32),
                 ('Rain Fall Detection', 'rf', np.float32),
                 ('Soil Moisture Correction Flag', 'corr_flag', np.uint8),
                 ('Soil Moisture Processing Flag', 'proc_flag', np.uint8),
                 ('Soil Moisture Quality', 'quality_flag', np.uint8),
                 ('Snow Cover', 'snow_cover', np.uint8),
                 ('Frozen Land Surface Fraction', 'frozen_soil', np.uint8),
                 ('Inundation And Wetland Fraction', 'wetland', np.uint8),
                 ('Topographic Complexity', 'topo', np.uint8)]

    for old_name, new_name, dtype in new_names:
        ds[new_name] = (['obs'], ds[old_name].astype(dtype))
        ds = ds.drop_vars([old_name])

    fields = [('inc', 'Radar Incidence Angle', np.float32),
              ('sig', 'Backscatter', np.float32),
              ('azi', 'Antenna Beam Azimuth', np.float32),
              ('f_usable', 'ASCAT Sigma-0 Usability', np.uint8),
              ('beam_num', 'Beam Identifier', np.uint8),
              ('kp_noise', 'Radiometric Resolution (Noise Value)', np.float32),
              ('kp', 'ASCAT KP Estimate Quality', np.float32),
              ('f_land', 'ASCAT Land Fraction', np.float32)]

    for name, bufr_name, dtype in fields:
        all_fields = ['{}_{}'.format(b, bufr_name) for b in ['f', 'm', 'a']]
        ds[name] = (['obs', 'beam'], ds[
            all_fields].to_array().data.T.astype(dtype))
        ds = ds.drop_vars(all_fields)

    ds['as_des_pass'] = (obs_dim, (ds['sat_azi_track'] < 270).astype(np.uint8))

    if ds['ctcn'].max() == 82:
        val = 41
    elif ds['ctcn'].max() == 42:
        val = 21
    else:
        raise ValueError('Unsuspected node number')

    ds['swath_indicator'] = (obs_dim, (ds['ctcn'] > val).astype(np.uint8))

    sat_id = np.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
    ds['sat_id'] = (
        obs_dim, sat_id[ds['Satellite Identifier'].data.astype(int)])
    ds = ds.drop_vars(['Satellite Identifier'])

    return ds


class BUFRReader():

    """
    BUFR reader based on the pybufr-ecmwf package but faster.
    """

    def __init__(self, filename, kelem_guess=500, max_tries=10):
        """

        Parameters
        ----------
        filename : string
            filename of the bufr file
        kelem_guess : int, optional
            if the elements per variable in as message are known
            please specify here.
            Otherwise the elements will be found out via trial and error
            This works most of the time but is not 100 percent failsafe
            Default: 500
        max_tries : int, optional
            the Reader will try max_tries times to unpack a bufr message.
            Some messages can not be read even if the array sizes are ok.
            Most of the time these files are corrupt.
        """
        self.bufr = raw_bufr_file.RawBUFRFile()
        self.bufr.open(filename, 'rb')
        self.nr_messages = self.bufr.get_num_bufr_msgs()
        self.max_tries = max_tries

        if 'BUFR_TABLES' not in os.environ:
            path = os.path.split(ecmwfbufr.__file__)[0]
            os.environ["BUFR_TABLES"] = os.path.join(
                path, 'ecmwf_bufrtables' + os.sep)
            # os.environ['PRINT_TABLE_NAMES'] = "false"

        self.size_ksup = ecmwfbufr_parameters.JSUP
        self.size_ksec0 = ecmwfbufr_parameters.JSEC0
        self.size_ksec1 = ecmwfbufr_parameters.JSEC1
        self.size_ksec2 = ecmwfbufr_parameters.JSEC2
        self.size_key = ecmwfbufr_parameters.JKEY
        self.size_ksec3 = ecmwfbufr_parameters.JSEC3
        self.size_ksec4 = ecmwfbufr_parameters.JSEC4

        self.kelem_guess = kelem_guess

    def messages(self):
        """
        Read messages.

        Raises
        ------
        IOError :
            if a message cannot be unpacked after max_tries tries

        Returns
        -------
        data : numpy.ndarray
            Results of messages
        """
        count = 0
        for i in np.arange(self.nr_messages) + 1:
            tries = 0

            ksup = np.zeros(self.size_ksup, dtype=np.int)
            ksec0 = np.zeros(self.size_ksec0, dtype=np.int)
            ksec1 = np.zeros(self.size_ksec1, dtype=np.int)
            ksec2 = np.zeros(self.size_ksec2, dtype=np.int)
            ksec3 = np.zeros(self.size_ksec3, dtype=np.int)
            ksec4 = np.zeros(self.size_ksec4, dtype=np.int)

            kerr = 0
            data = self.bufr.get_raw_bufr_msg(i)

            ecmwfbufr.bus012(data[0],  # input
                             ksup,  # output
                             ksec0,  # output
                             ksec1,  # output
                             ksec2,  # output
                             kerr)  # output

            kelem = self.kelem_guess
            ksup_first = ksup[5]
            kvals = ksup_first * kelem
            max_kelem = 500000
            self.init_values = np.zeros(kvals, dtype=np.float64)
            self.cvals = np.zeros((kvals, 80), dtype='S1')

            # try to expand bufr message with the first guess for kelem
            increment_arraysize = True
            while increment_arraysize:
                cnames = np.zeros((kelem, 64), dtype='|S1')
                cunits = np.zeros((kelem, 24), dtype='|S1')

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=DeprecationWarning)
                    ecmwfbufr.bufrex(data[0],  # input
                                     ksup,  # output
                                     ksec0,  # output
                                     ksec1,  # output
                                     ksec2,  # output
                                     ksec3,  # output
                                     ksec4,  # output
                                     cnames,  # output
                                     cunits,  # output
                                     self.init_values,  # output
                                     self.cvals,  # output
                                     kerr)  # output
                # no error - stop loop
                if kerr == 0 and ksec4[0] != 0:
                    increment_arraysize = False
                # error increase array size and try to unpack again
                else:
                    tries += 1
                    if tries >= self.max_tries:
                        raise IOError('This file seems corrupt')
                    kelem = kelem * 5
                    kvals = ksup_first * kelem

                    if kelem > max_kelem:
                        kelem = kvals / 2
                        max_kelem = kvals

                    self.init_values = np.zeros(kvals, dtype=np.float64)
                    self.cvals = np.zeros((kvals, 80), dtype=np.character)

            decoded_values = ksup[4]
            # set kelem_guess to decoded values of last message
            # only increases reading speed if all messages are the same
            # not sure if this is the best option
            self.kelem_guess = decoded_values
            decoded_msg = ksup[5]
            # calculate first dimension of 2D array
            factor = int(kvals / kelem)

            # reshape and trim the array to the actual size of the data
            values = self.init_values.reshape((factor, kelem))
            values = values[:decoded_msg, :decoded_values]
            count += values.shape[0]

            yield values

    def __enter__(self):
        return self

    def __exit__(self, exc, val, trace):
        self.bufr.close()
