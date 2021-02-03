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
Readers for ASCAT Level 1b and Level 2 data in BUFR format.
"""

import os
import warnings
from tempfile import NamedTemporaryFile
from gzip import GzipFile
from collections import defaultdict

import numpy as np
import xarray as xr
from cadati.cal_date import cal2dt

from ascat.utils import get_toi_subset, get_roi_subset

try:
    from pybufr_ecmwf import raw_bufr_file
    from pybufr_ecmwf import ecmwfbufr
    from pybufr_ecmwf import ecmwfbufr_parameters
except ImportError:
    warnings.warn(
        'pybufr-ecmwf can not be imported, BUFR data cannot be read.')


def tmp_unzip(filename):
    """
    Unzip file to temporary directory.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    unzipped_filename : str
        Unzipped filename
    """
    with NamedTemporaryFile(delete=False) as tmp_fid:
        with GzipFile(filename) as gz_fid:
            tmp_fid.write(gz_fid.read())
        unzipped_filename = tmp_fid.name

    return unzipped_filename


class AscatL1BufrFile():

    """
    Read ASCAT Level 1 file in BUFR format.
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None):
        """
        Initialization of i/o object.

        Parameters
        ----------
        filename : str
            Filename.
        mode : str, optional
            Opening mode (default: 'r').
        msg_name_lookup: dict, optional
            Dictionary mapping bufr msg number to parameter name.
            See :ref:`ascatformattable`.
        """
        if os.path.splitext(filename)[1] == '.gz':
            self.filename = tmp_unzip(filename)
        else:
            self.filename = filename

        self.mode = mode

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
        ds : xarray.Dataset
            ASCAT Level 1 data.
        """
        data_var = defaultdict(list)
        dates = []
        latitude = []
        longitude = []

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
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)
        n_records = latitude.shape[0]

        for num, name in self.msg_name_lookup.items():
            if name not in data_var:
                continue

            arr = np.concatenate(data_var[name])

            if len(arr.shape) == 1:
                dim = ['obs']
            elif len(arr.shape) == 2:
                dim = ['obs', '2']

            if 'Direction Of Motion Of Moving Observing Platform' == name:
                data_var['as_des_pass'] = (dim, (arr < 270).astype(np.uint8))

            if 'Cross-Track Cell Number' == name:
                if arr.max() == 82:
                    val = 41
                elif arr.max() == 42:
                    val = 21
                else:
                    raise ValueError('Unsuspected node number')

                data_var['swath_indicator'] = (dim, 1 * (arr > val))

                n_lines = n_records / max(arr)
                data_var['line_num'] = (
                    dim, np.arange(n_lines).repeat(arr.max()))

            data_var[name] = (dim, arr)

        # 1 ERS-1
        # 2 ERS-2
        # 3 Metop-1(B)
        # 4 Metop-2(A)
        # 5 Metop-3(C)
        sat_id = {3: 1, 4: 2, 5: 3}

        metadata = {}
        metadata['spacecraft_id'] = np.int8(
            sat_id[data_var.pop('Satellite Identifier')[1][0]])
        metadata['orbit_start'] = np.uint32(data_var['Orbit Number'][1][0])

        coords = {"lon": longitude, "lat": latitude, "time": dates}

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


class AscatL2BufrFile():

    """
    Read ASCAT Level 2 file in BUFR format.
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None):
        """
        Initialization of i/o object.

        Parameters
        ----------
        filename : str
            Filename path.
        mode : str, optional
            Opening mode. Default: r
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

    def read(self, toi=None, roi=None):
        """
        Read ASCAT Level 2 data.

        Parameters
        ----------
        toi : tuple of datetime, optional
            Filter data for given time of interest (default: None).
        roi : tuple of 4 float, optional
            Filter data for region of interest (default: None).
            e.g. latmin, lonmin, latmax, lonmax

        Returns
        -------
        ds : xarray.Dataset
            ASCAT Level 2 data.
        """
        data_var = defaultdict(list)
        dates = []
        latitude = []
        longitude = []

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
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)
        n_records = latitude.shape[0]

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

            if 'Direction Of Motion Of Moving Observing Platform' == name:
                data_var['as_des_pass'] = (dim, (arr < 270).astype(np.uint8))

            if 'Cross-Track Cell Number' == name:
                if arr.max() == 82:
                    val = 41
                elif arr.max() == 42:
                    val = 21
                else:
                    raise ValueError('Unsuspected node number')

                data_var['swath_indicator'] = (dim, 1 * (arr > val))

                n_lines = n_records / max(arr)
                data_var['line_num'] = (
                    dim, np.arange(n_lines).repeat(arr.max()))

            data_var[name] = (dim, arr)

        # 1 ERS-1
        # 2 ERS-2
        # 3 METOP-1 (Metop-B)
        # 4 METOP-2 (Metop-A)
        # 5 METOP-3 (Metop-C)
        sat_id = {3: 1, 4: 2, 5: 3}

        metadata = {}
        metadata['spacecraft_id'] = np.int8(
            sat_id[data_var.pop('Satellite Identifier')[1][0]])
        metadata['orbit_start'] = np.uint32(data_var['Orbit Number'][1][0])

        coords = {"lon": (['obs'], longitude), "lat": (['obs'], latitude),
                  "time": ('obs', dates)}

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

        if toi:
            ds = get_toi_subset(ds, toi)

        if roi:
            ds = get_roi_subset(ds, roi)

        return ds

    def close(self):
        pass


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
        Raises
        ------
        IOError:
            if a message cannot be unpacked after max_tries tries

        Returns
        -------
        data : yield results of messages
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
            # try to expand bufr message with the first guess for
            # kelem
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
