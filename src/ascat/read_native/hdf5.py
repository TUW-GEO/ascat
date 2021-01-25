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
Readers for ASCAT Level 1b in HDF5 format.
"""

from collections import OrderedDict

import h5py
import numpy as np
import xarray as xr

from ascat.read_native.eps_native import set_flags


class AscatL1Hdf5File:
    """
    Read ASCAT Level 1 file in HDF5 format.
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
        raw_data = {}
        metadata = {}

        root = 'U-MARF/EPS/ASCA_SZF_1B/'
        mdr_path = root + 'DATA/MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'
        mdr_descr_path = root + 'DATA/MDR_1B_FULL_ASCA_Level_1_DESCR'
        metadata_path = root + 'METADATA'

        with h5py.File(self.filename, mode='r') as fid:
            mdr = fid[mdr_path]
            mdr_descr = fid[mdr_descr_path]
            mdr_metadata = fid[metadata_path]
            var_names = list(mdr.dtype.names)

            for var_name in var_names:
                raw_data[var_name.lower()] = mdr[var_name]

                if var_name.encode() in mdr_descr['EntryName']:
                    pos = mdr_descr['EntryName'] == var_name.encode()
                    scale = mdr_descr['Scale Factor'][pos][0].decode()
                    if scale != 'n/a':
                        raw_data[var_name.lower()] = raw_data[
                            var_name.lower()] / (10 ** float(scale))

            fields = ['SPACECRAFT_ID', 'ORBIT_START',
                      'PROCESSOR_MAJOR_VERSION', 'PROCESSOR_MINOR_VERSION',
                      'FORMAT_MAJOR_VERSION', 'FORMAT_MINOR_VERSION']

            for f in fields:
                pos = np.core.defchararray.startswith(
                    mdr_metadata['MPHR/MPHR_TABLE']['EntryName'], f.encode())
                var = mdr_metadata['MPHR/MPHR_TABLE']['EntryValue'][
                    pos][0].decode()

                if f == 'SPACECRAFT_ID':
                    var = var[-1]

                metadata[f.lower()] = int(var)

        raw_data['as_des_pass'] = (
            raw_data['sat_track_azi'] < 270).astype(np.uint8)

        # modify longitudes ([0,360] to [-180,180])
        mask = raw_data['longitude_full'] > 180
        raw_data['longitude_full'][mask] += -360.

        # modify azimuth angles
        if 'azi_angle_full' in var_names:
            mask = raw_data['azi_angle_full'] < 0
            raw_data['azi_angle_full'][mask] += 360

        raw_data['time'] = np.datetime64('2000-01-01') + raw_data[
            'utc_localisation-days'].astype('timedelta64[D]') + raw_data[
                'utc_localisation-milliseconds'].astype('timedelta64[ms]')

        set_flags(raw_data)

        # 1 Left Fore Antenna, 2 Left Mid Antenna 3 Left Aft Antenna
        # 4 Right Fore Antenna, 5 Right Mid Antenna, 6 Right Aft Antenna
        antennas = ['lf', 'lm', 'la', 'rf', 'rm', 'ra']
        ds = OrderedDict()
        for i, antenna in enumerate(antennas):
            data_var = {}
            subset = ((raw_data['beam_number']) == i+1)

            for k, v in raw_data.items():
                print(k, v.shape)
                if v.shape.size == 1:
                    dim = ['obs']
                elif v.shape.size == 2:
                    dim = ['obs', 'asdf']
                data_var[k] = (['obs'], v[subset])

            coords = {"lon": (['obs'], data_var.pop('longitude_full')[1]),
                      "lat": (['obs'], data_var.pop('latitude_full')[1]),
                      "time": (['obs'], data_var.pop('time')[1])}

            ds[antenna] = xr.Dataset(data_var, coords=coords, attrs=metadata)

        return ds

    def close(self):
        """
        Close file.
        """
        pass
