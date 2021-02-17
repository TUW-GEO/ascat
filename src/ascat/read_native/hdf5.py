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


class AscatL1bHdf5File:

    """
    Read ASCAT Level 1b file in HDF5 format.
    """

    def __init__(self, filename, mode='r'):
        """
        Initialize AscatL1bHdf5File.

        Parameters
        ----------
        filename : str
            Filename.
        mode : str, optional
            File mode (default: 'r')
        """
        self.filename = filename
        self.mode = mode

    def read(self):
        """
        Read ASCAT Level 1b data.

        Returns
        -------
        ds : dict of xarray.Dataset
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

        # convert spacecraft_id to internal sat_id
        sat_id = np.array([4, 3, 5])
        metadata['sat_id'] = sat_id[metadata['spacecraft_id']-1]

        # compute ascending/descending direction
        raw_data['as_des_pass'] = (
            raw_data['sat_track_azi'] < 270).astype(np.uint8)

        # modify longitudes [0, 360] to [-180, 180]
        mask = raw_data['longitude_full'] > 180
        raw_data['longitude_full'][mask] += -360.

        # modify azimuth angles to [0, 360]
        if 'azi_angle_full' in var_names:
            mask = raw_data['azi_angle_full'] < 0
            raw_data['azi_angle_full'][mask] += 360

        raw_data['time'] = np.datetime64('2000-01-01') + raw_data[
            'utc_localisation-days'].astype('timedelta64[D]') + raw_data[
                'utc_localisation-milliseconds'].astype('timedelta64[ms]')

        raw_data = set_flags(raw_data)
        raw_data['f_usable'] = raw_data['f_usable'].reshape(-1, 192)
        raw_data['f_land'] = raw_data['f_land'].reshape(-1, 192)

        skip_fields = ['utc_localisation-days',
                       'utc_localisation-milliseconds',
                       'degraded_inst_mdr', 'degraded_proc_mdr',
                       'beam_number', 'sat_track_azi', 'flagfield_rf1',
                       'flagfield_rf2', 'flagfield_pl', 'flagfield_gen1',
                       'flagfield_gen2']

        rename_fields = {'inc_angle_full': 'inc', 'azi_angle_full': 'azi',
                         'sigma0_full': 'sig'}

        # 1 Left Fore Antenna, 2 Left Mid Antenna 3 Left Aft Antenna
        # 4 Right Fore Antenna, 5 Right Mid Antenna, 6 Right Aft Antenna
        antennas = ['lf', 'lm', 'la', 'rf', 'rm', 'ra']
        ds = OrderedDict()
        for i, antenna in enumerate(antennas):
            data_var = {}
            subset = ((raw_data['beam_number']) == i+1)
            for k, v in raw_data.items():

                if k in skip_fields:
                    continue

                if len(v.shape) == 1:
                    dim = ['obs']
                elif len(v.shape) == 2:
                    dim = ['obs', 'echo']
                else:
                    raise RuntimeError('Wrong number of dimensions')

                if k in rename_fields:
                    name = rename_fields[k]
                else:
                    name = k

                data_var[name] = (dim, v[subset])

            coords = {"lon": (['obs', 'echo'],
                              data_var.pop('longitude_full')[1]),
                      "lat": (['obs', 'echo'],
                              data_var.pop('latitude_full')[1]),
                      "time": (['obs'], data_var.pop('time')[1])}

            metadata['beam_number'] = i+1
            metadata['beam_name'] = antenna
            ds[antenna] = xr.Dataset(data_var, coords=coords, attrs=metadata)

        return ds

    def close(self):
        """
        Close file.
        """
        pass
