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

    def __init__(self, filename):
        """
        Initialize AscatL1bHdf5File.

        Parameters
        ----------
        filename : str
            Filename.
        """
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
        data = {}
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
                data[var_name.lower()] = mdr[var_name]

                if var_name.encode() in mdr_descr['EntryName']:
                    pos = mdr_descr['EntryName'] == var_name.encode()
                    scale = mdr_descr['Scale Factor'][pos][0].decode()

                    if scale != 'n/a':
                        data[var_name.lower()] = (data[
                            var_name.lower()] / (10. ** float(scale))).astype(
                                np.float32)

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

        # modify longitudes [0, 360] to [-180, 180]
        mask = data['longitude_full'] > 180
        data['longitude_full'][mask] += -360.

        data['time'] = np.datetime64('2000-01-01') + data[
            'utc_localisation-days'].astype('timedelta64[D]') + data[
                'utc_localisation-milliseconds'].astype('timedelta64[ms]')

        # modify azimuth angles to [0, 360]
        if 'azi_angle_full' in var_names:
            mask = data['azi_angle_full'] < 0
            data['azi_angle_full'][mask] += 360

        rename_coords = {'longitude_full': ('lon', np.float32),
                         'latitude_full': ('lat', np.float32)}

        for var_name, (new_name, new_dtype) in rename_coords.items():
            data[new_name] = data.pop(var_name).astype(new_dtype)

        if generic:
            data = conv_hdf5l1b_generic(data, metadata)

        # 1 Left Fore Antenna, 2 Left Mid Antenna 3 Left Aft Antenna
        # 4 Right Fore Antenna, 5 Right Mid Antenna, 6 Right Aft Antenna
        antennas = ['lf', 'lm', 'la', 'rf', 'rm', 'ra']
        ds = OrderedDict()

        for i, antenna in enumerate(antennas):

            subset = data['beam_number'] == i+1
            metadata['beam_number'] = i+1
            metadata['beam_name'] = antenna

            # convert dict to xarray.Dataset or numpy.ndarray
            if to_xarray:
                sub_data = {}
                for var_name in data.keys():

                    if var_name == 'beam_number' and generic:
                        continue

                    if len(data[var_name].shape) == 1:
                        dim = ['obs']
                    elif len(data[var_name].shape) == 2:
                        dim = ['obs', 'echo']

                    sub_data[var_name] = (dim, data[var_name][subset])

                coords = {}
                coords_fields = ['lon', 'lat', 'time']

                for cf in coords_fields:
                    coords[cf] = sub_data.pop(cf)

                ds[antenna] = xr.Dataset(sub_data, coords=coords,
                                         attrs=metadata)
            else:
                # collect dtype info
                dtype = []
                for var_name in data.keys():

                    if len(data[var_name][subset].shape) == 1:
                        dtype.append(
                            (var_name, data[var_name][subset].dtype.str))
                    elif len(data[var_name][subset].shape) > 1:
                        dtype.append((var_name, data[var_name][
                            subset].dtype.str, data[var_name][
                                subset].shape[1:]))

                ds[antenna] = np.empty(
                    data['time'][subset].size, dtype=np.dtype(dtype))

                for var_name in data.keys():
                    if var_name == 'beam_number' and generic:
                        continue
                    ds[antenna][var_name] = data[var_name][subset]

        return ds

    def close(self):
        """
        Close file.
        """
        pass


def conv_hdf5l1b_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Original dataset.
    metadata : dict
        Metadata.

    Returns
    -------
    data : dict of numpy.ndarray
        Converted dataset.
    """
    # convert spacecraft_id to internal sat_id
    sat_id = np.array([4, 3, 5])
    metadata['sat_id'] = sat_id[metadata['spacecraft_id']-1]

    # compute ascending/descending direction
    data['as_des_pass'] = (
        data['sat_track_azi'] < 270).astype(np.uint8)

    flags = {'flagfield_rf1': np.tile(data['flagfield_rf1'], 192),
             'flagfield_rf2': np.tile(data['flagfield_rf2'], 192),
             'flagfield_pl': np.tile(data['flagfield_pl'], 192),
             'flagfield_gen1': data['flagfield_gen1'].flatten(),
             'flagfield_gen2': data['flagfield_gen2'].flatten()}

    data['f_usable'] = set_flags(flags)
    data['f_usable'] = data['f_usable'].reshape(-1, 192)

    data['swath_indicator'] = np.int8(data['beam_number'].flatten() > 3)

    skip_fields = ['utc_localisation-days', 'utc_localisation-milliseconds',
                   'degraded_inst_mdr', 'degraded_proc_mdr', 'flagfield_rf1',
                   'flagfield_rf2', 'flagfield_pl', 'flagfield_gen1',
                   'flagfield_gen2']

    gen_fields_lut = {'inc_angle_full': ('inc', np.float32),
                      'azi_angle_full': ('azi', np.float32),
                      'sigma0_full': ('sig', np.float32)}

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    num_cells = data['lat'].shape[1]

    for var_name in data.keys():
        if len(data[var_name].shape) == 1:
            data[var_name] = np.repeat(data[var_name], num_cells)
        if len(data[var_name].shape) == 2:
            data[var_name] = data[var_name].flatten()

        if var_name in gen_fields_lut.items():
            new_name = gen_fields_lut[var_name][0]
            new_dtype = gen_fields_lut[var_name][1]
            data[new_name] = data.pop(var_name).astype(new_dtype)

    return data
