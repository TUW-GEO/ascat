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
Readers for SZF data in h5 format.
"""
from __future__ import division

import numpy as np
import h5py

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

# 1.1.2000 00:00:00 in jd
julian_epoch = 2451544.5


class AscatL1H5File(ImageBase):
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

    def __init__(self, filename, mode='r', h5_keys=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL1H5File, self).__init__(filename, mode=mode,
                                            **kwargs)
        self.h5_keys = h5_keys
        self.ds = None

    def read(self, timestamp=None):
        """
        reads from the netCDF file given by the filename

        Returns
        -------
        data : pygeobase.object_base.Image
        """

        if self.ds is None:
            self.ds = h5py.File(self.filename)
            while len(self.ds.keys()) == 1:
                self.ds = self.ds[list(self.ds.keys())[0]]
        raw_data = self.ds['DATA']
        raw_metadata = self.ds['METADATA']

        # store data in dictionary
        data = {}
        metadata = {}

        if self.h5_keys is None:
            var_to_read = list(
                raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'].dtype.names)
        else:
            var_to_read = self.h5_keys

        # make sure that essential variables are read always:
        if 'LATITUDE_FULL' not in var_to_read:
            var_to_read.append('LATITUDE_FULL')
        if 'LONGITUDE_FULL' not in var_to_read:
            var_to_read.append('LONGITUDE_FULL')

        num_cells = raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'][
            'LATITUDE_FULL'].shape[1]
        num_lines = raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'][
            'LATITUDE_FULL'].shape[0]

        # read the requested variables and scale them if they have a
        # scaling factor
        # encode() is needed for py3 comparison between str and byte
        for name in var_to_read:
            variable = raw_data['MDR_1B_FULL_ASCA_Level_1_ARRAY_000001'][name]
            if name.encode() in \
                    raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'].value[
                        'EntryName']:
                var_index = np.where(
                    raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'][
                        'EntryName'] == name.encode())[0][0]
                if raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'].value[
                        'Scale Factor'][var_index] != "n/a".encode():
                    sf = 10 ** float(
                        raw_data['MDR_1B_FULL_ASCA_Level_1_DESCR'].value[
                            'Scale Factor'][var_index])
                    variable = variable / sf
            data[name] = variable[:].flatten()
            if len(variable.shape) == 1:
                # If the data is 1D then we repeat it for each cell
                data[name] = np.repeat(data[name], num_cells)

        data['AS_DES_PASS'] = (data['SAT_TRACK_AZI'] < 270).astype(np.uint8)

        for name in raw_metadata.keys():
            for subname in raw_metadata[name].keys():
                if name not in metadata:
                    metadata[name] = dict()
                metadata[name][subname] = raw_metadata[name][subname].value

        # modify longitudes from [0,360] to [-180,180]
        mask = data['LONGITUDE_FULL'] > 180
        data['LONGITUDE_FULL'][mask] += -360.

        if 'AZI_ANGLE_FULL' in var_to_read:
            mask = data['AZI_ANGLE_FULL'] < 0
            data['AZI_ANGLE_FULL'][mask] += 360

        if 'UTC_LOCALISATION-days' in var_to_read and \
                'UTC_LOCALISATION-milliseconds' in var_to_read:
            data['jd'] = shortcdstime2jd(data['UTC_LOCALISATION-days'],
                                         data['UTC_LOCALISATION-milliseconds'])

        set_flags(data)

        fields = ['SPACECRAFT_ID', 'ORBIT_START',
                  'PROCESSOR_MAJOR_VERSION', 'PROCESSOR_MINOR_VERSION',
                  'FORMAT_MAJOR_VERSION', 'FORMAT_MINOR_VERSION'
                  ]
        for field in fields:
            var_index = np.where(
                np.core.defchararray.startswith(
                    metadata['MPHR']['MPHR_TABLE']['EntryName'],
                    field.encode()))[0][0]
            var = metadata['MPHR']['MPHR_TABLE']['EntryValue'][
                var_index].decode()
            if field == 'SPACECRAFT_ID':
                var = var[-1]
            metadata[field] = int(var)

        image_dict = {'img1': {}, 'img2': {}, 'img3': {}, 'img4': {},
                      'img5': {}, 'img6': {}}
        data_full = {'d1': {}, 'd2': {}, 'd3': {}, 'd4': {}, 'd5': {},
                     'd6': {}}

        # separate data into single beam images
        for i in range(1, 7):
            dataset = 'd' + str(i)
            img = 'img' + str(i)
            mask = ((data['BEAM_NUMBER']) == i)
            for field in data:
                data_full[dataset][field] = data[field][mask]

            lon = data_full[dataset].pop('LONGITUDE_FULL')
            lat = data_full[dataset].pop('LATITUDE_FULL')
            image_dict[img] = Image(lon, lat, data_full[dataset], metadata,
                                    timestamp, timekey='jd')

        return image_dict

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


def set_flags(data):
    """
    Compute summary flag for each measurement with a value of 0, 1 or 2
    indicating nominal, slightly degraded or severely degraded data.

    Parameters
    ----------
    data : numpy.ndarray
        SZF data.
    """

    # category:status = 'red': 2, 'amber': 1, 'warning': 0
    flag_status_bit = {'FLAGFIELD_RF1': {'2': [2, 4],
                                         '1': [0, 1, 3]},

                       'FLAGFIELD_RF2': {'2': [0, 1]},

                       'FLAGFIELD_PL': {'2': [0, 1, 2, 3],
                                        '0': [4]},

                       'FLAGFIELD_GEN1': {'2': [1],
                                          '0': [0]},

                       'FLAGFIELD_GEN2': {'2': [2],
                                          '1': [0],
                                          '0': [1]}
                       }

    for flagfield in flag_status_bit.keys():
        # get flag data in binary format to get flags
        unpacked_bits = np.unpackbits(data[flagfield])

        # find indizes where a flag is set
        set_bits = np.where(unpacked_bits == 1)[0]
        if set_bits.size != 0:
            pos_8 = 7 - (set_bits % 8)

            for category in sorted(flag_status_bit[flagfield].keys()):
                if (int(category) == 0) and (flagfield != 'FLAGFIELD_GEN2'):
                    continue

                for bit2check in flag_status_bit[flagfield][category]:
                    pos = np.where(pos_8 == bit2check)[0]
                    data['F_USABLE'] = np.zeros(data['FLAGFIELD_GEN2'].size)
                    data['F_USABLE'][set_bits[pos] // 8] = int(category)

                    # land points
                    if (flagfield == 'FLAGFIELD_GEN2') and (bit2check == 1):
                        data['F_LAND'] = np.zeros(data['FLAGFIELD_GEN2'].size)
                        data['F_LAND'][set_bits[pos] // 8] = 1


def shortcdstime2jd(days, milliseconds):
    """
    Convert cds time to julian date
    """
    offset = days + (milliseconds / 1000.) / (24. * 60. * 60.)
    return julian_epoch + offset
