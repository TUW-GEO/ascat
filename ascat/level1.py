# Copyright (c) 2018, TU Wien, Department of Geodesy and Geoinformation
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
General Level 1 data readers for ASCAT data in all formats. Not specific to distributor.
"""
import os

import numpy as np
from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

import ascat.read_native.eps_native as eps_native
import ascat.read_native.bufr as bufr
import ascat.read_native.nc as nc
import ascat.read_native.hdf5 as h5


class AscatL1Image(ImageBase):
    """
    General Level 1b Image for ASCAT data
    """
    def __init__(self, *args, **kwargs):
        """
        Initialization of i/o object.
        """
        super(AscatL1Image, self).__init__(*args, **kwargs)

    def read(self, timestamp=None, file_format=None, native=False, **kwargs):

        if file_format == None:
            file_format = get_file_format(self.filename)

        if file_format == ".nat":
            if native:
                img = eps_native.AscatL1bEPSImage(self.filename).read(timestamp)
            else:
                img_raw = eps_native.AscatL1bEPSImage(self.filename).read(timestamp)
                img = eps2generic(img_raw)

        elif file_format == ".nc":
            if native:
                img = nc.AscatL1NcFile(self.filename).read(timestamp)
            else:
                img_raw = nc.AscatL1NcFile(self.filename).read(timestamp)
                img = nc2generic(img_raw)

        elif file_format == ".bfr" or file_format == ".buf":
            if native:
                img = bufr.AscatL1BufrFile(self.filename).read(timestamp)
            else:
                img_raw = bufr.AscatL1BufrFile(self.filename).read(timestamp)
                img = bfr2generic(img_raw)

        elif file_format == ".h5":
            if native:
                img = h5.AscatL1H5File(self.filename).read(timestamp)
            else:
                img_raw = h5.AscatL1H5File(self.filename).read(timestamp)
                img = hdf2generic(img_raw)

        else:
            raise RuntimeError(
                "Format not found, please indicate the file format. [\".nat\", \".nc\", \".bfr\", \".h5\"]")

        return img

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def get_file_format(filename):
    """
    Try to guess the file format from the extension.
    """
    if os.path.splitext(filename)[1] == '.gz':
        file_format = os.path.splitext(os.path.splitext(filename)[0])[1]
    else:
        file_format = os.path.splitext(filename)[1]
    return file_format


def eps2generic(native_Image):
    """
    Convert the native eps Image into a generic one.
    """
    template = template_ASCATL1()
    generic_data = {}
    if type(native_Image) is dict:
        img = {'img1': {}, 'img2': {}, 'img3': {}, 'img4': {},
                      'img5': {}, 'img6': {}}
        for szf_img in native_Image:
            n_records = native_Image[szf_img].lat.shape[0]
            generic_data = np.repeat(template, n_records)

            fields = [('jd', 'jd'),
                      # ('node_num', 'NODE_NUM'),
                      # ('line_num', 'LINE_NUM'),
                      ('swath', 'SWATH_INDICATOR'),
                      ('azif', 'f_AZI_ANGLE_TRIP'),
                      ('azim', 'm_AZI_ANGLE_TRIP'),
                      ('azia', 'a_AZI_ANGLE_TRIP'),
                      ('incf', 'f_INC_ANGLE_TRIP'),
                      ('incm', 'm_INC_ANGLE_TRIP'),
                      ('inca', 'a_INC_ANGLE_TRIP'),
                      ('sigf', 'f_SIGMA0_TRIP'),
                      ('sigm', 'm_SIGMA0_TRIP'),
                      ('siga', 'a_SIGMA0_TRIP'),
                      ('kpf', 'f_KP'),
                      ('kpm', 'm_KP'),
                      ('kpa', 'a_KP'),
                      # ('num_obs', np.ubyte),
                      # ('usable_flag', np.uint8)
                      ]
            for field in fields:
                generic_data[field[0]] = native_Image[szf_img].data[field[1]]

            fields = [('sat_id', 'SPACECRAFT_ID'),
                      ('abs_orbit_nr', 'ORBIT_START')]
            for field in fields:
                generic_data[field[0]] = native_Image[szf_img].metadata[
                    field[1]].repeat(n_records)

            img[szf_img] = Image(native_Image[szf_img].lon,
                                 native_Image[szf_img].lat,
                                 generic_data,
                                 native_Image[szf_img].metadata,
                                 native_Image[szf_img].timestamp,
                                 timekey='jd')

    else:
        n_records = native_Image.lat.shape[0]
        # generic_data = np.repeat(template, n_records)

        fields = [('jd', 'jd'),
                  ('node_num', 'NODE_NUM'),
                  ('line_num', 'LINE_NUM'),
                  ('swath', 'SWATH INDICATOR'),
                  ('azif', 'f_AZI_ANGLE_TRIP'),
                  ('azim', 'm_AZI_ANGLE_TRIP'),
                  ('azia', 'a_AZI_ANGLE_TRIP'),
                  ('incf', 'f_INC_ANGLE_TRIP'),
                  ('incm', 'm_INC_ANGLE_TRIP'),
                  ('inca', 'a_INC_ANGLE_TRIP'),
                  ('sigf', 'f_SIGMA0_TRIP'),
                  ('sigm', 'm_SIGMA0_TRIP'),
                  ('siga', 'a_SIGMA0_TRIP'),
                  ('kpf', 'f_KP'),
                  ('kpm', 'm_KP'),
                  ('kpa', 'a_KP'),
                  # ('num_obs', np.ubyte),
                  # ('usable_flag', np.uint8)
                  ]
        for field in fields:
            generic_data[field[0]] = native_Image.data[field[1]]

        fields = [('sat_id', 'SPACECRAFT_ID'),
                  ('abs_orbit_nr', 'ORBIT_START')]
        for field in fields:
            generic_data[field[0]] = native_Image.metadata[field[1]].repeat(n_records)

        img = Image(native_Image.lon, native_Image.lat, generic_data,
                    native_Image.metadata, native_Image.timestamp,
                    timekey='jd')

    return img


def nc2generic(native_Image):
    """
    Convert the native nc Image into a generic one.
    """
    template = template_ASCATL1()
    generic_data = {}

    n_records = native_Image.lat.shape[0]
    # generic_data = np.repeat(template, n_records)

    fields = [('jd', 'jd'),
              # ('node_num', 'NODE_NUM'),
              # ('line_num', 'LINE_NUM'),
              ('swath', 'swath_indicator'),
              ('azif', 'f_azi_angle_trip'),
              ('azim', 'm_azi_angle_trip'),
              ('azia', 'a_azi_angle_trip'),
              ('incf', 'f_inc_angle_trip'),
              ('incm', 'm_inc_angle_trip'),
              ('inca', 'a_inc_angle_trip'),
              ('sigf', 'f_sigma0_trip'),
              ('sigm', 'm_sigma0_trip'),
              ('siga', 'a_sigma0_trip'),
              ('kpf', 'f_kp'),
              ('kpm', 'm_kp'),
              ('kpa', 'a_kp'),
              # ('num_obs', np.ubyte),
              # ('usable_flag', 'f_usable')
              ]
    for field in fields:
        generic_data[field[0]] = native_Image.data[field[1]]

    img = Image(native_Image.lon, native_Image.lat, generic_data,
                native_Image.metadata, native_Image.timestamp,
                timekey='jd')

    return img


def bfr2generic(native_Image):
    """
    Convert the native bfr Image into a generic one.
    """
    template = template_ASCATL1()
    generic_data = {}

    n_records = native_Image.lat.shape[0]
    # generic_data = np.repeat(template, n_records)

    fields = [('jd', 'jd'),
              ('sat_id', 'Satellite Identifier'),
              ('abs_orbit_nr', 'Orbit Number'),
              # ('node_num', 'NODE_NUM'),
              # ('line_num', 'LINE_NUM'),
              # ('swath', 'Swath Indicator'),
              ('azif', 'f_Antenna Beam Azimuth'),
              ('azim', 'm_Antenna Beam Azimuth'),
              ('azia', 'a_Antenna Beam Azimuth'),
              ('incf', 'f_Radar Incidence Angle'),
              ('incm', 'm_Radar Incidence Angle'),
              ('inca', 'a_Radar Incidence Angle'),
              ('sigf', 'f_Backscatter'),
              ('sigm', 'm_Backscatter'),
              ('siga', 'a_Backscatter'),
              ('kpf', 'f_Radiometric Resolution (Noise Value)'),
              ('kpm', 'm_Radiometric Resolution (Noise Value)'),
              ('kpa', 'a_Radiometric Resolution (Noise Value)'),
              # ('num_obs', np.ubyte),
              # ('usable_flag', np.uint8)
              ]

    kp_vars = ['kpf', 'kpm', 'kpa']
    for field in fields:
        if field[0] in kp_vars:
            generic_data[field[0]] = native_Image.data[field[1]]/100
        else:
            generic_data[field[0]] = native_Image.data[field[1]]


    img = Image(native_Image.lon, native_Image.lat, generic_data,
                native_Image.metadata, native_Image.timestamp,
                timekey='jd')

    return img


def hdf2generic(native_Image):
    """
    Convert the native nc Image into a generic one.
    """
    img = native_Image
    if type(img) is dict:
        pass
    return img


def template_ASCATL1():
    """
    Generic lvl1b template. (from generic IO: ASCRS009)
    """
    metadata = {'temp_name': 'ASCATL1'}

    struct = np.dtype([('jd', np.double),
                       ('sat_id', np.byte),
                       ('abs_line_nr', np.uint32),
                       ('abs_orbit_nr', np.uint32),
                       ('node_num', np.uint8),
                       ('line_num', np.uint16),
                       ('orb_dir', np.dtype('S1')),
                       ('swath', np.byte),
                       ('azif', np.float32),
                       ('azim', np.float32),
                       ('azia', np.float32),
                       ('incf', np.float32),
                       ('incm', np.float32),
                       ('inca', np.float32),
                       ('sigf', np.float32),
                       ('sigm', np.float32),
                       ('siga', np.float32),
                       ('kpf', np.float32),
                       ('kpm', np.float32),
                       ('kpa', np.float32),
                       ('kpf_quality', np.float32),
                       ('kpm_quality', np.float32),
                       ('kpa_quality', np.float32),
                       ('num_obs', np.ubyte),
                       ('land_flag', np.uint8),
                       ('usable_flag', np.uint8)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset
