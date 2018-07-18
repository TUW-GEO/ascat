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
General Level 2 data readers for ASCAT data in all formats. Not specific to distributor.
"""
import os

import numpy as np
from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

import ascat.read_native.eps_native as read_eps
import ascat.read_native.bufr as read_bufr
import ascat.read_native.nc as read_nc

byte_nan = np.iinfo(np.byte).min
ubyte_nan = np.iinfo(np.ubyte).max
uint8_nan = np.iinfo(np.uint8).max
uint16_nan = np.iinfo(np.uint16).max
uint32_nan = np.iinfo(np.uint32).max
float32_nan = np.finfo(np.float32).min
float64_nan = np.finfo(np.float64).min
long_nan = np.iinfo(np.int32).min
int_nan = np.iinfo(np.int16).min

class AscatL2Image(ImageBase):
    """
    General Level 2 Image
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization of i/o object.
        """
        super(AscatL2Image, self).__init__(*args, **kwargs)

    def read(self, timestamp=None, file_format=None, native=False, **kwargs):

        if file_format == None:
            file_format = get_file_format(self.filename)

        if file_format == ".nat":
            if native:
                img = read_eps.AscatL2EPSImage(self.filename).read(timestamp)
            else:
                img_raw = read_eps.AscatL2EPSImage(self.filename).read(timestamp)
                img = eps2generic(img_raw)

        elif file_format == ".nc":
            if native:
                img = read_nc.AscatL2SsmNcFile(self.filename).read(timestamp)
            else:
                img_raw = read_nc.AscatL2SsmNcFile(self.filename).read(timestamp)
                img = nc2generic(img_raw)

        elif file_format == ".bfr" or file_format == ".buf":
            if native:
                img = read_bufr.AscatL2SsmBufrFile(self.filename).read(timestamp)
            else:
                img_raw = read_bufr.AscatL2SsmBufrFile(self.filename).read(timestamp)
                img = bfr2generic(img_raw)

        else:
            raise RuntimeError(
                "Format not found, please indicate the file_format. [\".nat\", \".nc\", \".bfr\"]")

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


def nc2generic(native_Image):
    """
    Convert the native nc Image into a generic one.
    """
    n_records = native_Image.lat.shape[0]
    generic_data = get_template_ASCATL2_SMX(n_records)

    fields = [('jd', 'jd'),
              ('sat_id', None),
              ('abs_line_nr', 'abs_line_number'),
              ('abs_orbit_nr', None),
              ('node_num', 'node_num'),
              ('line_num', 'line_num'),
              ('as_des_pass', 'as_des_pass'),
              ('swath', 'swath_indicator'),
              ('azif', None),
              ('azim', None),
              ('azia', None),
              ('incf', None),
              ('incm', None),
              ('inca', None),
              ('sigf', None),
              ('sigm', None),
              ('siga', None),
              ('sm', 'soil_moisture'),
              ('sm_noise', 'soil_moisture_error'),
              ('sm_sensitivity', 'soil_moisture_sensitivity'),
              ('sig40', 'sigma40'),
              ('sig40_noise', 'sigma40_error'),
              ('slope40', 'slope40'),
              ('slope40_noise', 'slope40_error'),
              ('dry_backscatter', 'dry_backscatter'),
              ('wet_backscatter', 'wet_backscatter'),
              ('mean_surf_sm', 'mean_soil_moisture')]

    for field in fields:
        if field[1] is None:
            continue

        if (type(native_Image.data[field[1]]) == np.ma.core.MaskedArray):
            valid_mask = ~native_Image.data[field[1]].mask
            generic_data[field[0]][valid_mask] = native_Image.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_Image.data[field[1]]

    # flag_fields need to be treated differently since they are not masked arrays
    # so we need to check for nan values
    flags = [('correction_flag', 'corr_flags'),
              # There is a processing flag but it is different to the other formats
              ('processing_flag', None),
              ('aggregated_quality_flag', 'aggregated_quality_flag'),
              ('snow_cover_probability', 'snow_cover_probability'),
              ('frozen_soil_probability', 'frozen_soil_probability'),
              ('innudation_or_wetland', 'wetland_flag'),
              ('topographical_complexity', 'topography_flag')]

    for field in flags:
        if field[1] is None:
            continue

        valid_mask = (native_Image.data[field[1]] != ubyte_nan)
        generic_data[field[0]][valid_mask] = native_Image.data[field[1]][
            valid_mask]

    fields = [('sat_id', 'sat_id'),
              ('abs_orbit_nr', 'orbit_start')]
    for field in fields:
        generic_data[field[0]] = np.repeat(native_Image.metadata[field[1]],
                                           n_records)

    # convert sat_id (spacecraft id) to the department intern definition
    # use an array as look up table
    sat_id_lut = np.array([0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_Image.lon, native_Image.lat, generic_data,
                native_Image.metadata, native_Image.timestamp,
                timekey='jd')

    return img


def eps2generic(native_Image):
    """
    Convert the native eps Image into a generic one.
    """
    n_records = native_Image.lat.shape[0]
    generic_data = get_template_ASCATL2_SMX(n_records)

    fields = [('jd', 'jd', None),
              ('sat_id', None, None),
              ('abs_line_nr', 'ABS_LINE_NUMBER', None),
              ('abs_orbit_nr', None, None),
              ('node_num', 'NODE_NUM', None),
              ('line_num', 'LINE_NUM', None),
              ('as_des_pass', 'AS_DES_PASS', None),
              ('swath', 'SWATH_INDICATOR', byte_nan),
              ('azif', 'f_AZI_ANGLE_TRIP', int_nan),
              ('azim', 'm_AZI_ANGLE_TRIP', int_nan),
              ('azia', 'a_AZI_ANGLE_TRIP', int_nan),
              ('incf', 'f_INC_ANGLE_TRIP', uint16_nan),
              ('incm', 'm_INC_ANGLE_TRIP', uint16_nan),
              ('inca', 'a_INC_ANGLE_TRIP', uint16_nan),
              ('sigf', 'f_SIGMA0_TRIP', long_nan),
              ('sigm', 'm_SIGMA0_TRIP', long_nan),
              ('siga', 'a_SIGMA0_TRIP', long_nan),
              ('sm', 'SOIL_MOISTURE', uint16_nan),
              ('sm_noise', 'SOIL_MOISTURE_ERROR', uint16_nan),
              ('sm_sensitivity', 'SOIL_MOISTURE_SENSETIVITY', np.float32(uint32_nan)),
              ('sig40', 'SIGMA40', long_nan),
              ('sig40_noise', 'SIGMA40_ERROR', long_nan),
              ('slope40', 'SLOPE40', long_nan),
              ('slope40_noise', 'SLOPE40_ERROR', long_nan),
              ('dry_backscatter', 'DRY_BACKSCATTER', long_nan),
              ('wet_backscatter', 'WET_BACKSCATTER', long_nan),
              ('mean_surf_sm', 'MEAN_SURF_SOIL_MOISTURE', uint16_nan),
              ('correction_flag', 'CORRECTION_FLAGS', uint16_nan),
              ('processing_flag', 'PROCESSING_FLAGS', uint16_nan),
              ('aggregated_quality_flag', 'AGGREGATED_QUALITY_FLAG', ubyte_nan),
              ('snow_cover_probability', 'SNOW_COVER_PROBABILITY', ubyte_nan),
              ('frozen_soil_probability', 'FROZEN_SOIL_PROBABILITY', ubyte_nan),
              ('innudation_or_wetland', 'INNUDATION_OR_WETLAND', ubyte_nan),
              ('topographical_complexity', 'TOPOGRAPHICAL_COMPLEXITY', ubyte_nan)]

    for field in fields:
        if field[1] is None:
            continue

        if field[2] is not None:
            valid_mask = (native_Image.data[field[1]] != field[2])
            generic_data[field[0]][valid_mask] = native_Image.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_Image.data[field[1]]

    fields = [('sat_id', 'SPACECRAFT_ID'),
              ('abs_orbit_nr', 'ORBIT_START')]
    for field in fields:
        generic_data[field[0]] = np.repeat(native_Image.metadata[field[1]], n_records)

    # convert sat_id (spacecraft id) to the department intern definition
    # use an array as look up table
    sat_id_lut = np.array([0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_Image.lon, native_Image.lat, generic_data,
                native_Image.metadata, native_Image.timestamp,
                timekey='jd')

    return img


def bfr2generic(native_Image):
    """
    Convert the native bfr Image into a generic one.
    """
    n_records = native_Image.lat.shape[0]
    generic_data = get_template_ASCATL2_SMX(n_records)

    fields = [('jd', 'jd', None),
              ('sat_id', 'Satellite Identifier', None),
              ('abs_line_nr', None, None),
              ('abs_orbit_nr', 'Orbit Number', None),
              ('node_num', 'Cross-Track Cell Number', None),
              ('line_num', 'line_num', None),
              ('as_des_pass', 'as_des_pass', None),
              ('swath', 'swath_indicator', None),
              ('azif', 'f_Antenna Beam Azimuth', None),
              ('azim', 'm_Antenna Beam Azimuth', None),
              ('azia', 'a_Antenna Beam Azimuth', None),
              ('incf', 'f_Radar Incidence Angle', None),
              ('incm', 'm_Radar Incidence Angle', None),
              ('inca', 'a_Radar Incidence Angle', None),
              ('sigf', 'f_Backscatter', None),
              ('sigm', 'm_Backscatter', None),
              ('siga', 'a_Backscatter', None),
              ('sm', 'Surface Soil Moisture (Ms)', 1.7e+38),
              ('sm_noise', 'Estimated Error In Surface Soil Moisture', 1.7e+38),
              ('sm_sensitivity', 'Soil Moisture Sensitivity', 1.7e+38),
              ('sig40', 'Backscatter', 1.7e+38),
              ('sig40_noise', 'Estimated Error In Sigma0 At 40 Deg Incidence Angle', 1.7e+38),
              ('slope40', 'Slope At 40 Deg Incidence Angle', 1.7e+38),
              ('slope40_noise', 'Estimated Error In Slope At 40 Deg Incidence Angle', 1.7e+38),
              ('dry_backscatter', 'Dry Backscatter', 1.7e+38),
              ('wet_backscatter', 'Wet Backscatter', 1.7e+38),
              ('mean_surf_sm', 'Mean Surface Soil Moisture', 1.7e+40),
              ('correction_flag', 'Soil Moisture Correction Flag', 1.7e+38),
              ('processing_flag', 'Soil Moisture Processing Flag', 1.7e+38),
              ('aggregated_quality_flag', None),
              ('snow_cover_probability', 'Snow Cover', 1.7e+38),
              ('frozen_soil_probability', 'Frozen Land Surface Fraction', 1.7e+38),
              ('innudation_or_wetland', 'Inundation And Wetland Fraction', 1.7e+38),
              ('topographical_complexity', 'Topographic Complexity', 1.7e+38)]

    for field in fields:
        if field[1] is None:
            continue

        if field[2] is not None:
            valid_mask = (native_Image.data[field[1]] != field[2])
            generic_data[field[0]][valid_mask] = native_Image.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_Image.data[field[1]]

    # convert sat_id (spacecraft id) to the department intern definition
    # use an array as look up table
    sat_id_lut = np.array([0, 0, 0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_Image.lon, native_Image.lat, generic_data,
                native_Image.metadata, native_Image.timestamp,
                timekey='jd')

    return img

def get_template_ASCATL2_SMX(n=1):
    """
    Generic lvl2 SMX template.
    """
    metadata = {'temp_name': 'ASCATL2'}

    struct = np.dtype([('jd', np.float64),
                       ('sat_id', np.byte),
                       ('abs_line_nr', np.uint32),
                       ('abs_orbit_nr', np.uint32),
                       ('node_num', np.uint8),
                       ('line_num', np.uint16),
                       ('as_des_pass', np.byte),
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
                       ('sm', np.float32),
                       ('sm_noise', np.float32),
                       ('sm_sensitivity', np.float32),
                       ('sig40', np.float32),
                       ('sig40_noise', np.float32),
                       ('slope40', np.float32),
                       ('slope40_noise', np.float32),
                       ('dry_backscatter', np.float32),
                       ('wet_backscatter', np.float32),
                       ('mean_surf_sm', np.float32),
                       ('correction_flag', np.uint8),
                       ('processing_flag', np.uint16),
                       ('aggregated_quality_flag', np.uint8),
                       ('snow_cover_probability', np.float32),
                       ('frozen_soil_probability', np.float32),
                       ('innudation_or_wetland', np.float32),
                       ('topographical_complexity', np.float32)],
                       metadata=metadata)

    record = np.array([(float64_nan, byte_nan, uint32_nan, uint32_nan,
                        uint8_nan, uint16_nan, byte_nan, byte_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, uint8_nan, uint16_nan,
                        uint8_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan)], dtype=struct)

    return np.repeat(record, n)