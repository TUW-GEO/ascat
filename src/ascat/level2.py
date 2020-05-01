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
General Level 2 data readers for ASCAT data in all formats.
"""

import os
import numpy as np
from datetime import datetime

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

import ascat.read_native.eps_native as read_eps
import ascat.read_native.bufr as read_bufr
import ascat.read_native.nc as read_nc
from ascat.math import db2lin, lin2db, hamming_window
from ascat.base import ASCAT_MultiTemporalImageBase

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
        """
        Read ASCAT swath files.

        Parameters
        ----------
        timestamp : str, optional
            Time stamp of file (default: None).
        file_format : str, optional
            File format: '.nat', '.nc', '.bfr', '.h5' (default: None).
            If None file format will be guessed based on the file ending.
        native : bool, optional
            Return native or generic data set format (default: False).
            The main difference is that in the original native format fields
            are called differently.

        Returns
        -------
        img : pygeobase.object_base.Image
            ASCAT swath image.
        """

        if file_format is None:
            file_format = get_file_format(self.filename)

        if file_format in [".nat", ".nat.gz"]:
            img = read_eps.AscatL2EPSImage(self.filename).read(timestamp)
            if not native:
                img = eps2generic(img)

        elif file_format in [".nc", ".nc.gz"]:
            img = read_nc.AscatL2SsmNcFile(self.filename).read(timestamp)
            if not native:
                img = nc2generic(img)

        elif file_format in [".bfr", ".bfr.gz", ".buf", "buf.gz"]:
            img = read_bufr.AscatL2SsmBufrFile(self.filename).read(timestamp)
            if not native:
                img = bfr2generic(img)

        else:
            raise RuntimeError(
                "Format not found, please indicate the file_format. "
                "[\".nat\", \".nc\", \".bfr\"]")

        return img

    def read_masked_data(self, correction_flag=0, processing_flag=0,
                         aggregated_quality_flag=100,
                         snow_cover_probability=100,
                         frozen_soil_probability=100,
                         innudation_or_wetland=100,
                         topographical_complexity=100, **kwargs):
        """
        Read ASCAT swath files and mask unusable observations.

        Parameters
        ----------
        correction_flag : int, optional
            Correction flag (default: 0).
        aggregated_quality_flag : int, optional
            Aggregated quality flag (default: 0).
        snow_cover_probability : float, optional
            Snow cover probability (default: 100).
        frozen_soil_probability : float, optional
            Frozen soil probability (default: 100).
        innudation_or_wetland : float, optional
            Innundation and wetland flag (default: 100).
        topographical_complexity, float, optional
            Toographical complexity flag (default: 100).

        Returns
        -------
        img : pygeobase.object_base.Image
            ASCAT swath image with masking applied.
        """
        orbit = self.read(**kwargs)

        valid = np.ones(orbit.data[orbit.data.dtype.names[0]].shape,
                        dtype=np.bool)

        # bitwise comparison, if any bitflag is set by the user and is active
        # for the datarecord the result is bigger than 0 and the not valid
        valid = (valid & (((orbit.data['correction_flag']
                            & correction_flag) == 0)
                          | (orbit.data['correction_flag'] == uint8_nan)))

        valid = (valid & (((orbit.data['processing_flag']
                            & processing_flag) == 0)
                          | (orbit.data['processing_flag'] == uint16_nan)))

        # if any probability/flag is too high the datarecord is not used
        # nan values are not considered since not all formats provide all flags
        # and the values are set to nan there to keep a generic structure
        valid = (valid & ((orbit.data['aggregated_quality_flag']
                           < aggregated_quality_flag)
                          | (orbit.data['aggregated_quality_flag']
                             == uint8_nan)))

        valid = (valid & ((orbit.data['snow_cover_probability']
                           < snow_cover_probability)
                          | (orbit.data['snow_cover_probability']
                             == float32_nan)))

        valid = (valid & ((orbit.data['frozen_soil_probability']
                           < frozen_soil_probability)
                          | (orbit.data['frozen_soil_probability']
                             == float32_nan)))

        valid = (valid & ((orbit.data['innudation_or_wetland']
                           < innudation_or_wetland) |
                          (orbit.data['innudation_or_wetland']
                           == float32_nan)))

        valid = (valid & ((orbit.data['topographical_complexity']
                           < topographical_complexity)
                          | (orbit.data['topographical_complexity']
                             == float32_nan)))

        valid_num = orbit.data['jd'][valid].shape[0]
        masked_data = get_template_ASCATL2_SMX(valid_num)

        for key in orbit.data.dtype.names:
            masked_data[key] = orbit.data[key][valid]

        img = Image(orbit.lon[valid], orbit.lat[valid], masked_data,
                    orbit.metadata, orbit.timestamp, timekey='jd')

        return img

    def resample_data(self, data, index, distance, windowRadius, **kwargs):
        """

        Parameters
        ----------


        Returns
        -------
        img : pygeobase.object_base.Image
            ASCAT swath image with masking applied.

        """
        # target template
        template = get_resample_template_ASCATL2_SMX()
        res_orbit = np.repeat(template, index.shape[0])

        # get weights
        weights, _ = hamming_window(windowRadius, distance)

        # resample soil moiusture variables [%]
        sm = ['sm', 'sm_noise', 'mean_surf_sm']

        for n in sm:
            # account for nan values: remove them from calculation by setting
            # them to np.nan and using np.nansum later on
            data[n][data[n] == float32_nan] = np.nan
            # the weighting also depends on nan values of the data
            # if we don't use a data value we have to exclude the
            # corresponding weight from the total_weights calculation
            weights_exc = weights.copy()
            weights_exc[np.isnan(data[n])[index]] = np.nan
            total_weights_exc = np.nansum(weights, axis=1)
            res_orbit[n] = (np.nansum(data[n][index] * weights, axis=1)
                            / total_weights_exc)
            # set the empty values to nan to keep consistency and avoid
            # 0 which could also be an actual value
            res_orbit[n][np.isnan(res_orbit[n])] = float32_nan
            res_orbit[n][np.nansum(weights_exc, axis=1) == 0] = float32_nan

        # resample dB variables
        fields = ['sigf', 'sigm', 'siga', 'sig40', 'sig40_noise',
                  'slope40', 'slope40_noise', 'sm_sensitivity',
                  'dry_backscatter', 'wet_backscatter']

        for sigma in fields:
            data[sigma][data[sigma] == float32_nan] = np.nan

            weights_exc = weights.copy()
            weights_exc[np.isnan(data[sigma])[index]] = np.nan
            total_weights_exc = np.nansum(weights, axis=1)
            res_orbit[sigma] = lin2db(np.nansum(db2lin(data[sigma])[index]
                                                * weights,
                                                axis=1)
                                      / total_weights_exc)
            res_orbit[sigma][np.isnan(res_orbit[sigma])] = float32_nan
            res_orbit[sigma][np.nansum(weights_exc, axis=1) == 0] = float32_nan

        # resample measurement geometry
        measgeos = ['incf', 'incm', 'inca', 'azif', 'azim', 'azia']

        for mg in measgeos:
            data[mg][data[mg] == float32_nan] = np.nan

            weights_exc = weights.copy()
            weights_exc[np.isnan(data[mg])[index]] = np.nan
            total_weights_exc = np.nansum(weights, axis=1)
            res_orbit[mg] = (np.nansum(data[mg][index] * weights, axis=1)
                             / total_weights_exc)
            res_orbit[mg][np.isnan(res_orbit[mg])] = float32_nan
            res_orbit[mg][np.nansum(weights_exc, axis=1) == 0] = float32_nan

        # nearest neighbour resampling values
        nnResample = ['jd', 'sat_id', 'abs_line_nr', 'abs_orbit_nr',
                      'node_num', 'line_num', 'swath', 'as_des_pass']
        # index of min. distance is equal to 0 because of kd-tree usage
        for nn in nnResample:
            res_orbit[nn] = data[nn][index][:, 0]

        # set number of measurements for resampling
        res_orbit['num_obs'] = np.sum(distance != np.inf, axis=1)

        return res_orbit

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class AscatL2Bufr(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level2 images in bufr format.

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month then please specify the
        string that should be used in datetime.datetime.strftime
    day_search_str: string, optional
        to provide an iterator over all images of a day the method
        _get_possible_timestamps looks for all available images on a day on the
        harddisk.
        Default: '*-ASCA-*-NA-*-%Y%m%d*.bfr'
    file_search_str: string, optional
        this string is used to find a bufr file by the exact date.
        Default: '*-ASCA-*-NA-*-{datetime}*.bfr'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d%H%M%S
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name.
        See bufr.AscatL1BufrFile
    eo_portal : boolean optional
        If your data is from the EUMETSAT EO portal you can set this flag to
        True. This way the the datetime can automatically be read from the
        filename. Otherwise it needs the filename_datetime_format class
        variable set correctly.
    """

    def __init__(self, path, month_path_str='',
                 day_search_str='*-ASCA-*-NA-*-%Y%m%d*-*-*.bfr',
                 file_search_str='*-ASCA-*-NA-*-{datetime}.*.bfr',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(25, 39, '%Y%m%d%H%M%S'),
                 msg_name_lookup=None,
                 eo_portal=False):

        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.datetime_format = datetime_format
        self.filename_datetime_format = filename_datetime_format
        self.eo_portal = eo_portal

        super(AscatL2Bufr, self).__init__(
            path, AscatL2Image, subpath_templ=[month_path_str],
            fname_templ=file_search_str, datetime_format=datetime_format,
            exact_templ=False, ioclass_kws={'msg_name_lookup':
                                            msg_name_lookup})

    def _get_orbit_start_date(self, filename):
        """
        Returns the datetime of the file.

        Parameters
        ----------
        filename : full name (including the path) of the file

        Returns
        -------
        dates : datetime object
            datetime from the filename
        """
        # if your data comes from the EUMETSAT EO Portal this function can
        if self.eo_portal is True:
            filename_base = os.path.basename(filename)
            fln_spl = filename_base.split('-')[5]
            fln_datetime = fln_spl.split('.')[0]
            return datetime.strptime(fln_datetime, self.datetime_format)

        else:
            orbit_start_str = os.path.basename(filename)[
                self.filename_datetime_format[0]:
                self.filename_datetime_format[1]]
            return datetime.strptime(orbit_start_str,
                                     self.filename_datetime_format[2])


class AscatL2Eps(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level2 images in eps format.

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month then please specify the
        string that should be used in datetime.datetime.strftime
    day_search_str: string, optional
        to provide an iterator over all images of a day the method
        _get_possible_timestamps looks for all available images on a day on the
        harddisk.
        Default: 'ASCA_*_*_*_%Y%m%d*_*_*_*_*.nat'
    file_search_str: string, optional
        this string is used to find a bufr file by the exact date.
        Default: 'ASCA_*_*_*_{datetime}Z_*_*_*_*.nat'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d%H%M%S
    eo_portal : boolean optional
        If your data is from the EUMETSAT EO portal you can set this flag to
        True. This way the the datetime can automatically be read from the
        filename. Otherwise it needs the filename_datetime_format class
        variable set correctly.
    """

    def __init__(self, path, month_path_str='',
                 day_search_str='ASCA_*_*_*_%Y%m%d*_*_*_*_*.nat',
                 file_search_str='ASCA_*_*_*_{datetime}Z_*_*_*_*.nat',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(16, 30, '%Y%m%d%H%M%S'),
                 eo_portal=False):

        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.datetime_format = datetime_format
        self.filename_datetime_format = filename_datetime_format
        self.eo_portal = eo_portal

        super(AscatL2Eps, self).__init__(
            path, AscatL2Image, subpath_templ=[month_path_str],
            fname_templ=file_search_str, datetime_format=datetime_format,
            exact_templ=False)


class AscatL2Nc(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level2 images in nc format.

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month then please specify the
        string that should be used in datetime.datetime.strftime
    day_search_str: string, optional
        to provide an iterator over all images of a day the method
        _get_possible_timestamps looks for all available images on a day on the
        harddisk.
        Default: 'W_XX-*_EUMP_%Y%m%d*.nc'
    file_search_str: string, optional
        this string is used to find a bufr file by the exact date.
        Default: 'W_XX-*_EUMP_{datetime}*.nc'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d%H%M%S
    msg_name_lookup: dict, optional
        Dictionary mapping nc msg number to parameter name.
        See nc.AscatL1NcFile
    eo_portal : boolean optional
        If your data is from the EUMETSAT EO portal you can set this flag to
        True. This way the the datetime can automatically be read from the
        filename. Otherwise it needs the filename_datetime_format class
        variable set correctly.
    """

    def __init__(self, path, month_path_str='',
                 day_search_str='W_XX-*_EUMP_%Y%m%d*.nc',
                 file_search_str='W_XX-*_EUMP_{datetime}*.nc',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(62, 76, '%Y%m%d%H%M%S'),
                 msg_name_lookup=None, eo_portal=False):

        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.datetime_format = datetime_format
        self.filename_datetime_format = filename_datetime_format
        self.eo_portal = eo_portal

        super(AscatL2Nc, self).__init__(
            path, AscatL2Image, subpath_templ=[month_path_str],
            fname_templ=file_search_str, datetime_format=datetime_format,
            exact_templ=False, ioclass_kws={'msg_name_lookup':
                                            msg_name_lookup})


def get_file_format(filename):
    """
    Try to guess the file format from the extension.
    """
    if os.path.splitext(filename)[1] == '.gz':
        file_format = os.path.splitext(os.path.splitext(filename)[0])[1]
    else:
        file_format = os.path.splitext(filename)[1]
    return file_format


def nc2generic(native_img):
    """
    Convert the native nc image into a generic one.

    Parameters
    ----------
    native_img : pygeobase.object_base.Image
        Native image.

    Returns
    -------
    img : pygeobase.object_base.Image
        Generic images.
    """
    n_records = native_img.lat.shape[0]
    generic_data = get_template_ASCATL2_SMX(n_records)

    fields = [('jd', 'jd'),
              ('sat_id', None),
              ('abs_line_nr', None),
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

        if type(native_img.data[field[1]]) == np.ma.core.MaskedArray:
            valid_mask = ~native_img.data[field[1]].mask
            generic_data[field[0]][valid_mask] = native_img.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_img.data[field[1]]

    if 'abs_line_number' in native_img.data:
        generic_data['abs_line_nr'] = native_img.data['abs_line_number']

    # flag_fields need to be treated differently since they are not masked
    # arrays so we need to check for nan values
    flags = [('correction_flag', 'corr_flags'),
             # There is a processing flag but it is different to the other
             # formats
             ('processing_flag', None),
             ('aggregated_quality_flag', 'aggregated_quality_flag'),
             ('snow_cover_probability', 'snow_cover_probability'),
             ('frozen_soil_probability', 'frozen_soil_probability'),
             ('innudation_or_wetland', 'wetland_flag'),
             ('topographical_complexity', 'topography_flag')]

    for field in flags:
        if field[1] is None:
            continue

        valid_mask = (native_img.data[field[1]] != ubyte_nan)
        generic_data[field[0]][valid_mask] = native_img.data[field[1]][
            valid_mask]

    fields = [('sat_id', 'sat_id'),
              ('abs_orbit_nr', 'orbit_start')]

    for field in fields:
        generic_data[field[0]] = np.repeat(native_img.metadata[field[1]],
                                           n_records)

    # convert sat_id (spacecraft id) to the intern definition
    sat_id_lut = np.array([0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_img.lon, native_img.lat, generic_data,
                native_img.metadata, native_img.timestamp,
                timekey='jd')

    return img


def eps2generic(native_img):
    """
    Convert the native eps Image into a generic one.

    Parameters
    ----------
    native_img : pygeobase.object_base.Image
        Native image.

    Returns
    -------
    img : pygeobase.object_base.Image
        Converted image.
    """
    n_records = native_img.lat.shape[0]
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
              ('sm_sensitivity', 'SOIL_MOISTURE_SENSETIVITY',
               np.float32(uint32_nan)),
              ('sig40', 'SIGMA40', long_nan),
              ('sig40_noise', 'SIGMA40_ERROR', long_nan),
              ('slope40', 'SLOPE40', long_nan),
              ('slope40_noise', 'SLOPE40_ERROR', long_nan),
              ('dry_backscatter', 'DRY_BACKSCATTER', long_nan),
              ('wet_backscatter', 'WET_BACKSCATTER', long_nan),
              ('mean_surf_sm', 'MEAN_SURF_SOIL_MOISTURE', uint16_nan),
              ('correction_flag', 'CORRECTION_FLAGS', uint16_nan),
              ('processing_flag', 'PROCESSING_FLAGS', uint16_nan),
              (
              'aggregated_quality_flag', 'AGGREGATED_QUALITY_FLAG', ubyte_nan),
              ('snow_cover_probability', 'SNOW_COVER_PROBABILITY', ubyte_nan),
              (
              'frozen_soil_probability', 'FROZEN_SOIL_PROBABILITY', ubyte_nan),
              ('innudation_or_wetland', 'INNUDATION_OR_WETLAND', ubyte_nan),
              ('topographical_complexity', 'TOPOGRAPHICAL_COMPLEXITY',
               ubyte_nan)]

    for field in fields:
        if field[1] is None:
            continue

        if field[2] is not None:
            valid_mask = (native_img.data[field[1]] != field[2])
            generic_data[field[0]][valid_mask] = native_img.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_img.data[field[1]]

    fields = [('sat_id', 'SPACECRAFT_ID'), ('abs_orbit_nr', 'ORBIT_START')]

    for field in fields:
        generic_data[field[0]] = np.repeat(native_img.metadata[field[1]],
                                           n_records)

    # convert sat_id (spacecraft id) to the department intern definition
    # use an array as look up table
    sat_id_lut = np.array([0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_img.lon, native_img.lat, generic_data,
                native_img.metadata, native_img.timestamp,
                timekey='jd')

    return img


def bfr2generic(native_img):
    """
    Convert the native bfr image into a generic one.

    Parameters
    ----------
    native_img : pygeobase.object_base.Image
        Native image.

    Returns
    -------
    img : pygeobase.object_base.Image
        Generic images.
    """
    n_records = native_img.lat.shape[0]
    generic_data = get_template_ASCATL2_SMX(n_records)

    fields = [('jd', 'jd', None),
              ('sat_id', 'Satellite Identifier', None),
              ('abs_line_nr', None, None),
              ('abs_orbit_nr', 'Orbit Number', None),
              ('node_num', 'Cross-Track Cell Number', None),
              ('line_num', 'line_num', None),
              ('as_des_pass', 'as_des_pass', None),
              ('swath', 'swath_indicator', None),
              ('azif', 'f_Antenna Beam Azimuth', 1.7e+38),
              ('azim', 'm_Antenna Beam Azimuth', 1.7e+38),
              ('azia', 'a_Antenna Beam Azimuth', 1.7e+38),
              ('incf', 'f_Radar Incidence Angle', 1.7e+38),
              ('incm', 'm_Radar Incidence Angle', 1.7e+38),
              ('inca', 'a_Radar Incidence Angle', 1.7e+38),
              ('sigf', 'f_Backscatter', 1.7e+38),
              ('sigm', 'm_Backscatter', 1.7e+38),
              ('siga', 'a_Backscatter', 1.7e+38),
              ('sm', 'Surface Soil Moisture (Ms)', 1.7e+38),
              (
              'sm_noise', 'Estimated Error In Surface Soil Moisture', 1.7e+38),
              ('sm_sensitivity', 'Soil Moisture Sensitivity', 1.7e+38),
              ('sig40', 'Backscatter', 1.7e+38),
              ('sig40_noise',
               'Estimated Error In Sigma0 At 40 Deg Incidence Angle', 1.7e+38),
              ('slope40', 'Slope At 40 Deg Incidence Angle', 1.7e+38),
              ('slope40_noise',
               'Estimated Error In Slope At 40 Deg Incidence Angle', 1.7e+38),
              ('dry_backscatter', 'Dry Backscatter', 1.7e+38),
              ('wet_backscatter', 'Wet Backscatter', 1.7e+38),
              ('mean_surf_sm', 'Mean Surface Soil Moisture', 1.7e+40),
              ('correction_flag', 'Soil Moisture Correction Flag', 1.7e+38),
              ('processing_flag', 'Soil Moisture Processing Flag', 1.7e+38),
              ('aggregated_quality_flag', None),
              ('snow_cover_probability', 'Snow Cover', 1.7e+38),
              ('frozen_soil_probability', 'Frozen Land Surface Fraction',
               1.7e+38),
              ('innudation_or_wetland', 'Inundation And Wetland Fraction',
               1.7e+38),
              ('topographical_complexity', 'Topographic Complexity', 1.7e+38)]

    for field in fields:
        if field[1] is None:
            continue

        if field[2] is not None:
            valid_mask = (native_img.data[field[1]] != field[2])
            generic_data[field[0]][valid_mask] = native_img.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_img.data[field[1]]

    # convert sat_id (spacecraft id) to the intern definition
    sat_id_lut = np.array([0, 0, 0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_img.lon, native_img.lat, generic_data,
                native_img.metadata, native_img.timestamp,
                timekey='jd')

    return img


def get_template_ASCATL2_SMX(n=1):
    """
    Generic Level 2 SMX template.

    Parameters
    ----------
    n : int, optional
        Number of records (default: 1).

    Returns
    -------
    records : numpy.ndarray
        Array filled with default values.
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


def get_resample_template_ASCATL2_SMX(n=1):
    """
    Generic Level 2 SMX template.

    Parameters
    ----------
    n : int, optional
        Number of records (default: 1).

    Returns
    -------
    records : numpy.ndarray
        Array filled with default values.
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
                       ('num_obs', np.uint16)], metadata=metadata)

    record = np.array([(float64_nan, byte_nan, uint32_nan, uint32_nan,
                        uint8_nan, uint16_nan, byte_nan, byte_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, uint16_nan)], dtype=struct)

    return np.repeat(record, n)
