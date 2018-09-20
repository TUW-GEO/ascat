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
General Level 1 data readers for ASCAT data in all formats.
Not specific to distributor.
"""
import os
import numpy as np
from datetime import datetime

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

import ascat.read_native.eps_native as eps_native
import ascat.read_native.bufr as bufr
import ascat.read_native.nc as nc
import ascat.read_native.hdf5 as h5
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

        if file_format is None:
            file_format = get_file_format(self.filename)

        if file_format == ".nat":
            if native:
                img = eps_native.AscatL1bEPSImage(self.filename).read(
                    timestamp)
            else:
                img_raw = eps_native.AscatL1bEPSImage(self.filename).read(
                    timestamp)
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
                "Format not found, please indicate the file_format. "
                "[\".nat\", \".nc\", \".bfr\", \".h5\"]")

        return img

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class AscatL1Bufr(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level1 images in bufr format.

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

    def __init__(self, path,
                 month_path_str='',
                 day_search_str='*-ASCA-*-NA-*-%Y%m%d*-*-*.bfr',
                 file_search_str='*-ASCA-*-NA-*-{datetime}.*.bfr',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(26, 40, '%Y%m%d%H%M%S'),
                 msg_name_lookup=None,
                 eo_portal=False):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.datetime_format = datetime_format
        self.filename_datetime_format = filename_datetime_format
        self.eo_portal = eo_portal
        super(AscatL1Bufr, self).__init__(path, AscatL1Image,
                                          subpath_templ=[month_path_str],
                                          fname_templ=file_search_str,
                                          datetime_format=datetime_format,
                                          exact_templ=False,
                                          ioclass_kws={
                                              'msg_name_lookup':
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


class AscatL1Eps(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level1 images in eps format.

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

    def __init__(self, path,
                 month_path_str='',
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
        super(AscatL1Eps, self).__init__(path, AscatL1Image,
                                         subpath_templ=[month_path_str],
                                         fname_templ=file_search_str,
                                         datetime_format=datetime_format,
                                         exact_templ=False)


class AscatL1Hdf5(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level1 images in hdf5 format.

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
        Default: 'ASCA_*_*_*_%Y%m%d*_*_*_*_*.h5'
    file_search_str: string, optional
        this string is used to find a bufr file by the exact date.
        Default: 'ASCA_*_*_*_{datetime}Z_*_*_*_*.h5'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d%H%M%S
    eo_portal : boolean optional
        If your data is from the EUMETSAT EO portal you can set this flag to
        True. This way the the datetime can automatically be read from the
        filename. Otherwise it needs the filename_datetime_format class
        variable set correctly.
    """

    def __init__(self, path,
                 month_path_str='',
                 day_search_str='ASCA_*_*_*_%Y%m%d*_*_*_*_*.h5',
                 file_search_str='ASCA_*_*_*_{datetime}Z_*_*_*_*.h5',
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
        super(AscatL1Hdf5, self).__init__(path, AscatL1Image,
                                          subpath_templ=[month_path_str],
                                          fname_templ=file_search_str,
                                          datetime_format=datetime_format,
                                          exact_templ=False)


class AscatL1Nc(ASCAT_MultiTemporalImageBase):
    """
    Class for reading multiple ASCAT level1 images in nc format.

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

    def __init__(self, path,
                 month_path_str='',
                 day_search_str='W_XX-*_EUMP_%Y%m%d*.nc',
                 file_search_str='W_XX-*_EUMP_{datetime}*.nc',
                 datetime_format='%Y%m%d%H%M%S',
                 filename_datetime_format=(62, 76, '%Y%m%d%H%M%S'),
                 msg_name_lookup=None,
                 eo_portal=False):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.datetime_format = datetime_format
        self.filename_datetime_format = filename_datetime_format
        self.eo_portal = eo_portal
        super(AscatL1Nc, self).__init__(path, AscatL1Image,
                                        subpath_templ=[month_path_str],
                                        fname_templ=file_search_str,
                                        datetime_format=datetime_format,
                                        exact_templ=False,
                                        ioclass_kws={
                                            'msg_name_lookup':
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


def eps2generic(native_Image):
    """
    Convert the native eps Image into a generic one.
    Use different convert-function for szf and szx data
    """
    if type(native_Image) is dict:
        img = eps_dict2generic(native_Image)
    else:
        img = eps_img2generic(native_Image)

    return img


def eps_img2generic(native_Image):
    """
    Convert single eps Images into a generic one
    """
    n_records = native_Image.lat.shape[0]
    generic_data = get_template_ASCATL1B_SZX(n_records)

    fields = [('jd', 'jd', None),
              ('sat_id', None, None),
              ('abs_line_nr', None, None),
              ('abs_orbit_nr', None, None),
              ('node_num', 'NODE_NUM', None),
              ('line_num', 'LINE_NUM', None),
              ('as_des_pass', 'AS_DES_PASS', None),
              ('swath', 'SWATH INDICATOR', byte_nan),
              ('azif', 'f_AZI_ANGLE_TRIP', int_nan),
              ('azim', 'm_AZI_ANGLE_TRIP', int_nan),
              ('azia', 'a_AZI_ANGLE_TRIP', int_nan),
              ('incf', 'f_INC_ANGLE_TRIP', uint16_nan),
              ('incm', 'm_INC_ANGLE_TRIP', uint16_nan),
              ('inca', 'a_INC_ANGLE_TRIP', uint16_nan),
              ('sigf', 'f_SIGMA0_TRIP', long_nan),
              ('sigm', 'm_SIGMA0_TRIP', long_nan),
              ('siga', 'a_SIGMA0_TRIP', long_nan),
              ('kpf', 'f_KP', uint16_nan),
              ('kpm', 'm_KP', uint16_nan),
              ('kpa', 'a_KP', uint16_nan),
              ('kpf_quality', 'f_F_KP', byte_nan),
              ('kpm_quality', 'm_F_KP', byte_nan),
              ('kpa_quality', 'a_F_KP', byte_nan),
              ('land_flagf', 'f_F_LAND', uint16_nan),
              ('land_flagm', 'm_F_LAND', uint16_nan),
              ('land_flaga', 'a_F_LAND', uint16_nan),
              ('usable_flagf', 'f_F_USABLE', byte_nan),
              ('usable_flagm', 'm_F_USABLE', byte_nan),
              ('usable_flaga', 'a_F_USABLE', byte_nan),
              ]
    for field in fields:
        if field[1] is None:
            continue

        if field[2] is not None:
            valid_mask = (native_Image.data[field[1]] != field[2])
            generic_data[field[0]][valid_mask] = native_Image.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_Image.data[field[1]]

    if 'ABS_LINE_NUMBER' in native_Image.data:
        generic_data['abs_line_nr'] = native_Image.data['ABS_LINE_NUMBER']

    fields = [('sat_id', 'SPACECRAFT_ID'),
              ('abs_orbit_nr', 'ORBIT_START')]
    for field in fields:
        generic_data[field[0]] = native_Image.metadata[field[1]].repeat(
            n_records)

    # convert sat_id (spacecraft id) to the department intern definition
    # use an array as look up table
    sat_id_lut = np.array([0, 4, 3, 5])
    generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

    img = Image(native_Image.lon, native_Image.lat, generic_data,
                native_Image.metadata, native_Image.timestamp,
                timekey='jd')
    return img


def eps_dict2generic(native_Image):
    """
    Convert dict of eps Images into a dict with generic Images
    """
    img = {'img1': {}, 'img2': {}, 'img3': {}, 'img4': {}, 'img5': {},
           'img6': {}}
    for szf_img in native_Image:
        n_records = native_Image[szf_img].lat.shape[0]
        generic_data = get_template_ASCATL1B_SZF(n_records)

        fields = [('jd', 'jd'),
                  ('sat_id', None),
                  ('beam_number', 'BEAM_NUMBER'),
                  ('abs_orbit_nr', None),
                  ('node_num', 'node_num'),
                  ('line_num', 'line_num'),
                  ('as_des_pass', 'AS_DES_PASS'),
                  ('azi', 'AZI_ANGLE_FULL'),
                  ('inc', 'INC_ANGLE_FULL'),
                  ('sig', 'SIGMA0_FULL'),
                  ('land_frac', 'LAND_FRAC'),
                  ('flagfield_rf1', 'FLAGFIELD_RF1'),
                  ('flagfield_rf2', 'FLAGFIELD_RF2'),
                  ('flagfield_pl', 'FLAGFIELD_PL'),
                  ('flagfield_gen1', 'FLAGFIELD_GEN1'),
                  ('flagfield_gen2', 'FLAGFIELD_GEN2'),
                  ('land_flag', 'F_LAND'),
                  ('usable_flag', 'F_USABLE')
                  ]
        for field in fields:
            if field[1] is None:
                continue
            generic_data[field[0]] = native_Image[szf_img].data[field[1]]

        fields = [('sat_id', 'SPACECRAFT_ID'),
                  ('abs_orbit_nr', 'ORBIT_START')]
        for field in fields:
            generic_data[field[0]] = native_Image[szf_img].metadata[
                field[1]].repeat(n_records)

        # convert sat_id (spacecraft id) to the department intern definition
        # use an array as look up table
        sat_id_lut = np.array([0, 4, 3, 5])
        generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

        img[szf_img] = Image(native_Image[szf_img].lon,
                             native_Image[szf_img].lat,
                             generic_data,
                             native_Image[szf_img].metadata,
                             native_Image[szf_img].timestamp,
                             timekey='jd')
    return img


def nc2generic(native_Image):
    """
    Convert the native nc Image into a generic one.
    """
    n_records = native_Image.lat.shape[0]
    generic_data = get_template_ASCATL1B_SZX(n_records)

    fields = [('jd', 'jd'),
              ('sat_id', None),
              ('abs_line_nr', 'abs_line_number'),
              ('abs_orbit_nr', None),
              ('node_num', 'node_num'),
              ('line_num', 'line_num'),
              ('as_des_pass', 'as_des_pass'),
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
              ('kpf_quality', 'f_f_kp'),
              ('kpm_quality', 'm_f_kp'),
              ('kpa_quality', 'a_f_kp'),
              ('land_flagf', 'f_f_land'),
              ('land_flagm', 'm_f_land'),
              ('land_flaga', 'a_f_land'),
              ('usable_flagf', 'f_f_usable'),
              ('usable_flagm', 'm_f_usable'),
              ('usable_flaga', 'a_f_usable'),
              ]
    for field in fields:
        if field[1] is None:
            continue

        if type(native_Image.data[field[1]]) == np.ma.core.MaskedArray:
            valid_mask = ~native_Image.data[field[1]].mask
            generic_data[field[0]][valid_mask] = native_Image.data[field[1]][
                valid_mask]
        else:
            generic_data[field[0]] = native_Image.data[field[1]]

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


def bfr2generic(native_Image):
    """
    Convert the native bfr Image into a generic one.
    """
    n_records = native_Image.lat.shape[0]
    generic_data = get_template_ASCATL1B_SZX(n_records)

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
              ('kpf', 'f_Radiometric Resolution (Noise Value)', 1.7e+38),
              ('kpm', 'm_Radiometric Resolution (Noise Value)', 1.7e+38),
              ('kpa', 'a_Radiometric Resolution (Noise Value)', 1.7e+38),
              ('kpf_quality', 'f_ASCAT KP Estimate Quality', None),
              ('kpm_quality', 'm_ASCAT KP Estimate Quality', None),
              ('kpa_quality', 'a_ASCAT KP Estimate Quality', None),
              ('land_flagf', 'f_ASCAT Land Fraction', None),
              ('land_flagm', 'm_ASCAT Land Fraction', None),
              ('land_flaga', 'a_ASCAT Land Fraction', None),
              ('usable_flagf', 'f_ASCAT Sigma-0 Usability', None),
              ('usable_flagm', 'm_ASCAT Sigma-0 Usability', None),
              ('usable_flaga', 'a_ASCAT Sigma-0 Usability', None),
              ]
    kp_vars = ['kpf', 'kpm', 'kpa']
    for field in fields:
        if field[1] is None:
            continue

        if field[2] is not None:
            valid_mask = (native_Image.data[field[1]] != field[2])
            if field[0] in kp_vars:
                generic_data[field[0]][valid_mask] = \
                    native_Image.data[field[1]][valid_mask] / 100
            else:
                generic_data[field[0]][valid_mask] = \
                    native_Image.data[field[1]][valid_mask]
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


def hdf2generic(native_Image):
    """
    Convert the native nc Image into a generic one.
    """
    img = {'img1': {}, 'img2': {}, 'img3': {}, 'img4': {}, 'img5': {},
           'img6': {}}
    for szf_img in native_Image:
        n_records = native_Image[szf_img].lat.shape[0]
        generic_data = get_template_ASCATL1B_SZF(n_records)

        fields = [('jd', 'jd'),
                  ('sat_id', None),
                  ('beam_number', 'BEAM_NUMBER'),
                  ('abs_orbit_nr', None),
                  ('node_num', 'node_num'),
                  ('line_num', 'line_num'),
                  ('as_des_pass', 'AS_DES_PASS'),
                  ('azi', 'AZI_ANGLE_FULL'),
                  ('inc', 'INC_ANGLE_FULL'),
                  ('sig', 'SIGMA0_FULL'),
                  ('land_frac', 'LAND_FRAC'),
                  ('flagfield_rf1', 'FLAGFIELD_RF1'),
                  ('flagfield_rf2', 'FLAGFIELD_RF2'),
                  ('flagfield_pl', 'FLAGFIELD_PL'),
                  ('flagfield_gen1', 'FLAGFIELD_GEN1'),
                  ('flagfield_gen2', 'FLAGFIELD_GEN2'),
                  ('land_flag', 'F_LAND'),
                  ('usable_flag', 'F_USABLE')
                  ]
        for field in fields:
            if field[1] is None:
                continue
            generic_data[field[0]] = native_Image[szf_img].data[field[1]]

        fields = [('sat_id', 'SPACECRAFT_ID'),
                  ('abs_orbit_nr', 'ORBIT_START')]
        for field in fields:
            generic_data[field[0]] = np.repeat(native_Image[szf_img].metadata[
                                                   field[1]], n_records)

        # convert sat_id (spacecraft id) to the department intern definition
        # use an array as look up table
        sat_id_lut = np.array([0, 4, 3, 5])
        generic_data['sat_id'] = sat_id_lut[generic_data['sat_id']]

        img[szf_img] = Image(native_Image[szf_img].lon,
                             native_Image[szf_img].lat,
                             generic_data,
                             native_Image[szf_img].metadata,
                             native_Image[szf_img].timestamp,
                             timekey='jd')

    return img


def get_template_ASCATL1B_SZX(n=1):
    """
    Generic lvl1b SZX template.
    """
    metadata = {'temp_name': 'ASCATL1B_SZX'}

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
                       ('kpf', np.float32),
                       ('kpm', np.float32),
                       ('kpa', np.float32),
                       ('kpf_quality', np.float32),
                       ('kpm_quality', np.float32),
                       ('kpa_quality', np.float32),
                       ('land_flagf', np.uint8),
                       ('land_flagm', np.uint8),
                       ('land_flaga', np.uint8),
                       ('usable_flagf', np.uint8),
                       ('usable_flagm', np.uint8),
                       ('usable_flaga', np.uint8),
                       ], metadata=metadata)

    record = np.array([(float64_nan, byte_nan, uint32_nan, uint32_nan,
                        uint8_nan, uint16_nan, byte_nan, byte_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, float32_nan,
                        float32_nan, float32_nan, uint8_nan, uint8_nan,
                        uint8_nan, uint8_nan, uint8_nan, uint8_nan)],
                      dtype=struct)

    return np.repeat(record, n)


def get_template_ASCATL1B_SZF(n=1):
    """
    Generic lvl1b SZF template.
    """
    metadata = {'temp_name': 'ASCATL1B_SZF'}

    struct = np.dtype([('jd', np.float64),
                       ('sat_id', np.byte),
                       ('beam_number', np.byte),
                       ('abs_orbit_nr', np.uint32),
                       ('node_num', np.uint8),
                       ('line_num', np.uint16),
                       ('as_des_pass', np.byte),
                       ('azi', np.float32),
                       ('inc', np.float32),
                       ('sig', np.float32),
                       ('land_frac', np.float32),
                       ('flagfield_rf1', np.uint8),
                       ('flagfield_rf2', np.uint8),
                       ('flagfield_pl', np.uint8),
                       ('flagfield_gen1', np.uint8),
                       ('flagfield_gen2', np.uint8),
                       ('land_flag', np.uint8),
                       ('usable_flag', np.uint8)], metadata=metadata)

    record = np.array([(float64_nan, byte_nan, byte_nan, uint32_nan,
                        uint8_nan, uint16_nan, byte_nan, float32_nan,
                        float32_nan, float32_nan, float32_nan, uint8_nan,
                        uint8_nan, uint8_nan, uint8_nan, uint8_nan, uint8_nan,
                        uint8_nan)], dtype=struct)

    return np.repeat(record, n)
