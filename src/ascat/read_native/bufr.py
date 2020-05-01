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
Readers for lvl1b and lvl2 data in bufr format.
"""

import os
from datetime import datetime, timedelta
import warnings
from tempfile import NamedTemporaryFile
from gzip import GzipFile

import numpy as np

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.io_base import IntervalReadingMixin
from pygeobase.object_base import Image

try:
    from pybufr_ecmwf import raw_bufr_file
    from pybufr_ecmwf import ecmwfbufr
    from pybufr_ecmwf import ecmwfbufr_parameters
except ImportError:
    warnings.warn(
        'pybufr-ecmwf can not be imported, H08 and H07 images can '
        'not be read.')


class AscatL1BufrFile(ImageBase):
    """
    Reads ASCAT L1b data in BUFR format.

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name.
        See :ref:`ascatformattable`.

        Default:

             === =====================================================
             Key Value
             === =====================================================
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
             62: "a_ASCAT Land Fraction"
             === =====================================================
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None, **kwargs):
        """
        Initialization of i/o object.

        """
        zipped = False
        if os.path.splitext(filename)[1] == '.gz':
            zipped = True

        # for zipped files use an unzipped temporary copy
        if zipped:
            with NamedTemporaryFile(delete=False) as tmp_fid:
                with GzipFile(filename) as gz_fid:
                    tmp_fid.write(gz_fid.read())
                filename = tmp_fid.name

        super(AscatL1BufrFile, self).__init__(filename, mode=mode,
                                              **kwargs)
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

    def read(self, timestamp=None):
        """
        Read specific image for given datetime timestamp.

        Parameters
        ----------
        timestamp : datetime.datetime
            exact observation timestamp of the image that should be read

        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold the metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        time_var : string or None
            variable name of observation times in the data dict, if None all
            observations have the same timestamp
        """
        # lookup table between names and message number in the BUFR file

        data = {}
        dates = []
        # 13: Latitude (High Accuracy)
        latitude = []
        # 14: Longitude (High Accuracy)
        longitude = []

        with BUFRReader(self.filename) as bufr:
            for message in bufr.messages():
                # read fixed fields
                latitude.append(message[:, 12])
                longitude.append(message[:, 13])
                years = message[:, 6].astype(int)
                months = message[:, 7].astype(int)
                days = message[:, 8].astype(int)
                hours = message[:, 9].astype(int)
                minutes = message[:, 10].astype(int)
                seconds = message[:, 11].astype(int)

                dates.append(
                    julday(months, days, years, hours, minutes, seconds))

                # read optional data fields
                for mid in self.msg_name_lookup:
                    name = self.msg_name_lookup[mid]

                    if name not in data:
                        data[name] = []

                    data[name].append(message[:, mid - 1])

        dates = np.concatenate(dates)
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)
        n_records = latitude.shape[0]

        for mid in self.msg_name_lookup:
            name = self.msg_name_lookup[mid]
            data[name] = np.concatenate(data[name])

        data['jd'] = dates
        if 'Direction Of Motion Of Moving Observing Platform' in data:
            data['as_des_pass'] = (data[
                "Direction Of Motion Of Moving Observing Platform"
            ] < 270).astype(np.uint8)

        if 'Cross-Track Cell Number' in data:
            if data['Cross-Track Cell Number'].max() == 82:
                data['swath_indicator'] = 1 * (
                    data['Cross-Track Cell Number'] > 41)
            elif data['Cross-Track Cell Number'].max() == 42:
                data['swath_indicator'] = 1 * (
                    data['Cross-Track Cell Number'] > 21)
            else:
                raise ValueError("Unsuspected node number.")
            n_lines = n_records / max(data['Cross-Track Cell Number'])
            data['line_num'] = np.arange(n_lines).repeat(
                max(data['Cross-Track Cell Number']))

        # There are strange elements with a value of 32.32 instead of the
        # typical nan_values
        # Since some elements rly have this value we check the other triplet
        # data of that beam to filter the nan_values out
        beams = ['f', 'm', 'a']
        for beam in beams:
            azi = beam + '_Antenna Beam Azimuth'
            sig = beam + '_Backscatter'
            inc = beam + '_Radar Incidence Angle'
            if azi in data:
                mask_azi = data[azi] == 32.32
                mask_sig = data[sig] == 1.7e+38
                mask_inc = data[inc] == 1.7e+38
                mask = np.all([mask_azi, mask_sig, mask_inc], axis=0)
                data[azi][mask] = 1.7e+38

        # 1 ERS 1
        # 2 ERS 2
        # 3 METOP-1 (Metop-B)
        # 4 METOP-2 (Metop-A)
        # 5 METOP-3 (Metop-C)
        sat_id = {3: 1, 4: 2, 5: 3}

        metadata = {}
        metadata['SPACECRAFT_ID'] = np.int8(
            sat_id[data['Satellite Identifier'][0]])
        metadata['ORBIT_START'] = np.uint32(data['Orbit Number'][0])

        return Image(longitude, latitude, data, metadata,
                     timestamp, timekey='jd')

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class AscatL2SsmBufrFile(ImageBase):
    """
    Reads ASCAT SSM swath files in BUFR format. There are the
    following products:

    - H101 SSM ASCAT-A NRT O 12.5 Metop-A ASCAT NRT SSM orbit geometry
    12.5 km sampling
    - H102 SSM ASCAT-A NRT O 25.0 Metop-A ASCAT NRT SSM orbit geometry
    25 km sampling
    - H16  SSM ASCAT-B NRT O 12.5 Metop-B ASCAT NRT SSM orbit geometry
    12.5 km sampling
    - H103 SSM ASCAT-B NRT O 25.0 Metop-B ASCAT NRT SSM orbit geometry
    25 km sampling
    - H104 SSM ASCAT-C NRT O 12.5 Metop-C ASCAT NRT SSM orbit geometry
    12.5 km sampling
    - H105 SSM ASCAT-C NRT O 25.0 Metop-C ASCAT NRT SSM orbit geometry
    25 km sampling
    - EUMETSAT ASCAT Soil Moisture at 12.5 km Swath Grid - Metop in BUFR format
    - EUMETSAT ASCAT Soil Moisture at 25.0 km Swath Grid - Metop in BUFR format

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name.
        See :ref:`ascatformattable`.

        Default:

             === =====================================================
             Key Value
             === =====================================================
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
             82: "Topographic Complexity"
             === =====================================================
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL2SsmBufrFile, self).__init__(filename, mode=mode,
                                                 **kwargs)
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

    def read(self, timestamp=None, ssm_masked=False):
        """
        Read specific image for given datetime timestamp.

        Parameters
        ----------
        timestamp : datetime.datetime (optional)
            exact observation timestamp of the image that should be read
        ssm_masked : flag (optional)
            set to True to filter data by ssm values


        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold the metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        time_var : string or None
            variable name of observation times in the data dict, if None all
            observations have the same timestamp
        """
        # lookup table between names and message number in the BUFR file

        data = {}
        dates = []
        # 13: Latitude (High Accuracy)
        latitude = []
        # 14: Longitude (High Accuracy)
        longitude = []

        with BUFRReader(self.filename) as bufr:
            for message in bufr.messages():
                # read fixed fields
                latitude.append(message[:, 12])
                longitude.append(message[:, 13])
                years = message[:, 6].astype(int)
                months = message[:, 7].astype(int)
                days = message[:, 8].astype(int)
                hours = message[:, 9].astype(int)
                minutes = message[:, 10].astype(int)
                seconds = message[:, 11].astype(int)

                dates.append(
                    julday(months, days, years, hours, minutes, seconds))

                # read optional data fields
                for mid in self.msg_name_lookup:
                    name = self.msg_name_lookup[mid]

                    if name not in data:
                        data[name] = []

                    data[name].append(message[:, mid - 1])

        dates = np.concatenate(dates)
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)
        n_records = latitude.shape[0]

        for mid in self.msg_name_lookup:
            name = self.msg_name_lookup[mid]
            data[name] = np.concatenate(data[name])
            if mid == 74:
                # ssm mean is encoded differently
                data[name] = data[name] * 100

        data['jd'] = dates

        if 'Direction Of Motion Of Moving Observing Platform' in data:
            data['as_des_pass'] = (data[
                "Direction Of Motion Of Moving Observing Platform"]
                <= 270).astype(np.uint8)

        if 'Cross-Track Cell Number' in data:
            if data['Cross-Track Cell Number'].max() == 82:
                data['swath_indicator'] = 1 * (
                    data['Cross-Track Cell Number'] > 41)
            elif data['Cross-Track Cell Number'].max() == 42:
                data['swath_indicator'] = 1 * (
                    data['Cross-Track Cell Number'] > 21)
            else:
                raise ValueError("Unsuspected node number.")
            n_lines = n_records / max(data['Cross-Track Cell Number'])
            data['line_num'] = np.arange(n_lines).repeat(
                max(data['Cross-Track Cell Number']))

        # There are strange elements with a value of 32.32 instead of the
        # typical nan_values
        # Since some elements rly have this value we check the other triplet
        # data of that beam to filter the nan_values out
        beams = ['f', 'm', 'a']
        for beam in beams:
            azi = beam + '_Antenna Beam Azimuth'
            sig = beam + '_Backscatter'
            inc = beam + '_Radar Incidence Angle'
            if azi in data:
                mask_azi = data[azi] == 32.32
                mask_sig = data[sig] == 1.7e+38
                mask_inc = data[inc] == 1.7e+38
                mask = np.all([mask_azi, mask_sig, mask_inc], axis=0)
                data[azi][mask] = 1.7e+38

        # if the ssm_masked is True we mask out data with missing ssm value
        if 65 in self.msg_name_lookup and ssm_masked is True:
            # mask all the arrays based on fill_value of soil moisture
            valid_data = np.where(data[self.msg_name_lookup[65]] != 1.7e+38)
            latitude = latitude[valid_data]
            longitude = longitude[valid_data]
            for name in data:
                data[name] = data[name][valid_data]

        # 1 ERS 1
        # 2 ERS 2
        # 3 METOP-1 (Metop-B)
        # 4 METOP-2 (Metop-A)
        # 5 METOP-3 (Metop-C)
        sat_id = {3: 1, 4: 2, 5: 3}

        metadata = {}

        try:
            metadata['SPACECRAFT_ID'] = np.int8(
                sat_id[data['Satellite Identifier'][0]])
        except KeyError:
            metadata['SPACECRAFT_ID'] = 0

        try:
            metadata['ORBIT_START'] = np.uint32(data['Orbit Number'][0])
        except KeyError:
            metadata['ORBIT_START'] = 0

        return Image(longitude, latitude, data, metadata,
                     timestamp, timekey='jd')

    def resample_data(self, image, index, distance, weights, **kwargs):
        """
        Takes an image and resample (interpolate) the image data to
        arbitrary defined locations given by index and distance.

        Parameters
        ----------
        image : object
            pygeobase.object_base.Image object
        index : np.array
            Index into image data defining a look-up table for data elements
            used in the interpolation process for each defined target
            location.
        distance : np.array
            Array representing the distances of the image data to the
            arbitrary defined locations.
        weights : np.array
            Array representing the weights of the image data that should be
            used during resampling.
            The weights of points not to use are set to np.nan
            This array is of shape (x, max_neighbors)

        Returns
        -------
        image : object
            pygeobase.object_base.Image object
        """
        total_weights = np.nansum(weights, axis=1)

        resOrbit = {}
        # resample backscatter
        for name in image.dtype.names:
            if name in ['Soil Moisture Correction Flag',
                        'Soil Moisture Processing Flag']:
                # The flags are resampled by taking the minimum flag This works
                # since any totally valid observation has the flag 0 and
                # overrides the flagged observations. This is true in cases
                # where the data was set to NaN by the flag as well as when the
                # data was set to 0 or 100. The last image element is the one
                # standing for NaN so we fill it with all flags filled to not
                # interfere with the minimum.
                image[name][-1] = 255
                bits = np.unpackbits(image[name].reshape(
                    (-1, 1)).astype(np.uint8), axis=1)
                resampled_bits = np.min(bits[index, :], axis=1)
                resOrbit[name] = np.packbits(resampled_bits)
            else:
                resOrbit[name] = np.nansum(
                    image[name][index] * weights, axis=1) / total_weights

        return resOrbit

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class AscatL2SsmBufr(MultiTemporalImageBase):
    """
    Class for reading HSAF ASCAT SSM images in bufr format.
    The images have the same structure as the ASCAT 3 minute pdu files
    and these 2 readers could be merged in the future
    The images have to be uncompressed in the following folder structure
    path - month_path_str (default 'h07_%Y%m_buf')

    For example if path is set to /home/user/hsaf07 and month_path_str is left
    to the default 'h07_%Y%m_buf' then the images for March 2012 have to be in
    the folder /home/user/hsaf07/h07_201203_buf/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the
        HSAF FTP Server then please specify the string that should be used in
        datetime.datetime.strftime Default: 'h07_%Y%m_buf'
    day_search_str: string, optional
        to provide an iterator over all images of a day the
        method _get_possible_timestamps looks for all available images on a day
        on the harddisk. This string is used in datetime.datetime.strftime and
        in glob.glob to search for all files on a day.
        Default : 'h07_%Y%m%d_*.buf'
    file_search_str: string, optional
        this string is used in datetime.datetime.strftime and glob.glob to find
        a 3 minute bufr file by the exact date.
        Default: 'h07_{datetime}*.buf'
    datetime_format: string, optional
        datetime format by which {datetime} will be replaced in file_search_str
        Default: %Y%m%d_%H%M%S
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name.
        See :ref:`ascatformattable`.

        Default:

             === =====================================================
             Key Value
             === =====================================================
             4:  "Satellite Identifier",
             6:  "Direction Of Motion Of Moving Observing Platform",
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
             82: "Topographic Complexity"
             === =====================================================
    """

    def __init__(self, path, month_path_str='h07_%Y%m_buf',
                 day_search_str='h07_%Y%m%d_*.buf',
                 file_search_str='h07_{datetime}*.buf',
                 datetime_format='%Y%m%d_%H%M%S',
                 filename_datetime_format=(4, 19, '%Y%m%d_%H%M%S'),
                 msg_name_lookup=None):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        self.filename_datetime_format = filename_datetime_format
        super(AscatL2SsmBufr, self).__init__(path, AscatL2SsmBufrFile,
                                             subpath_templ=[month_path_str],
                                             fname_templ=file_search_str,
                                             datetime_format=datetime_format,
                                             exact_templ=False,
                                             ioclass_kws={
                                                 'msg_name_lookup':
                                                     msg_name_lookup})

    def _get_orbit_start_date(self, filename):
        orbit_start_str = \
            os.path.basename(filename)[self.filename_datetime_format[0]:
                                       self.filename_datetime_format[1]]
        return datetime.strptime(orbit_start_str,
                                 self.filename_datetime_format[2])

    def tstamps_for_daterange(self, startdate, enddate):
        """
        Get the timestamps as datetime array that are possible for the
        given day.

        For this product it is not fixed but has to be looked up from
        the hard disk since bufr files are not regular spaced and only
        europe is in this product. For a global product a 3 minute
        spacing could be used as a fist approximation

        Parameters
        ----------
        startdate : datetime.date or datetime.datetime
            start date
        enddate : datetime.date or datetime.datetime
            end date

        Returns
        -------
        dates : list
            list of datetimes
        """
        file_list = []
        delta_all = enddate - startdate
        timestamps = []

        for i in range(delta_all.days + 1):
            timestamp = startdate + timedelta(days=i)

            files = self._search_files(
                timestamp, custom_templ=self.day_search_str)

            file_list.extend(sorted(files))

        for filename in file_list:
            timestamps.append(self._get_orbit_start_date(filename))

        timestamps = [dt for dt in timestamps if (
            startdate <= dt <= enddate)]
        return timestamps


class AscatL2SsmBufrChunked(IntervalReadingMixin, AscatL2SsmBufr):
    """
    Reads BUFR files but does not return them on a file by file basis but in
    bigger chunks. For example it allows to read multiple 3 minute PDU's in
    half orbit chunks of 50 minutes. This speeds up operations like e.g.
    resampling of the data.

    Parameters
    ----------
    chunk_minutes: int, optional
        How many minutes should a chunk of data cover.
    """

    def __init__(self, path, month_path_str='h07_%Y%m_buf',
                 day_search_str='h07_%Y%m%d_*.buf',
                 file_search_str='h07_{datetime}*.buf',
                 datetime_format='%Y%m%d_%H%M%S',
                 filename_datetime_format=(4, 19, '%Y%m%d_%H%M%S'),
                 msg_name_lookup=None, chunk_minutes=50):
        super(AscatL2SsmBufrChunked, self).__init__(
            path,
            month_path_str=month_path_str,
            day_search_str=day_search_str,
            file_search_str=file_search_str,
            datetime_format=datetime_format,
            filename_datetime_format=filename_datetime_format,
            msg_name_lookup=msg_name_lookup,
            chunk_minutes=chunk_minutes)


class BUFRReader(object):
    """
    BUFR reader based on the pybufr-ecmwf package but faster

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

    def __init__(self, filename, kelem_guess=500, max_tries=10):
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
            self.cvals = np.zeros((kvals, 80), dtype=np.character)
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


def julday(month, day, year, hour=0, minute=0, second=0):
    """
    Julian date from month, day, and year (can be scalars or arrays)
    (function from pytesmo)
    Parameters
    ----------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32
        Year.
    hour : numpy.ndarray or int32, optional
        Hour.
    minute : numpy.ndarray or int32, optional
        Minute.
    second : numpy.ndarray or int32, optional
        Second.
    Returns
    -------
    jul : numpy.ndarray or double
        Julian day.
    """
    month = np.array(month)
    day = np.array(day)
    inJanFeb = month <= 2
    jy = year - inJanFeb
    jm = month + 1 + inJanFeb * 12

    jul = np.int32(np.floor(365.25 * jy) +
                   np.floor(30.6001 * jm) + (day + 1720995.0))
    ja = np.int32(0.01 * jy)
    jul += 2 - ja + np.int32(0.25 * ja)

    jul = jul + hour / 24.0 - 0.5 + minute / 1440.0 + second / 86400.0

    return jul
