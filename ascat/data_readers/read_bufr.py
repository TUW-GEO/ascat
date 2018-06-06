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

import numpy as np
import pandas as pd

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

from ascat.bufr import BUFRReader


class AscatL1SsmBufrFile(ImageBase):
    """
    Reads ASCAT SSM swath files in BUFR format. There are the
    following products:

    Parameters
    ----------
    filename : str
        Filename path.
    mode : str, optional
        Opening mode. Default: r
    msg_name_lookup: dict, optional
        Dictionary mapping bufr msg number to parameter name. See :ref:`ascatformattable`.

        Default:

             === =====================================================
             Key Value
             === =====================================================
             6   'Direction Of Motion Of Moving Observing Platform',
             16  'Orbit Number',
             65  'Surface Soil Moisture (Ms)',
             66  'Estimated Error In Surface Soil Moisture',
             67  'Backscatter',
             68  'Estimated Error In Sigma0 At 40 Deg Incidence Angle',
             69  'Slope At 40 Deg Incidence Angle',
             70  'Estimated Error In Slope At 40 Deg Incidence Angle',
             71  'Soil Moisture Sensitivity',
             72  'Dry Backscatter',
             73  'Wet Backscatter',
             74  'Mean Surface Soil Moisture',
             75  'Rain Fall Detection',
             76  'Soil Moisture Correction Flag',
             77  'Soil Moisture Processing Flag',
             78  'Soil Moisture Quality',
             79  'Snow Cover',
             80  'Frozen Land Surface Fraction',
             81  'Inundation And Wetland Fraction',
             82  'Topographic Complexity'
             === =====================================================
    """

    def __init__(self, filename, mode='r', msg_name_lookup=None, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL1SsmBufrFile, self).__init__(filename, mode=mode,
                                                 **kwargs)
        if msg_name_lookup is None:
            msg_name_lookup = {
                6: "Direction Of Motion Of Moving Observing Platform",
                16: "Orbit Number",
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

                df = pd.to_datetime(pd.DataFrame({'month': months,
                                                  'year': years,
                                                  'day': days,
                                                  'hour': hours,
                                                  'minute': minutes,
                                                  'second': seconds}))
                dates.append(pd.DatetimeIndex(df).to_julian_date().values)

                # read optional data fields
                for mid in self.msg_name_lookup:
                    name = self.msg_name_lookup[mid]

                    if name not in data:
                        data[name] = []

                    data[name].append(message[:, mid - 1])

        dates = np.concatenate(dates)
        longitude = np.concatenate(longitude)
        latitude = np.concatenate(latitude)

        for mid in self.msg_name_lookup:
            name = self.msg_name_lookup[mid]
            data[name] = np.concatenate(data[name])
            if mid == 74:
                # ssm mean is encoded differently
                data[name] = data[name] * 100

        data['jd'] = dates

        if 65 in self.msg_name_lookup:
            # mask all the arrays based on fill_value of soil moisture
            valid_data = np.where(data[self.msg_name_lookup[65]] != 1.7e+38)
            latitude = latitude[valid_data]
            longitude = longitude[valid_data]
            for name in data:
                data[name] = data[name][valid_data]

        return Image(longitude, latitude, data, {}, timestamp, timekey='jd')

    def read_masked_data(self, **kwargs):
        """
        It does not make sense to read a orbit file unmasked
        so we only have a masked implementation.
        """
        return self.read(**kwargs)

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