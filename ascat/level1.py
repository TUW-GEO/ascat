# Copyright (c) 2017,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
General Level 1 data readers for ASCAT data in netCDF format.
'''

from netCDF4 import Dataset
from netCDF4 import netcdftime
import numpy as np
from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image


class AscatL1NcFile(ImageBase):
    """
    Reads ASCAT Level 1 backscatter data from netCDF files as
    distributed by EUMETSAT

    Parameters
    ----------
    filename: string
        Filename on disk
    mode: str, optional
        Opening mode. Default: r
    satellite_id_translation: boolean, optional
        if set the satellite id will be translated according to the
        following scheme:

        SAT           | ID
        -------------------
        ERS-1         |  1
        ERS-2         |  2
        MetOp-A       |  3
        MetOp-B       |  4
        MetOp-C       |  5
        MetOp-SG-B-1  |  6
        MetOp-SG-B-2  |  7
        MetOp-SG-B-3  |  8
    """

    def __init__(self, filename, mode='r',
                 satellite_id_translation=False, **kwargs):
        """
        Initialization of i/o object.

        """
        super(AscatL1NcFile, self).__init__(filename, mode=mode,
                                            **kwargs)
        self.beams = ['f', 'm', 'a']
        self.satellite_id_translation = satellite_id_translation

        # EUMETSAT identifies MetOp-B with sat-id = 1
        # EUMETSAT identifies MetOp-A with sat-id = 2
        # EUMETSAT identifies MetOp-C with sat-id = 3
        self.sat_id_lookup = {1: 4,
                              2: 3,
                              3: 5}

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
        name_translation = {'utc_line_nodes': 'jd',
                            'start_orbit_number': 'abs_orbit_nr',
                            'platform': 'spacecraft_id',
                            'num_val_trip': 'num_val',
                            'azi_angle_trip': 'azi',
                            'inc_angle_trip': 'inc',
                            'product_minor_version': 'processor_minor_version',
                            'sigma0_trip': 'sig'}
        with Dataset(self.filename, mode=self.mode) as ds:

            data = {}
            nodes = ds.dimensions['numCells'].size
            rows = ds.dimensions['numRows'].size
            two_d_fields = ['swath_indicator']
            for field in two_d_fields:
                data[field] = ds.variables[field][:].flatten()

            one_d_fields = ['as_des_pass', 'utc_line_nodes', 'sat_track_azi']
            for field in one_d_fields:
                arr = ds.variables[field][:]
                if field == 'utc_line_nodes':
                    units = ds.variables[field].units
                    utime = netcdftime.utime(units)
                    dt = utime.num2date(arr)
                    jd = netcdftime.JulianDayFromDate(dt)
                    arr = jd

                data[field] = arr.repeat(nodes)

            attribute_fields = ['platform', 'start_orbit_number']
            attribute_conversion = {'platform': lambda x: int(x[2]),
                                    'start_orbit_number': lambda x: x}
            for field in attribute_fields:
                attribute = attribute_conversion[field](ds.getncattr(field))
                arr = np.empty(nodes * rows, dtype=type(attribute))
                arr.fill(attribute)
                data[field] = arr

            data['node_num'] = np.arange(1, nodes + 1).repeat(rows)
            data['line_num'] = np.arange(rows).repeat(nodes)

            fields = ['azi_angle_trip', 'inc_angle_trip',
                      'sigma0_trip', 'kp', 'f_land', 'f_usable', 'f_kp',
                      'f_f', 'f_v', 'f_oa', 'f_sa', 'f_tel',
                      'f_ref', 'num_val_trip']
            for field in fields:
                for i, beam in enumerate(self.beams):
                    arr = ds.variables[field][:, :, i].flatten()

                    field_name = field
                    if field in name_translation:
                        field_name = name_translation[field]

                    data[field_name + beam] = arr

            fields = ['processor_major_version',
                      'product_minor_version', 'format_major_version',
                      'format_minor_version']
            metadata = {}
            for field in fields:
                metadata[field] = ds.getncattr(field)

            lons = ds.variables['longitude'][:].flatten()
            lats = ds.variables['latitude'][:].flatten()

            for field in data:
                if field in name_translation:
                    data[name_translation[field]] = data[field]
                    del data[field]

            for field in metadata:
                if field in name_translation:
                    metadata[name_translation[field]] = metadata[field]
                    del metadata[field]

        if self.satellite_id_translation:
            old_id = data['spacecraft_id'][0]
            new_id = self.sat_id_lookup[old_id]
            data['spacecraft_id'].fill(new_id)

        return Image(lons, lats, data, metadata,
                     timestamp, timekey='jd')

    def read_masked_data(self, **kwargs):
        orbit = self.read(**kwargs)

        valid = np.ones(orbit.data[orbit.data.keys()[0]].shape, dtype=np.bool)

        for b in self.beams:
            valid = (valid & (orbit.data['f_usable' + b] < 2))
            valid = (valid & (orbit.data['f_land' + b] > 95))
        for key in orbit.data.keys():
            orbit.data[key] = orbit.data[key][valid]
        for key in orbit.metadata.keys():
            orbit.metadata[key] = orbit.metadata[key][valid]
        orbit.lon = orbit.lon[valid]
        orbit.lat = orbit.lat[valid]
        return orbit

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass

    def resample_data(self, data, index, distance, weights, **kwargs):
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
        resOrbit : dict
            dictionary containing resampled data
        """
        resOrbit = {}

        # get weights
        total_weights = np.nansum(weights, axis=1)

        # resample backscatter
        sigmaNought = ['sigf', 'sigm', 'siga']
        for sigma in sigmaNought:
            resOrbit[sigma] = lin2db(np.nansum(db2lin(data[sigma])[index] * weights,
                                               axis=1) / total_weights)

        # resample measurement geometry
        measgeos = ['incf', 'incm', 'inca', 'azif', 'azim', 'azia']
        for mg in measgeos:
            resOrbit[mg] = (np.nansum(data[mg][index] * weights,
                                      axis=1) / total_weights)

        # noise estimate
        noise = ['kpf', 'kpm', 'kpa']
        for n in noise:
            resOrbit[n] = (np.nansum(data[n][index] * weights,
                                     axis=1) / total_weights)

        # nearest neighbour resampling values
        nnResample = ['jd', 'swath', 'node_num', 'line_num',
                      'abs_orbit_nr']
        data_names = ['jd', 'swath_indicator', 'node_num', 'line_num',
                      'abs_orbit_nr']
        # index of min. distance is equal to 0 because of kd-tree usage
        for nn, dn in zip(nnResample, data_names):
            resOrbit[nn] = data[dn][index][:, 0]

        resOrbit['dir'] = np.empty(data['as_des_pass'].shape,
                                   dtype=np.dtype('S1'))
        # set as_des_pass as string
        resOrbit['dir'][data['as_des_pass'][index][:, 0] == 1] = 'D'
        resOrbit['dir'][data['as_des_pass'][index][:, 0] == 0] = 'A'

        # set number of measurements for resampling
        resOrbit['num_obs'] = np.sum(distance != np.inf, axis=1)
        sat_id = data['spacecraft_id'][index][:, 0]
        resOrbit['sat_id'] = sat_id

        return resOrbit


def db2lin(val):
    """
    Converting from linear to dB domain.

    Parameters
    ----------
    val : numpy.ndarray
        Values in dB domain.

    Returns
    -------
    val : numpy.ndarray
        Values in linear domain.
    """
    return 10 ** (val / 10.)


def lin2db(val):
    """
    Converting from linear to dB domain.

    Parameters
    ----------
    val : numpy.ndarray
        Values in linear domain.

    Returns
    -------
    val : numpy.ndarray
        Values in dB domain.
    """
    return 10. * np.log10(val)
