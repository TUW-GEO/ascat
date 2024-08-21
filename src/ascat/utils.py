# Copyright (c) 2024, TU Wien, Department of Geodesy and Geoinformation
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

from datetime import timedelta
from gzip import GzipFile
from tempfile import NamedTemporaryFile

import numpy as np
import xarray as xr

int8_nan = np.iinfo(np.int8).min
uint8_nan = np.iinfo(np.uint8).max
int16_nan = np.iinfo(np.int16).min
uint16_nan = np.iinfo(np.uint16).max
int32_nan = np.iinfo(np.int32).min
uint32_nan = np.iinfo(np.uint32).max
int64_nan = np.iinfo(np.int64).min
uint64_nan = np.iinfo(np.uint64).max
float32_nan = -999999.
float64_nan = -999999.

dtype_to_nan = {
    np.dtype('int8'): int8_nan,
    np.dtype('uint8'): uint8_nan,
    np.dtype('int16'): int16_nan,
    np.dtype('uint16'): uint16_nan,
    np.dtype('int32'): int32_nan,
    np.dtype('uint32'): uint32_nan,
    np.dtype('int64'): int64_nan,
    np.dtype('uint64'): uint64_nan,
    np.dtype('float32'): float32_nan,
    np.dtype('float64'): float64_nan,
    np.dtype('<U1'): None,
    np.dtype('O'): None,
}

def mask_dtype_nans(ds):
    """
    Mask NaNs in a dataset based on the dtypes of its variables.
    """
    for var in ds.data_vars:
        if ds[var].dtype in dtype_to_nan and ~ds[var].isnull().any():
            ds[var] = ds[var].where(ds[var] != dtype_to_nan[ds[var].dtype])
    return ds


def get_bit(a, bit_pos):
    """
    Returns 1 or 0 if bit is set or not.

    Parameters
    ----------
    a : int or numpy.ndarray
      Input array.
    bit_pos : int
      Bit position. First bit position is right.

    Returns
    -------
    b : numpy.ndarray
      1 if bit is set and 0 if not.
    """
    return np.clip(np.bitwise_and(a, 2**(bit_pos - 1)), 0, 1)


def set_bit(a, bit_pos, value=1):
    """
    Set bit at given position.

    Parameters
    ----------
    a : int or numpy.ndarray
      Input array.
    bit_pos : int
      Bit position. First bit starts right.
    value : 1 or 0, optional
      Set bit either to 1 or 0 (default: 1).

    Returns
    -------
    a : numpy.ndarray
      Modified input array with bit=value.
    """
    if value == 1:
        return np.bitwise_or(np.atleast_1d(a), 2**(bit_pos - 1))
    else:
        return np.bitwise_and(np.atleast_1d(a), ~(2**(bit_pos - 1)))


def daterange(start_date, end_date):
    """
    Generator for daily datetimes.

    Parameters
    ----------
    start_date : datetime
        Start date.
    end_date : datetime
        End date.
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def tmp_unzip(filename):
    """
    Unzip file to temporary directory.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    unzipped_filename : str
        Unzipped filename
    """
    with NamedTemporaryFile(delete=False) as tmp_fid:
        with GzipFile(filename) as gz_fid:
            tmp_fid.write(gz_fid.read())
        unzipped_filename = tmp_fid.name

    return unzipped_filename


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
    return 10**(val / 10.)


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


def get_window_radius(window, hp_radius):
    """
    Calculates the required radius of a window function in order to achieve
    the provided half power radius.

    Parameters
    ----------
    window : string
        Window function name.
        Current supported windows:
            - Hamming
            - Boxcar
    hp_radius : float32
        Half power radius. Radius of window function for weight
        equal to 0.5 (-3 dB). In the spatial domain this corresponds to
        half of the spatial resolution one would like to achieve with the
        given window.
    Returns
    -------
    r : float32
        Window radius needed to achieve the given half power radius

    """
    window = window.lower()
    hp_weight = 0.5
    if window == 'hamming':
        alpha = 0.54
        r = (np.pi * hp_radius) / np.arccos((hp_weight - alpha) / (1 - alpha))
    elif window == 'boxcar':
        r = hp_radius
    else:
        raise ValueError('Window name not supported.')

    return r


def hamming_window(radius, distances):
    """
    Hamming window filter.

    Parameters
    ----------
    radius : float32
        Radius of the window.
    distances : numpy.ndarray
        Array with distances.

    Returns
    -------
    weights : numpy.ndarray
        Distance weights.
    tw : float32
        Sum of weigths.
    """
    alpha = 0.54
    weights = alpha + (1 - alpha) * np.cos(np.pi / radius * distances)

    return weights, np.sum(weights)


def boxcar(radius, distance):
    """
    Boxcar filter

    Parameters
    ----------
    n : int
        Length.

    Returns
    -------
    weights : numpy.ndarray
        Distance weights.
    tw : float32
        Sum of weigths.
    """
    weights = np.zeros(distance.size)
    weights[distance <= radius] = 1.

    return weights, np.sum(weights)


def get_window_weights(window, radius, distance, norm=False):
    """
    Function returning weights for the provided window function

    Parameters
    ----------
    window : str
        Window function name
    radius : float
        Radius of the window.
    distance : numpy.ndarray
        Distance array
    norm : boolean
        If true, normalised weights will be returned.

    Returns
    -------
    weights : numpy.ndarray
        Weights according to distances and given window function

    """
    if window == 'hamming':
        weights, w_sum = hamming_window(radius, distance)
    elif window == 'boxcar':
        weights, w_sum = boxcar(radius, distance)
    else:
        raise ValueError('Window name not supported.')

    if norm is True:
        weights = weights / w_sum

    return weights


def get_toi_subset(ds, toi):
    """
    Filter dataset for given time of interest.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be filtered for time of interest.
    toi : tuple of datetime
        Time of interest.

    Returns
    -------
    ds : xarray.Dataset
        Filtered dataset.
    """
    if isinstance(ds, dict):
        for key in ds.keys():
            subset = np.where((ds[key]['time'] > np.datetime64(toi[0]))
                              & (ds[key]['time'] < np.datetime64(toi[1])))[0]
            if subset.size == 0:
                ds[key] = None
            else:
                if isinstance(ds[key], xr.Dataset):
                    ds[key] = ds[key].sel(obs=np.nonzero(subset.values)[0])
                elif isinstance(ds[key], np.ndarray):
                    ds[key] = ds[key][subset]
    else:
        subset = np.where((ds['time'] > np.datetime64(toi[0]))
                          & (ds['time'] < np.datetime64(toi[1])))[0]
        if subset.size == 0:
            ds = None
        else:
            if isinstance(ds, xr.Dataset):
                ds = ds.sel(obs=np.nonzero(subset.values)[0])
            elif isinstance(ds, np.ndarray):
                ds = ds[subset]

    return ds


def get_roi_subset(ds, roi):
    """
    Filter dataset for given region of interest.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be filtered for region of interest.
    roi : tuple of 4 float
        Region of interest: latmin, lonmin, latmax, lonmax

    Returns
    -------
    ds : xarray.Dataset
        Filtered dataset.
    """
    if isinstance(ds, dict):
        for key in ds.keys():
            subset = np.where((ds[key]['lat'] > roi[0])
                              & (ds[key]['lat'] < roi[2])
                              & (ds[key]['lon'] > roi[1])
                              & (ds[key]['lon'] < roi[3]))[0]
            if subset.size == 0:
                ds[key] = None
            else:
                if isinstance(ds[key], xr.Dataset):
                    ds[key] = ds[key].sel(obs=np.nonzero(subset.values)[0])
                elif isinstance(ds[key], np.ndarray):
                    ds[key] = ds[key][subset]
    else:
        subset = np.where((ds['lat'] > roi[0]) & (ds['lat'] < roi[2])
                          & (ds['lon'] > roi[1]) & (ds['lon'] < roi[3]))[0]
        if subset.size == 0:
            ds = None
        else:
            if isinstance(ds, xr.Dataset):
                ds = ds.sel(obs=np.nonzero(subset.values)[0])
            elif isinstance(ds, np.ndarray):
                ds = ds[subset]

    return ds


class Spacecraft:
    """
    Spacecraft class.
    """

    valid_spacecraft_names = [
        "METOPA", "METOPB", "METOPC", "METOP-A", "METOP-B", "METOP-C",
        "METOP-SG B1", "METOP-SG B2", "METOP-SG B3"
    ]

    def __init__(self, name):
        """
        Initialize spacecraft class.

        Parameters
        ----------
        name : str
            Spacecraft name.
        """
        if name not in Spacecraft.valid_spacecraft_names:
            valid_names = ' ,'.join(Spacecraft.valid_spacecraft_names)
            msg = f"Spacecraft {name} unknown. Valid options: {valid_names}"
            raise RuntimeError(msg)

        satellite_dict = {
            "METOPA": "METOPA",
            "METOPB": "METOPB",
            "METOPC": "METOPC",
            "METOP-A": "METOPA",
            "METOP-B": "METOPB",
            "METOP-C": "METOPC",
            "METOP-SG B1": "METOP-SGB1",
            "METOP-SG B2": "METOP-SGB2",
            "METOP-SG B3": "METOP-SGB3"
        }

        platform_dict = {
            "METOPA": "Metop",
            "METOPB": "Metop",
            "METOPC": "Metop",
            "METOP-SGB1": "Metop-SG",
            "METOP-SGB2": "Metop-SG",
            "METOP-SGB3": "Metop-SG"
        }

        sensor_dict = {"Metop": "ASCAT", "Metop-SG": "SCA"}

        sat_name_dict = {
            "METOPA": "A",
            "METOPB": "B",
            "METOPC": "C",
            "METOP-SGB1": "B1",
            "METOP-SGB2": "B2",
            "METOP-SGB3": "B3"
        }

        sat_id_dict = {
            "METOPA": 3,
            "METOPB": 4,
            "METOPC": 5,
            "METOP-SGB1": 6,
            "METOP-SGB2": 7,
            "METOP-SGB3": 8
        }

        self.satellite = satellite_dict[name]
        self.platform = platform_dict[self.satellite]
        self.sensor = sensor_dict[self.platform]
        self.sat_name = sat_name_dict[self.satellite]
        self.sat_id = sat_id_dict[self.satellite]
