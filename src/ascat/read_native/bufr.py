# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Readers for ASCAT Level 1b and Level 2 data in BUFR format.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from cadati.cal_date import cal2dt

try:
    import eccodes
except ImportError:
    pass

from ascat.utils import tmp_unzip
from ascat.utils import mask_dtype_nans
from ascat.utils import uint8_nan
from ascat.utils import uint16_nan
from ascat.utils import int32_nan
from ascat.utils import float32_nan
from ascat.read_native import AscatFile

bufr_nan = 1.7e+38

nan_val_dict = {
    np.float32: float32_nan,
    np.uint8: uint8_nan,
    np.uint16: uint16_nan,
    np.int32: int32_nan
}


def read_bufr_data(filename, key_lookup):
    """
    Read selected fields from a BUFR file using eccodes array access.

    This reads the requested (rank-qualified) keys directly with
    ``codes_get_array`` instead of expanding every key of every subset, which
    is orders of magnitude faster than ``pdbufr.read_bufr(..., flat=True)`` for
    the large ASCAT BUFR messages.

    Parameters
    ----------
    filename : str
        BUFR filename.
    key_lookup : dict
        Mapping of output field name to the eccodes key to read, e.g.
        ``{"f_Backscatter": "#1#backscatter"}``. Keys yielding a single value
        per message (compressed scalars) are broadcast to all subsets.

    Returns
    -------
    data : pandas.DataFrame
        One row per observation with the requested fields plus ``lat``, ``lon``
        and ``time``.
    """
    time_keys = ("year", "month", "day", "hour", "minute", "second")
    columns = {name: [] for name in key_lookup}
    aux = {name: [] for name in ("lat", "lon", *time_keys)}

    with open(filename, "rb") as fh:
        while True:
            handle = eccodes.codes_bufr_new_from_file(fh)
            if handle is None:
                break
            try:
                eccodes.codes_set(handle, "unpack", 1)
                n_obs = eccodes.codes_get(handle, "numberOfSubsets")

                def get(key):
                    arr = np.atleast_1d(eccodes.codes_get_array(handle, key))
                    if arr.size == 1 and n_obs != 1:
                        arr = np.repeat(arr, n_obs)
                    return arr

                for name, key in key_lookup.items():
                    columns[name].append(get(key))
                aux["lat"].append(get("#1#latitude").astype(np.float32))
                aux["lon"].append(get("#1#longitude").astype(np.float32))
                for tkey in time_keys:
                    aux[tkey].append(get("#1#" + tkey).astype(int))
            finally:
                eccodes.codes_release(handle)

    data = {name: np.concatenate(parts) for name, parts in columns.items()}
    # eccodes returns its missing-value sentinel (~1.7e38) for absent values;
    # pdbufr's flat reader returned NaN, so match that for float fields.
    for name, arr in data.items():
        if np.issubdtype(arr.dtype, np.floating):
            arr[np.abs(arr) > 1e37] = np.nan

    data = pd.DataFrame(data)
    data["lat"] = np.concatenate(aux["lat"])
    data["lon"] = np.concatenate(aux["lon"])

    cal_dates = np.vstack(
        [np.concatenate(aux[tkey]) for tkey in time_keys]
        + [np.zeros(data.shape[0])]).T
    data["time"] = cal2dt(cal_dates)

    return data

class AscatL1bBufrFile(AscatFile):
    """
    Read ASCAT Level 1b file in BUFR format.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize AscatL1bBufrFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        super().__init__(filename, **kwargs)

        for i, fname in enumerate(self.filenames):
            if os.path.splitext(fname)[1] == '.gz':
                self.filenames[i] = tmp_unzip(fname)

        # Output field name -> rank-qualified eccodes key. The three antenna
        # beams (fore/mid/aft) share the same keys at successive ranks
        # (#1#/#2#/#3#).
        self.msg_key_lookup = {
                "Satellite Identifier": "#1#satelliteIdentifier",
                "Direction Of Motion Of Moving Observing Platform":
                    "#1#directionOfMotionOfMovingObservingPlatform",
                "Orbit Number": "#1#orbitNumber",
                "Cross-Track Cell Number": "#1#crossTrackCellNumber",
                "f_Beam Identifier": "#1#beamIdentifier",
                "f_Radar Incidence Angle": "#1#radarIncidenceAngle",
                "f_Antenna Beam Azimuth": "#1#antennaBeamAzimuth",
                "f_Backscatter": "#1#backscatter",
                "f_Radiometric Resolution (Noise Value)":
                    "#1#radiometricResolutionNoiseValue",
                "f_ASCAT KP Estimate Quality": "#1#ascatKpEstimateQuality",
                "f_ASCAT Sigma-0 Usability": "#1#ascatSigma0Usability",
                "f_ASCAT Land Fraction": "#1#landFraction",
                "m_Beam Identifier": "#2#beamIdentifier",
                "m_Radar Incidence Angle": "#2#radarIncidenceAngle",
                "m_Antenna Beam Azimuth": "#2#antennaBeamAzimuth",
                "m_Backscatter": "#2#backscatter",
                "m_Radiometric Resolution (Noise Value)":
                    "#2#radiometricResolutionNoiseValue",
                "m_ASCAT KP Estimate Quality": "#2#ascatKpEstimateQuality",
                "m_ASCAT Sigma-0 Usability": "#2#ascatSigma0Usability",
                "m_ASCAT Land Fraction": "#2#landFraction",
                "a_Beam Identifier": "#3#beamIdentifier",
                "a_Radar Incidence Angle": "#3#radarIncidenceAngle",
                "a_Antenna Beam Azimuth": "#3#antennaBeamAzimuth",
                "a_Backscatter": "#3#backscatter",
                "a_Radiometric Resolution (Noise Value)":
                    "#3#radiometricResolutionNoiseValue",
                "a_ASCAT KP Estimate Quality": "#3#ascatKpEstimateQuality",
                "a_ASCAT Sigma-0 Usability": "#3#ascatSigma0Usability",
                "a_ASCAT Land Fraction": "#3#landFraction",
            }

    def _read(self, filename, generic=False, to_xarray=False):
        """
        Read one ASCAT Level 1b BUFR file.

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
        data = read_bufr_data(filename, self.msg_key_lookup)
        data = data.to_records(index=False)
        data = {name:data[name] for name in data.dtype.names}

        metadata = {}
        metadata['platform_id'] = data['Satellite Identifier'][0].astype(int)
        metadata['orbit_start'] = np.uint32(data['Orbit Number'][0])
        metadata['filename'] = os.path.basename(filename)

        # add/rename/remove fields according to generic format
        if generic:
            data = conv_bufrl1b_generic(data, metadata)

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data.keys():
                if len(data[k].shape) == 1:
                    dim = ['obs']
                elif len(data[k].shape) == 2:
                    dim = ['obs', 'beam']

                data[k] = (dim, data[k])

            coords = {}
            coords_fields = ['lon', 'lat', 'time']
            for cf in coords_fields:
                coords[cf] = data.pop(cf)

            data = xr.Dataset(data, coords=coords, attrs=metadata)
            if generic:
                data = mask_dtype_nans(data)
        else:
            # collect dtype info
            dtype = []
            for var_name in data.keys():
                if len(data[var_name].shape) == 1:
                    dtype.append((var_name, data[var_name].dtype.str))
                elif len(data[var_name].shape) > 1:
                    dtype.append((var_name, data[var_name].dtype.str,
                                  data[var_name].shape[1:]))

            ds = np.empty(data['time'].size, dtype=np.dtype(dtype))
            for k, v in data.items():
                ds[k] = v

            data = ds

        return data, metadata

    def _merge(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : numpy.ndarray or xarray.Dataset
            Data.
        """
        if isinstance(data[0], tuple):
            data, metadata = zip(*data)
            if isinstance(data[0], xr.Dataset):
                data = xr.concat(data,
                                 dim="obs",
                                 combine_attrs="drop_conflicts")
            else:
                data = np.hstack(data)
            data = (data, metadata)
        else:
            data = np.hstack(data)
        return data

class AscatL1bBufrFileGeneric(AscatL1bBufrFile):
    """
    The same as AscatL1bBufrFile but with generic=True by default.
    """
    def _read(self, filename, generic=True, to_xarray=False, **kwargs):
        return super()._read(filename, generic=generic, to_xarray=to_xarray, **kwargs)


def conv_bufrl1b_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Spacecraft_id vs sat_id encoding

    BUFR encoding - Spacecraft_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-1 (Metop-B)
    - 4 Metop-2 (Metop-A)
    - 5 Metop-3 (Metop-C)

    Internal encoding - sat_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-2 (Metop-A)
    - 4 Metop-1 (Metop-B)
    - 5 Metop-3 (Metop-C)

    Parameters
    ----------
    data: dict of numpy.ndarray
        Original dataset.
    metadata: dict
        Metadata.

    Returns
    -------
    data: dict of numpy.ndarray
        Converted dataset.
    """
    skip_fields = ['Satellite Identifier']

    gen_fields_beam = {
        'Radar Incidence Angle': ('inc', np.float32, bufr_nan, 1),
        'Backscatter': ('sig', np.float32, bufr_nan, 1),
        'Antenna Beam Azimuth': ('azi', np.float32, bufr_nan, 1),
        'ASCAT Sigma-0 Usability': ('f_usable', np.uint8, None, 1),
        'Beam Identifier': ('beam_num', np.uint8, None, 1),
        'Radiometric Resolution (Noise Value)':
        ('kp', np.float32, bufr_nan, 0.01),
        'ASCAT KP Estimate Quality': ('kp_quality', np.uint8, bufr_nan, 1),
        'ASCAT Land Fraction': ('f_land', np.float32, None, 1)
    }

    gen_fields_lut = {
        'Orbit Number': ('abs_orbit_nr', np.int32),
        'Cross-Track Cell Number': ('node_num', np.uint8),
        'Direction Of Motion Of Moving Observing Platform':
        ('sat_track_azi', np.float32)
    }

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name, (new_name, new_dtype) in gen_fields_lut.items():
        data[new_name] = data.pop(var_name).astype(new_dtype)

    for var_name, (new_name, new_dtype, nan_val, s) in gen_fields_beam.items():
        f = ['{}_{}'.format(b, var_name) for b in ['f', 'm', 'a']]
        data[new_name] = np.vstack((data.pop(f[0]), data.pop(f[1]),
                                    data.pop(f[2]))).T.astype(new_dtype)
        if nan_val is not None:
            valid = data[new_name] != nan_val
            data[new_name][~valid] = nan_val_dict[new_dtype]
            data[new_name][valid] *= s

    if data['node_num'].max() == 82:
        data['swath_indicator'] = 1 * (data['node_num'] > 41)
    elif data['node_num'].max() == 42:
        data['swath_indicator'] = 1 * (data['node_num'] > 21)
    else:
        raise ValueError('Cross-track cell number size unknown')

    n_lines = data['lat'].shape[0] / data['node_num'].max()
    data['line_num'] = np.arange(n_lines).repeat(data['node_num'].max())

    sat_id = np.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
    data['sat_id'] = np.zeros(data['time'].size, dtype=np.uint8) + sat_id[int(
        metadata['platform_id'])]

    # compute ascending/descending direction
    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data


class AscatL2BufrFile(AscatFile):
    """
    Read ASCAT Level 2 file in BUFR format.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize AscatL2BufrFile.

        Parameters
        ----------
        filename: str
            Filename.
        """
        super().__init__(filename, **kwargs)

        for i, fname in enumerate(self.filenames):
            if os.path.splitext(fname)[1] == '.gz':
                self.filenames[i] = tmp_unzip(fname)

        # Output field name -> rank-qualified eccodes key. Beams use #1#/#2#/#3#;
        # the L2 sigma0/dry/wet backscatter reuse the backscatter key at higher
        # ranks (#4#/#5#/#6#).
        self.msg_key_lookup = {
            "Satellite Identifier": "#1#satelliteIdentifier",
            "Direction Of Motion Of Moving Observing Platform":
                "#1#directionOfMotionOfMovingObservingPlatform",
            "Orbit Number": "#1#orbitNumber",
            "Cross-Track Cell Number": "#1#crossTrackCellNumber",
            "f_Beam Identifier": "#1#beamIdentifier",
            "f_Radar Incidence Angle": "#1#radarIncidenceAngle",
            "f_Antenna Beam Azimuth": "#1#antennaBeamAzimuth",
            "f_Backscatter": "#1#backscatter",
            "f_Radiometric Resolution (Noise Value)":
                "#1#radiometricResolutionNoiseValue",
            "f_ASCAT KP Estimate Quality": "#1#ascatKpEstimateQuality",
            "f_ASCAT Sigma-0 Usability": "#1#ascatSigma0Usability",
            "f_ASCAT Land Fraction": "#1#landFraction",
            "m_Beam Identifier": "#2#beamIdentifier",
            "m_Radar Incidence Angle": "#2#radarIncidenceAngle",
            "m_Antenna Beam Azimuth": "#2#antennaBeamAzimuth",
            "m_Backscatter": "#2#backscatter",
            "m_Radiometric Resolution (Noise Value)":
                "#2#radiometricResolutionNoiseValue",
            "m_ASCAT KP Estimate Quality": "#2#ascatKpEstimateQuality",
            "m_ASCAT Sigma-0 Usability": "#2#ascatSigma0Usability",
            "m_ASCAT Land Fraction": "#2#landFraction",
            "a_Beam Identifier": "#3#beamIdentifier",
            "a_Radar Incidence Angle": "#3#radarIncidenceAngle",
            "a_Antenna Beam Azimuth": "#3#antennaBeamAzimuth",
            "a_Backscatter": "#3#backscatter",
            "a_Radiometric Resolution (Noise Value)":
                "#3#radiometricResolutionNoiseValue",
            "a_ASCAT KP Estimate Quality": "#3#ascatKpEstimateQuality",
            "a_ASCAT Sigma-0 Usability": "#3#ascatSigma0Usability",
            "a_ASCAT Land Fraction": "#3#landFraction",
            "Surface Soil Moisture (Ms)": "#1#surfaceSoilMoisture",
            "Estimated Error In Surface Soil Moisture":
                "#1#estimatedErrorInSurfaceSoilMoisture",
            "Backscatter": "#4#backscatter",
            "Estimated Error In Sigma0 At 40 Deg Incidence Angle":
                "#1#estimatedErrorInSigma0At40DegreesIncidenceAngle",
            "Slope At 40 Deg Incidence Angle":
                "#1#slopeAt40DegreesIncidenceAngle",
            "Estimated Error In Slope At 40 Deg Incidence Angle":
                "#1#estimatedErrorInSlopeAt40DegreesIncidenceAngle",
            "Soil Moisture Sensitivity": "#1#soilMoistureSensitivity",
            "Dry Backscatter": "#5#backscatter",
            "Wet Backscatter": "#6#backscatter",
            "Mean Surface Soil Moisture": "#1#meanSurfaceSoilMoisture",
            # "Rain Fall Detection": not read
            "Soil Moisture Correction Flag": "#1#soilMoistureCorrectionFlag",
            "Soil Moisture Processing Flag": "#1#soilMoistureProcessingFlag",
            "Soil Moisture Quality": "#1#soilMoistureQuality",
            "Snow Cover": "#1#snowCover",
            "Frozen Land Surface Fraction": "#1#frozenLandSurfaceFraction",
            "Inundation And Wetland Fraction": "#1#inundationAndWetlandFraction",
            "Topographic Complexity": "#1#topographicComplexity",
        }

    def _read(self, filename, generic=False, to_xarray=False):
        """
        Read one ASCAT Level 2 BUFR file.

        Parameters
        ----------
        generic : bool, optional
            'True' reading and converting into generic format or
            'False' reading original field names(default: False).
        to_xarray : bool, optional
            'True' return data as xarray.Dataset
            'False' return data as numpy.ndarray(default: False).

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.
        """
        data = read_bufr_data(filename, self.msg_key_lookup)
        data = data.to_records(index=False)
        data = {name:data[name] for name in data.dtype.names}

        data["Mean Surface Soil Moisture"] *= 100.

        metadata = {}
        metadata['platform_id'] = data['Satellite Identifier'][0].astype(int)
        metadata['orbit_start'] = np.uint32(data['Orbit Number'][0])
        metadata['filename'] = os.path.basename(filename)

        # add/rename/remove fields according to generic format
        if generic:
            data = conv_bufrl2_generic(data, metadata)

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data.keys():
                if len(data[k].shape) == 1:
                    dim = ['obs']
                elif len(data[k].shape) == 2:
                    dim = ['obs', 'beam']

                data[k] = (dim, data[k])

            coords = {}
            coords_fields = ['lon', 'lat', 'time']
            for cf in coords_fields:
                coords[cf] = data.pop(cf)

            data = xr.Dataset(data, coords=coords, attrs=metadata)
            if generic:
                data = mask_dtype_nans(data)
        else:
            # collect dtype info
            dtype = []
            # fill_value = []

            for var_name in data.keys():

                if len(data[var_name].shape) == 1:
                    dtype.append((var_name, data[var_name].dtype.str))
                    # fill_value.append(data[var_name].fill_value)

                elif len(data[var_name].shape) > 1:
                    dtype.append((var_name, data[var_name].dtype.str,
                                  data[var_name].shape[1:]))
                    # fill_value.append(data[var_name].shape[1] *
                    #                   [data[var_name].fill_value])

            ds = np.ma.empty(data['time'].size, dtype=np.dtype(dtype))
            # fill_value_arr = np.array((*fill_value, ), dtype=np.dtype(dtype))

            for k, v in data.items():
                ds[k] = v

            # ds.fill_value = fill_value_arr
            data = ds

        return data, metadata

    def _merge(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : numpy.ndarray or xarray.Dataset
            Data.
        """
        if isinstance(data[0], tuple):
            data, metadata = zip(*data)
            if isinstance(data[0], xr.Dataset):
                data = xr.concat(data,
                                 dim="obs",
                                 combine_attrs="drop_conflicts")
            else:
                data = np.hstack(data)
            data = (data, metadata)
        else:
            data = np.hstack(data)
        return data

class AscatL2BufrFileGeneric(AscatL2BufrFile):
    """
    The same as AscatL1bBufrFile but with generic=True by default.
    """
    def _read(self, filename, generic=True, to_xarray=False, **kwargs):
        return super()._read(filename, generic=generic, to_xarray=to_xarray, **kwargs)

def conv_bufrl2_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Spacecraft_id vs sat_id encoding

    BUFR encoding - Spacecraft_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-1 (Metop-B)
    - 4 Metop-2 (Metop-A)
    - 5 Metop-3 (Metop-C)

    Internal encoding - sat_id
    - 1 ERS 1
    - 2 ERS 2
    - 3 Metop-2 (Metop-A)
    - 4 Metop-1 (Metop-B)
    - 5 Metop-3 (Metop-C)

    Parameters
    ----------
    data: dict of numpy.ndarray
        Original dataset.
    metadata: dict
        Metadata.

    Returns
    -------
    data: dict of numpy.ndarray
        Converted dataset.
    """
    skip_fields = ['Satellite Identifier']

    gen_fields_beam = {
        'Radar Incidence Angle': ('inc', np.float32),
        'Backscatter': ('sig', np.float32),
        'Antenna Beam Azimuth': ('azi', np.float32),
        'ASCAT Sigma-0 Usability': ('f_usable', np.uint8),
        'Beam Identifier': ('beam_num', np.uint8),
        'Radiometric Resolution (Noise Value)': ('kp_noise', np.float32),
        'ASCAT KP Estimate Quality': ('kp', np.float32),
        'ASCAT Land Fraction': ('f_land', np.float32)
    }

    gen_fields_lut = {
        'Orbit Number': ('abs_orbit_nr', np.int32),
        'Cross-Track Cell Number': ('node_num', np.uint8),
        'Direction Of Motion Of Moving Observing Platform':
        ('sat_track_azi', np.float32),
        'Surface Soil Moisture (Ms)': ('sm', np.float32),
        'Estimated Error In Surface Soil Moisture': ('sm_noise', np.float32),
        'Backscatter': ('sig40', np.float32),
        'Estimated Error In Sigma0 At 40 Deg Incidence Angle':
        ('sig40_noise', np.float32),
        'Slope At 40 Deg Incidence Angle': ('slope40', np.float32),
        'Estimated Error In Slope At 40 Deg Incidence Angle':
        ('slope40_noise', np.float32),
        'Soil Moisture Sensitivity': ('sm_sens', np.float32),
        'Dry Backscatter': ('dry_sig40', np.float32),
        'Wet Backscatter': ('wet_sig40', np.float32),
        'Mean Surface Soil Moisture': ('sm_mean', np.float32),
        # 'Rain Fall Detection': ('rf', np.float32),
        'Soil Moisture Correction Flag': ('corr_flag', np.uint8),
        'Soil Moisture Processing Flag': ('proc_flag', np.uint8),
        'Soil Moisture Quality': ('agg_flag', np.uint8),
        'Snow Cover': ('snow_prob', np.uint8),
        'Frozen Land Surface Fraction': ('frozen_prob', np.uint8),
        'Inundation And Wetland Fraction': ('wetland', np.uint8),
        'Topographic Complexity': ('topo', np.uint8)
    }

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name, (new_name, new_dtype) in gen_fields_lut.items():
        mask = (data[var_name] == bufr_nan) |  (np.isnan(data[var_name]))
        data[var_name][mask] = nan_val_dict[new_dtype]

        data[new_name] = np.ma.array(data.pop(var_name).astype(new_dtype), mask=mask)
        data[new_name].fill_value = nan_val_dict[new_dtype]

    for var_name, (new_name, new_dtype) in gen_fields_beam.items():

        f = ['{}_{}'.format(b, var_name) for b in ['f', 'm', 'a']]

        mask = np.vstack((data[f[0]] == bufr_nan, data[f[1]] == bufr_nan,
                          data[f[2]] == bufr_nan)).T

        data[new_name] = np.ma.vstack((data.pop(f[0]), data.pop(f[1]),
                                       data.pop(f[2]))).T.astype(new_dtype)

        data[new_name].mask = mask
        data[new_name][mask] = nan_val_dict[new_dtype]

        data[new_name].fill_value = nan_val_dict[new_dtype]

    if data['node_num'].max() == 82:
        data['swath_indicator'] = np.ma.array(1 * (data['node_num'] > 41),
                                              dtype=np.uint8,
                                              mask=data['node_num'] > 82)
    elif data['node_num'].max() == 42:
        data['swath_indicator'] = np.ma.array(1 * (data['node_num'] > 21),
                                              dtype=np.uint8,
                                              mask=data['node_num'] > 42)
    else:
        raise ValueError('Cross-track cell number size unknown')

    n_lines = data['lat'].shape[0] / data['node_num'].max()
    line_num = np.arange(n_lines).repeat(data['node_num'].max())
    data['line_num'] = np.ma.array(line_num,
                                   dtype=np.uint16,
                                   mask=np.zeros_like(line_num),
                                   fill_value=uint16_nan)

    sat_id = np.ma.array([0, 0, 0, 4, 3, 5], dtype=np.uint8)
    data['sat_id'] = np.ma.zeros(data['time'].size,
                                 dtype=np.uint8) + sat_id[int(
                                     metadata['platform_id'])]
    data['sat_id'].mask = np.zeros(data['time'].size)
    data['sat_id'].fill_value = uint8_nan

    # compute ascending/descending direction
    data['as_des_pass'] = np.ma.array(data['sat_track_azi'] < 270,
                                      dtype=np.uint8,
                                      mask=np.zeros(data['time'].size),
                                      fill_value=uint8_nan)

    mask = data['lat'] == bufr_nan
    data['lat'] = np.ma.array(data['lat'], mask=mask, fill_value=float32_nan)

    mask = data['lon'] == bufr_nan
    data['lon'] = np.ma.array(data['lon'], mask=mask, fill_value=float32_nan)

    data['time'] = np.ma.array(data['time'], mask=mask, fill_value=0)

    return data
