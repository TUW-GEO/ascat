# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Readers for EPS-SG SCA Level 1b data.

The Scatterometer (SCA) on board the Metop-SG B satellites is the successor of
ASCAT. Its Level 1b products come as NetCDF files which, unlike the ASCAT ones,
keep all variables in groups: "status" holds the orbit and instrument state,
"quality" the summary flags and "data" the measurements.

Two product types are supported:

SZF
    Full resolution backscatter, one group per antenna beam. SCA has twelve
    beams, the fore and aft beams of both swaths in VV polarisation and the mid
    beams additionally in HH, VH and HV. The measurements are stored per beam as
    a (time, range) array, which is flattened to a single observation dimension
    here, the same layout the ASCAT SZF reader returns.

SZR
    Backscatter re-sampled onto a 12.5 km swath grid and collocated into
    quintuplets (fore-VV, mid-VV, aft-VV, mid-HH and mid-XX), stored as an
    (observation, beam) array, the same layout the ASCAT SZR reader returns.

Note that the two products store the backscatter in different units: the SZR
files declare dB, the SZF files carry no unit and hold linear values. Both are
passed through unchanged and end up in the generic field "sig", so take the
product type into account before comparing them or the ASCAT backscatter.
"""

import re

from collections import OrderedDict
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import netCDF4
import xarray as xr

from ascat.eumetsat.sca.flags import set_flags
from ascat.file_handling import ChronFiles
from ascat.read_native.base import AscatFile
from ascat.utils import netcdf_attrs
from ascat.utils import Spacecraft

#: SCA beam groups of the SZF product, keyed by an ASCAT style beam name.
szf_beams = OrderedDict([
    ("lf-vv", "left_fore_VV"),
    ("lm-vv", "left_mid_VV"),
    ("lm-hh", "left_mid_HH"),
    ("lm-vh", "left_mid_VH"),
    ("lm-hv", "left_mid_HV"),
    ("la-vv", "left_aft_VV"),
    ("rf-vv", "right_fore_VV"),
    ("rm-vv", "right_mid_VV"),
    ("rm-hh", "right_mid_HH"),
    ("rm-vh", "right_mid_VH"),
    ("rm-hv", "right_mid_HV"),
    ("ra-vv", "right_aft_VV"),
])

#: Product type as given in the file name, e.g. "...SGB1-SCA-1B-SZF_C_EUMT...".
product_type_pattern = re.compile(r"SCA-\w+-([A-Z0-9]+)_")

#: Beam order of the SZR quintuplets along the "beam" dimension.
szr_beams = ["fore-vv", "mid-vv", "aft-vv", "mid-hh", "mid-xx"]

#: Spacecraft identifier used in the products.
spacecraft_lut = {
    "SGB1": "METOP-SG B1",
    "SGB2": "METOP-SG B2",
    "SGB3": "METOP-SG B3",
}

# template - "original_name": ("generic_name", generic dtype)
#
# Only fields with an ASCAT counterpart are renamed. "flag_generic" and
# "flag_surface" keep their names: the former is a SCA specific bitfield
# unrelated to the ASCAT "flagfield", the latter has no ASCAT equivalent.
szf_gen_fields_lut = {
    "backscatter": ("sig", np.float32),
    "longitude": ("lon", np.float32),
    "latitude": ("lat", np.float32),
    "incidence_angle": ("inc", np.float32),
    "azimuth_angle": ("azi", np.float32),
    "lcr": ("f_land", np.float32),
    "flag_quality": ("f_usable", np.int8),
    "flag_pass": ("as_des_pass", np.uint8),
}

szr_gen_fields_lut = {
    "backscatter": ("sig", np.float32),
    "longitude": ("lon", np.float32),
    "latitude": ("lat", np.float32),
    "incidence_angle": ("inc", np.float32),
    "azimuth_angle": ("azi", np.float32),
    "lcr": ("f_land", np.float32),
    "kp": ("kp", np.float32),
    "corrected_cross_pol": ("sig_cross_pol", np.float32),
    "faraday_rotation_angle": ("faraday_rotation", np.float32),
    "flag_quality": ("f_usable", np.int8),
    "flag_pass": ("as_des_pass", np.uint8),
    "line_index": ("line_num", np.uint32),
    "node_index": ("node_num", np.int16),
}


def parse_time(variable):
    """
    Convert a SCA time variable to datetime64.

    Parameters
    ----------
    variable : netCDF4.Variable
        Time variable with a "seconds since <epoch>" unit.

    Returns
    -------
    time : numpy.ndarray
        Time as datetime64[ms].
    """
    epoch = variable.units.split("since")[1].strip().replace(" ", "T")
    seconds = np.ma.filled(variable[:], np.nan).astype(np.float64)

    return (np.datetime64(epoch, "ms")
            + np.round(seconds * 1e3).astype("int64").astype("timedelta64[ms]"))


def read_grid(source, to_xarray=False):
    """
    Read the swath grid of a SCA Level 1b SZF file.

    Next to the measurements of the antenna beams, the SZF products carry the
    grid onto which the SZR products resample them, as the coordinates of the
    nodes of the left and the right hand swath. It is a coarser sampling of the
    same acquisition, so it has its own dimensions and is not returned together
    with the beams.

    Parameters
    ----------
    source : str or netCDF4.Dataset
        Filename, or an open SCA Level 1b SZF file.
    to_xarray : boolean, optional
        Convert data to xarray.Dataset otherwise a dictionary of
        numpy.ndarray will be returned (default: False).

    Returns
    -------
    grid : dict of numpy.ndarray or xarray.Dataset
        Node coordinates of both swaths and the time of each line of nodes.

    Raises
    ------
    KeyError
        If the file has no grid, as is the case for the SZR products.
    """
    if isinstance(source, netCDF4.Dataset):
        return _read_grid(source, to_xarray)

    with netCDF4.Dataset(source) as fid:
        return _read_grid(fid, to_xarray)


def _read_grid(fid, to_xarray=False):
    """
    Read the "data/grid" group of an open SCA Level 1b file.

    Parameters
    ----------
    fid : netCDF4.Dataset
        Open SCA Level 1b file.
    to_xarray : boolean, optional
        Convert data to xarray.Dataset (default: False).

    Returns
    -------
    grid : dict of numpy.ndarray or xarray.Dataset
        Grid.
    """
    group = fid.groups["data"].groups.get("grid")

    if group is None:
        raise KeyError(
            f"{Path(fid.product_name).name} has no grid. Only the SZF "
            "products carry one, the SZR products are already on it.")

    grid = {"time": parse_time(group.variables["time"])}

    for var_name, variable in group.variables.items():
        if var_name != "time":
            grid[var_name] = variable[:]

    if not to_xarray:
        return grid

    dims = ("along_track", "across_track")
    variables = {name: (dims, value) for name, value in grid.items()
                 if name != "time"}
    variables["time"] = (dims[:1], grid["time"].astype("datetime64[ns]"))

    return xr.Dataset(variables, attrs=netcdf_attrs(read_metadata(fid)))


def read_quality(source):
    """
    Read the summary flags of a SCA Level 1b file.

    The "quality" group holds a summary of the flags of the whole file, of the
    flags of each beam, and for SZR how many grid points received a complete
    set of measurements. It is small, so it can be read without touching the
    measurements themselves, e.g. to decide whether a file is worth reading.

    Parameters
    ----------
    source : str or netCDF4.Dataset
        Filename, or an open SCA Level 1b file.

    Returns
    -------
    quality : dict
        Summary flags, keyed by the name of the variable prefixed with
        "quality_". Empty if the file has no "quality" group.
    """
    if isinstance(source, netCDF4.Dataset):
        return _read_quality(source)

    with netCDF4.Dataset(source) as fid:
        return _read_quality(fid)


def _read_quality(fid):
    """
    Read the "quality" group of an open SCA Level 1b file.

    Parameters
    ----------
    fid : netCDF4.Dataset
        Open SCA Level 1b file.

    Returns
    -------
    quality : dict
        Summary flags.
    """
    group = fid.groups.get("quality")

    if group is None:
        return {}

    quality = {}
    for var_name, variable in group.variables.items():
        value = np.ma.filled(variable[:])
        # scalars are stored as zero-dimensional arrays
        quality[f"quality_{var_name}"] = (value[()] if value.ndim == 0
                                          else value)

    return quality


def read_metadata(fid):
    """
    Read metadata from the global attributes and the "status" group.

    Parameters
    ----------
    fid : netCDF4.Dataset
        Open SCA Level 1b file.

    Returns
    -------
    metadata : dict
        Metadata.
    """
    spacecraft = Spacecraft(spacecraft_lut[fid.spacecraft])
    status = fid.groups["status"]

    metadata = {
        "spacecraft": fid.spacecraft,
        "sat_id": spacecraft.sat_id,
        "sat_name": spacecraft.sat_name,
        "sensor": spacecraft.sensor,
        "orbit_start": int(fid.orbit_start),
        "orbit_end": int(fid.orbit_end),
        "product_type": fid.type,
        "processing_level": fid.product_level,
        "filename": Path(fid.product_name).name,
    }

    for field in ["sensing_start_time_utc", "sensing_end_time_utc"]:
        metadata[field.replace("_utc", "")] = datetime.strptime(
            getattr(fid, field), "%Y-%m-%d %H:%M:%S.%f")

    instrument = status.groups["instrument"]
    metadata["instrument_mode"] = str(instrument.variables["instrument_mode"][0])

    metadata.update(_read_quality(fid))

    return metadata


def conv_scal1b_generic(data, metadata, gen_fields_lut):
    """
    Rename and convert data types of dataset.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Original dataset.
    metadata : dict
        Metadata.
    gen_fields_lut : dict
        Lookup table mapping original to generic field name and data type.

    Returns
    -------
    data : dict of numpy.ndarray
        Converted dataset.
    """
    for var_name, (new_name, new_dtype) in gen_fields_lut.items():
        if var_name not in data:
            continue
        data[new_name] = data.pop(var_name).astype(new_dtype)

    data["sat_id"] = np.repeat(metadata["sat_id"], data["time"].size)

    return data


def to_ds(data, metadata, beam_dim=None):
    """
    Convert a dictionary of arrays to an xarray.Dataset.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Dataset.
    metadata : dict
        Metadata, stored as dataset attributes.
    beam_dim : str, optional
        Name of the second dimension of two-dimensional fields.

    Returns
    -------
    ds : xarray.Dataset
        Dataset.
    """
    variables = {}
    for var_name, var_data in data.items():
        if var_data.ndim == 1:
            dim = ["obs"]
        else:
            dim = ["obs", beam_dim]
        if var_name == "time":
            var_data = var_data.astype("datetime64[ns]")
        variables[var_name] = (dim, var_data)

    # Without the generic conversion the fields keep their original names.
    coord_fields = ["lon", "longitude", "lat", "latitude", "time"]
    coords = {name: variables.pop(name)
              for name in coord_fields if name in variables}

    return xr.Dataset(variables, coords=coords, attrs=netcdf_attrs(metadata))


def to_rec_array(data):
    """
    Convert a dictionary of arrays to a masked structured array.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Dataset.

    Returns
    -------
    rec_array : numpy.ma.MaskedArray
        Structured array.
    """
    dtype = []
    for var_name, var_data in data.items():
        if var_data.ndim == 1:
            dtype.append((var_name, var_data.dtype.str))
        else:
            dtype.append((var_name, var_data.dtype.str, var_data.shape[1:]))

    rec_array = np.ma.empty(data["time"].shape[0], dtype=np.dtype(dtype))

    for var_name, var_data in data.items():
        rec_array[var_name] = var_data

    return rec_array


def get_product_type(filename):
    """
    Determine the product type of a SCA Level 1b file.

    The product type is taken from the file name, falling back to the "type"
    attribute of the file itself. Reading it from the name keeps the file from
    being opened twice per read, which the netCDF library does not always
    survive when several handles to the same file are around.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    product_type : str
        Product type, e.g. "SZF" or "SZR".
    """
    match = product_type_pattern.search(Path(filename).name)

    if match is not None:
        return match.group(1)

    with netCDF4.Dataset(filename) as fid:
        return fid.type


class ScaL1bSzfFile(AscatFile):
    """
    Class reading EPS-SG SCA Level 1b SZF files.
    """

    def _read(self, filename, generic=True, to_xarray=False,
              flag_kwargs=None):
        """
        Read one SCA Level 1b SZF file.

        Parameters
        ----------
        filename : str
            Filename.
        generic : boolean, optional
            Convert original data field names to generic field names
            (default: True).
        to_xarray : boolean, optional
            Convert data to xarray.Dataset otherwise numpy.ndarray will be
            returned (default: False).
        flag_kwargs : dict, optional
            If given, a second summary flag "f_usable_user" is computed from
            "flag_generic" with :func:`ascat.eumetsat.sca.flags.set_flags`,
            e.g. ``{"rfi_red": False}`` to not let a noise outlier render a
            measurement unusable. The summary stored in the product is always
            kept as "f_usable" (default: None).

        Returns
        -------
        data : dict of xarray.Dataset or numpy.ndarray
            SCA data, one entry per antenna beam.
        metadata : dict
            Metadata.
        """
        ds = OrderedDict()

        with netCDF4.Dataset(filename) as fid:
            metadata = read_metadata(fid)

            for beam, group in szf_beams.items():
                beam_group = fid.groups["data"].groups[group]
                num_range = beam_group.dimensions["range"].size
                data = {"time": np.repeat(
                    parse_time(beam_group.variables["time"]), num_range)}

                for var_name, variable in beam_group.variables.items():
                    if var_name == "time":
                        continue
                    # Measurements come as (time, range), everything given once
                    # per line is repeated to match.
                    if variable.ndim == 1:
                        data[var_name] = np.repeat(variable[:], num_range)
                    else:
                        data[var_name] = variable[:].ravel()

                # The beam name tells which swath a measurement belongs to.
                data["swath_indicator"] = np.full(
                    data["time"].size, int(beam.startswith("r")), dtype=np.int8)

                if flag_kwargs is not None:
                    data["f_usable_user"] = set_flags(data["flag_generic"],
                                                      **flag_kwargs)

                if generic:
                    data = conv_scal1b_generic(data, metadata,
                                               szf_gen_fields_lut)

                if to_xarray:
                    ds[beam] = to_ds(data, metadata)
                else:
                    ds[beam] = to_rec_array(data)

        return ds, metadata

    def _merge(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : dict of xarray.Dataset or numpy.ndarray
            Data.
        """
        metadata = {}

        if isinstance(data[0], tuple):
            data, metadata = zip(*data)

        merged_data = defaultdict(list)
        for beam in szf_beams:
            for d in data:
                merged_data[beam].append(d.pop(beam))
            if isinstance(merged_data[beam][0], xr.Dataset):
                merged_data[beam] = xr.concat(merged_data[beam],
                                              dim="obs",
                                              combine_attrs="drop_conflicts")
            else:
                merged_data[beam] = np.hstack(merged_data[beam])

        return merged_data, metadata


class ScaL1bSzrFile(AscatFile):
    """
    Class reading EPS-SG SCA Level 1b SZR files.
    """

    def _read(self, filename, generic=True, to_xarray=False,
              flag_kwargs=None):
        """
        Read one SCA Level 1b SZR file.

        Parameters
        ----------
        filename : str
            Filename.
        generic : boolean, optional
            Convert original data field names to generic field names
            (default: True).
        to_xarray : boolean, optional
            Convert data to xarray.Dataset otherwise numpy.ndarray will be
            returned (default: False).
        flag_kwargs : dict, optional
            If given, a second summary flag "f_usable_user" is computed from
            "flag_generic" with :func:`ascat.eumetsat.sca.flags.set_flags`,
            e.g. ``{"rfi_red": False}`` to not let a noise outlier render a
            measurement unusable. The summary stored in the product is always
            kept as "f_usable" (default: None).

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            SCA data.
        metadata : dict
            Metadata.
        """
        with netCDF4.Dataset(filename) as fid:
            metadata = read_metadata(fid)
            group = fid.groups["data"]

            data = {"time": parse_time(group.variables["time"])}
            for var_name, variable in group.variables.items():
                if var_name != "time":
                    data[var_name] = variable[:]

            if flag_kwargs is not None:
                data["f_usable_user"] = set_flags(data["flag_generic"],
                                                  **flag_kwargs)

            if generic:
                data = conv_scal1b_generic(data, metadata, szr_gen_fields_lut)

        if to_xarray:
            return to_ds(data, metadata, beam_dim="beam"), metadata

        return to_rec_array(data), metadata

    def _merge(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            Data.
        """
        metadata = {}

        if isinstance(data[0], tuple):
            data, metadata = zip(*data)

        if isinstance(data[0], xr.Dataset):
            merged_data = xr.concat(data, dim="obs",
                                    combine_attrs="drop_conflicts")
        else:
            merged_data = np.hstack(data)

        return merged_data, metadata


class ScaL1bFile:
    """
    Class reading EPS-SG SCA Level 1b files.
    """

    def __new__(cls, filename, product_type=None):
        """
        Return an instance of the appropriate SCA Level 1b file reader.

        Parameters
        ----------
        filename : str
            Filename.
        product_type : str, optional
            Product type: "SZF" or "SZR" (default: None).
            If None the product type is read from the file.
        """
        if product_type is None:
            if isinstance(filename, (str, Path)):
                first = filename
            else:
                first = filename[0]
            product_type = get_product_type(first)

        product_type = product_type.upper()

        if product_type == "SZF":
            return ScaL1bSzfFile(filename)
        if product_type == "SZR":
            return ScaL1bSzrFile(filename)

        raise RuntimeError(f"SCA Level 1b product type unknown: {product_type}")


class ScaL1bFileList(ChronFiles):
    """
    Class reading EPS-SG SCA Level 1b files.
    """

    fn_pattern = ("W_XX-EUMETSAT-Darmstadt,SAT,{sat}-SCA-1B-{product}"
                  "_C_EUMT_{placeholder1}_G_O_{date}_{placeholder2}"
                  "_O_N___.nc")

    def __init__(self, path, sat="SGB1", product="szf", filename_template=None):
        """
        Initialize.

        Parameters
        ----------
        path : str
            Path to input data.
        sat : str, optional
            Metop-SG satellite ("SGB1", "SGB2", "SGB3"), default: "SGB1".
        product : str, optional
            Product type ("szf", "szr"), default: "szf".
        filename_template : str, optional
            Filename template.
        """
        self.sat = "SG" + Spacecraft(sat).sat_name
        self.product = product.upper()

        if filename_template is None:
            filename_template = self.fn_pattern

        super().__init__(path, ScaL1bFile, filename_template, None)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_read_fmt : dict
            Filename format.
        sf_read_fmt : dict
            Subfolder format.
        fn_write_fmt : dict
            Filename format.
        sf_write_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {
            "date": timestamp.strftime("%Y%m%d%H%M%S"),
            "sat": self.sat,
            "product": self.product,
            "placeholder1": "*",
            "placeholder2": "*",
        }
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt
