# Copyright (c) 2025, TU Wien
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

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import netCDF4
import xarray as xr
import numpy as np
import dask.array as da

from fibgrid.realization import FibGrid
from ascat.file_handling import ChronFiles

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


def trim_dates(ds, date_range):
    """
    Trim dates of dataset to a given date range. Assumes the time variable is named
    "time", and observation dimension is named "obs"

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    date_range : tuple of datetime.datetime
        Date range to trim to.

    Returns
    -------
    xarray.Dataset
        Dataset with trimmed dates.
    """
    if date_range is None:
        return ds
    start_date = np.datetime64(date_range[0])
    end_date = np.datetime64(date_range[1])
    return ds.isel(obs=(ds.time >= start_date) & (ds.time <= end_date))


def _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size):
    # Adapted from @hmaarrfk on github: https://github.com/pydata/xarray/issues/1672
    # For time deltas, we must ensure that we use the same encoding as
    # what was previously stored.
    # We likely need to do this as well for variables that had custom
    # encodings too
    data.encoding = dict()
    if hasattr(nc_variable, 'calendar'):
        data.encoding['calendar'] = nc_variable.calendar
    if hasattr(nc_variable, 'calender'):
        data.encoding['calendar'] = nc_variable.calender

    if hasattr(nc_variable, 'dtype'):
        data.encoding['dtype'] = nc_variable.dtype
    if hasattr(nc_variable, 'units'):
        data.encoding['units'] = nc_variable.units
    if hasattr(nc_variable,
               '_FillValue') and data.attrs.get('_FillValue') is None:
        data.encoding['_FillValue'] = nc_variable._FillValue

    data_encoded = xr.conventions.encode_cf_variable(data)

    left_slices = data.dims.index(expanding_dim)
    right_slices = data.ndim - left_slices - 1
    nc_slice = ((slice(None),) * left_slices +
                (slice(nc_shape, nc_shape + added_size),) + (slice(None),) *
                (right_slices))
    nc_variable[nc_slice] = data_encoded.data


def append_to_netcdf(filename, ds_to_append, unlimited_dim):
    """Appends an xarray dataset to an existing netCDF file along a given unlimited dim.

    Parameters
    ----------
    filename : str or Path
        Filename of netCDF file to append to.
    ds_to_append : xarray.Dataset
        Dataset to append.
    unlimited_dim : str or list of str
        Name of the unlimited dimension to append along.

    Raises
    ------
    ValueError
        If more than one unlimited dim is given.
    """
    # By @hmaarrfk on github: https://github.com/pydata/xarray/issues/1672
    with netCDF4.Dataset(filename, mode='a') as nc:
        nc.set_auto_maskandscale(False)
        nc_coord = nc.dimensions[unlimited_dim]
        nc_shape = len(nc_coord)

        added_size = ds_to_append.sizes[unlimited_dim]
        variables, _ = xr.conventions.encode_dataset_coordinates(ds_to_append)

        for name, data in variables.items():
            if unlimited_dim not in data.dims:
                # Nothing to do, data assumed to the identical
                continue

            nc_variable = nc[name]
            _expand_variable(nc_variable, data, unlimited_dim, nc_shape,
                             added_size)


def var_order(ds):
    """
    Returns a reasonable variable order for a ragged array dataset,
    based on that used in existing datasets.

    Puts the count/index variable first depending on the ragged array type,
    then lon, lat, alt, location_id, location_description, and time,
    followed by the rest of the variables in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.

    Returns
    -------
    order : list of str
        List of dataset variable names in the determined order.
    """
    if "row_size" in ds.data_vars:
        first_var = "row_size"
    elif "locationIndex" in ds.data_vars:
        first_var = "locationIndex"
    else:
        raise ValueError("No row_size or locationIndex in ds."
                         " Cannot determine if indexed or ragged")
    order = [
        first_var,
        "lon",
        "lat",
        "alt",
        "location_id",
        "location_description",
        "time",
    ]
    order.extend([v for v in ds.data_vars if v not in order])

    return order


def set_attributes(ds, variable_attributes=None, global_attributes=None):
    """
    Parameters
    ----------
    ds : xarray.Dataset, Path
        Dataset.
    variable_attributes : dict, optional
        User-defined variable attributes to set. Should be a dictionary with format
        `{"varname": {"attr1": "value1", "attr2": "value2"}, "varname2": {"attr1": "value1"}}`
    global_attributes : dict, optional
        User-defined global attributes to set. Should be a dictionary with format
        `{"attr1": "value1", "attr2": "value2"}`

    Returns
    -------
    ds : xarray.Dataset
        Dataset with variable_attributes.
    """
    variable_attributes = variable_attributes or {}

    if "row_size" in ds.data_vars:
        first_var = "row_size"
    elif "locationIndex" in ds.data_vars:
        first_var = "locationIndex"

    first_var_attrs = {
        "row_size": {
            "long_name": "number of observations at this location",
            "sample_dimension": "obs",
        },
        "locationIndex": {
            "long_name": "which location this observation is for",
            "sample_dimension": "locations",
        },
    }

    default_variable_attributes = {
        first_var: first_var_attrs[first_var],
        "lon": {
            "standard_name": "longitude",
            "long_name": "location longitude",
            "units": "degrees_east",
            "valid_range": np.array([-180, 180], dtype=float),
        },
        "lat": {
            "standard_name": "latitude",
            "long_name": "location latitude",
            "units": "degrees_north",
            "valid_range": np.array([-90, 90], dtype=float),
        },
        "alt": {
            "standard_name": "height",
            "long_name": "vertical distance above the surface",
            "units": "m",
            "positive": "up",
            "axis": "Z",
        },
        "time": {
            "standard_name": "time",
            "long_name": "time of measurement",
        },
        "location_id": {},
        "location_description": {},
    }

    variable_attributes = {
        **default_variable_attributes,
        **variable_attributes
    }

    for var, attrs in variable_attributes.items():
        ds[var] = ds[var].assign_attrs(attrs)
        if var in [
                "row_size", "locationIndex", "location_id",
                "location_description"
        ]:
            ds[var].encoding["coordinates"] = None

    global_attributes = global_attributes or {}

    date_created = datetime.now().isoformat(" ", timespec="seconds")
    default_global_attributes = {
        "date_created": date_created,
        "featureType": "timeSeries",
    }
    global_attributes = {**default_global_attributes, **global_attributes}
    for key, item in global_attributes.items():
        ds.attrs[key] = item

    return ds


def create_variable_encodings(ds,
                              custom_variable_encodings=None,
                              custom_dtypes=None):
    """
    Create an encoding dictionary for a dataset, optionally
    overriding the default encoding or adding additional
    encoding parameters.
    New parameters cannot be added to default encoding for
    a variable, only overridden.

    E.g. if you want to add a "units" encoding to "lon",
    you should also pass "dtype", "zlib", "complevel",
    and "_FillValue" if you don't want to lose those.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    custom_variable_encodings : dict, optional
        Custom encodings.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with encodings.
    """
    if custom_variable_encodings is None:
        custom_variable_encodings = {}

    if "row_size" in ds.data_vars:
        # contiguous RA case
        first_var = {"row_size": {"dtype": "int64"}}
        coord_vars = {
            "lon": {"dtype": "float32",},
            "lat": {"dtype": "float32",},
            "alt": {"dtype": "float32",},
        }
    elif "locationIndex" in ds.data_vars:
        # indexed RA case
        first_var = {"locationIndex": {"dtype": "int64"}}
        coord_vars = {
            "lon": {"dtype": "float32"},
            "lat": {"dtype": "float32"},
            "alt": {"dtype": "float32"},
        }
    else:
        # swath file case
        first_var = {}
        coord_vars = {
            "lon": {"dtype": "float32"},
            "lat": {"dtype": "float32"},
            # "longitude": {"dtype": "float32"},
            # "latitude": {"dtype": "float32"},
        }

    # default encodings for coordinates and row_size
    default_encoding = {
        **first_var,
        **coord_vars,
        "location_id": {
            "dtype": "int64",
        },
        "time": {
            "dtype": "float64",
            "units": "days since 1900-01-01 00:00:00",
        },
    }

    for _, var_encoding in default_encoding.items():
        if var_encoding["dtype"] != "int64":
            var_encoding["_FillValue"] = None
        var_encoding["zlib"] = True
        var_encoding["complevel"] = 4

    default_encoding.update({
        var: {
            "dtype": dtype,
            "zlib": bool(np.issubdtype(dtype, np.number)),
            "complevel": 4,
            "_FillValue": dtype_to_nan[dtype],
        } for var, dtype in ds.dtypes.items() if var not in default_encoding
    })

    if custom_dtypes is not None:
        custom_variable_encodings = {
            var: {
                "dtype": custom_dtypes[var],
                "zlib": bool(np.issubdtype(custom_dtypes[var], np.number)),
                "complevel": 4,
                "_FillValue": dtype_to_nan[custom_dtypes[var]],
            } for var in custom_dtypes.names if var in ds.data_vars
        }

    encoding = {**default_encoding, **custom_variable_encodings}

    return encoding


class RaggedXArrayCellIOBase(ABC):
    """Base class for ascat xarray IO classes

    Attributes
    ----------
    source : str, Path, list
        Input filename(s).
    engine : str
        Engine to use for reading/writing files.
    """

    # @classmethod
    # def __init_subclass__(cls):
    #     required_class_attrs = [
    #         "grid_info",
    #         "grid",
    #         "grid_cell_size",
    #         "fn_format",
    #     ]
    #     for var in required_class_attrs:
    #         if not hasattr(cls, var):
    #             raise NotImplementedError(
    #                 f"{cls.__name__} must define required class attribute {var}"
    #             )

    def __init__(self, source, engine, obs_dim="time", **kwargs):
        self.source = source
        self.engine = engine
        self.expected_obs_dim = obs_dim
        if isinstance(source, list):
            # intended for opening multiple files from the /same cell/.
            # Won't work if two files have different locations dimension.
            self._ds = xr.open_mfdataset(
                source,
                engine=engine,
                preprocess=self._preprocess,
                decode_cf=False,
                mask_and_scale=False,
                concat_dim=obs_dim,
                combine="nested",
                data_vars="minimal",
                **kwargs)
        elif isinstance(source, (str, Path)):
            self._ds = self._ensure_obs_dim(
                xr.open_dataset(
                    source,
                    engine=engine,
                    decode_cf=False,
                    mask_and_scale=False,
                    **kwargs))
        elif isinstance(source, xr.Dataset):
            self._ds = self._ensure_obs_dim(source)

        self._kwargs = kwargs

        # if obs_dim in self._ds.dims:
        #     if obs_dim != "obs":
        #         self._ds = self._ds.rename_dims({obs_dim: "obs"})
        # else:
        #     raise ValueError(f"obs_dim '{obs_dim}' not found in dataset")

        # chunks = kwargs.pop("chunks", None)
        # if chunks is not None:
        #     self._ds = self._ds.chunk(chunks)

        if "row_size" in self._ds.data_vars:
            self._ds = self._ensure_indexed(self._ds)

    def _preprocess(self, ds):
        return self._ensure_obs_dim(ds)

    def _ensure_obs_dim(self, ds, preferred_obs_dim="obs"):
        """Rename the observations dimension of a DataSet if necessary."""
        if preferred_obs_dim in ds.dims:
            return ds
        if self.expected_obs_dim in ds.dims:
            if self.expected_obs_dim != preferred_obs_dim:
                ds = ds.rename_dims({self.expected_obs_dim: preferred_obs_dim})
            return ds
        raise ValueError(
            f"obs_dim '{self.expected_obs_dim}' not found in dataset")

    @property
    def date_range(self):
        """
        Return date range of dataset.

        Returns
        -------
        tuple
            Date range of dataset.
        """

        dates = xr.decode_cf(self._ds[["time"]], mask_and_scale=False)
        return dates.time.min().values, dates.time.max().values

    @abstractmethod
    def read(self, location_id=None, **kwargs):
        """
        Read data from file. Should be implemented by subclasses.

        Parameters
        ----------
        location_id : int, list, optional
            Location id(s) to read.
        **kwargs
            Additional keyword arguments passed to the read method.

        Returns
        -------
        xarray.Dataset
            Dataset containing the data for any specified location_id(s),
            or all location_ids in the file if none are specified.
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, filename, ra_type, **kwargs):
        """
        Write data to file. Should be implemented by subclasses.

        Parameters
        ----------
        filename : str, Path
            Filename to write data to.
        ra_type : str, optional
            Type of ragged array to write.
        **kwargs
            Additional keyword arguments passed to the write method.
        """
        raise NotImplementedError

    def _ensure_indexed(self, ds):
        """
        Convert a contiguous dataset to indexed dataset,
        if necessary. Indexed datasets pass through.

        Ragged array type is determined by the presence of
        either a "row_size" or "locationIndex" variable
        (for contiguous and indexed arrays, respectively).

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset in indexed or contiguous ragged array format.

        Returns
        -------
        xarray.Dataset
            Dataset in indexed ragged array format.
        """
        if ds is None or "locationIndex" in ds.data_vars:
            return ds

        row_size = np.where(ds["row_size"].values > 0, ds["row_size"].values,
                            0)

        locationIndex = np.repeat(np.arange(row_size.size), row_size)
        ds["locationIndex"] = ("obs", locationIndex)
        ds = ds.drop_vars(["row_size"])

        return ds[self._var_order(ds)]

    def _ensure_contiguous(self, ds):
        """
        Convert an indexed dataset to contiguous dataset,
        if necessary. Contiguous datasets pass through.

        Ragged array type is determined by the presence of
        either a "row_size" or "locationIndex" variable
        (for contiguous and indexed arrays, respectively).

        Parameters
        ----------
        ds : xarray.Dataset, Path
            Dataset in indexed or contiguous ragged array format.

        Returns
        -------
        xarray.Dataset
            Dataset in contiguous ragged array format.
        """
        if ds is None or "row_size" in ds.data_vars:
            return ds

        ds = ds.sortby(["locationIndex", "time"])
        idxs, sizes = np.unique(ds.locationIndex, return_counts=True)
        row_size = np.zeros_like(ds.location_id.values)
        row_size[idxs] = sizes
        ds["row_size"] = ("locations", row_size)
        ds = ds.drop_vars(["locationIndex"])
        return ds[self._var_order(ds)]

    def close(self):
        """Close file."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self):
        """Context manager initialization."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()


class AscatNetCDFCellBase(RaggedXArrayCellIOBase):

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None

    def read(self, date_range=None, location_id=None, mask_and_scale=True):
        """Read data from netCDF4 file.

        Read all or a subset of data from a netCDF4 file, with subset specified by the
        `location_id` argument.

        Parameters
        ----------
        date_range : tuple of datetime.datetime, optional
            Date range to read data for. If None, all data is read.
        location_id : int or list of int.
            The location_id(s) to read data for. If None, all data is read.
            Default is None.
        mask_and_scale : bool, optional
            If True, mask and scale the data according to its `scale_factor` and
            `_FillValue`/`missing_value` before returning. Default: True.
        """
        if location_id is not None:
            if isinstance(location_id, (int, np.int64, np.int32)):
                ds = self._read_location_id(location_id)
            elif isinstance(location_id, (list, np.ndarray)):
                ds = self._read_location_ids(location_id)
            else:
                raise ValueError("location_id must be int or list of ints")
        else:
            ds = self._ds

        return trim_dates(
            xr.decode_cf(ds, mask_and_scale=mask_and_scale), date_range)

    def write(self, filename, ra_type="indexed", **kwargs):
        """
        Write data to a netCDF file.

        Parameters
        ----------
        filename : str
            Output filename.
        ra_type : str, optional
            Type of ragged array to write. Default is "contiguous".
        **kwargs : dict
            Additional keyword arguments passed to xarray.to_netcdf().
        """

        if ra_type not in ["contiguous", "indexed"]:
            raise ValueError("ra_type must be 'contiguous' or 'indexed'")
        out_ds = self._ds
        if ra_type == "contiguous":
            out_ds = self._ensure_contiguous(out_ds)

        out_ds = out_ds[self._var_order(out_ds)]

        custom_variable_attrs = self._kwargs.get(
            "attributes", None) or self.custom_variable_attrs
        custom_global_attrs = self._kwargs.get(
            "global_attributes", None) or self.custom_global_attrs
        out_ds = self._set_attributes(out_ds, custom_variable_attrs,
                                      custom_global_attrs)

        custom_variable_encodings = self._kwargs.get(
            "encoding", None) or self.custom_variable_encodings
        encoding = self._create_variable_encodings(out_ds,
                                                   custom_variable_encodings)
        out_ds.encoding["unlimited_dims"] = ["obs"]

        for var, var_encoding in encoding.items():
            if "_FillValue" in var_encoding and "_FillValue" in out_ds[
                    var].attrs:
                del out_ds[var].attrs["_FillValue"]

        out_ds.to_netcdf(filename, encoding=encoding, **kwargs)

    def _read_location_id(self, location_id):
        """
        Read data for a single location_id
        """
        if location_id not in self._ds.location_id.values:
            return None
        ds = self._ds
        idx = np.where(ds.location_id.values == location_id)[0][0]
        locationIndex = np.where(ds.locationIndex.values == idx)[0]
        ds = ds.isel(obs=locationIndex, locations=idx)

        return ds

    def _read_location_ids(self, location_ids):
        """
        Read data for a list of location_ids.
        If there are location_ids not in the dataset,
        they are ignored.
        """
        if not np.any(np.isin(location_ids, self._ds.location_id.values)):
            return None
        ds = self._ds
        idxs = np.where(np.isin(ds.location_id.values, location_ids))[0]
        locationIndex = np.where(np.isin(ds.locationIndex.values, idxs))[0]
        ds = ds.isel(obs=locationIndex, locations=idxs)

        return ds

    @staticmethod
    def _var_order(ds):
        """
        A wrapper for var_order, which can be overridden in a child class
        if different logic for this function is needed.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        list of str
            List of dataset variable names in the determined order (result of var_order)
        """

        return var_order(ds)

    @staticmethod
    def _set_attributes(ds, variable_attributes=None, global_attributes=None):
        """
        A wrapper for xarray_io.set_attributes, which can be overriden in a child class
        if different logic for this function is needed.
        Parameters
        ----------
        ds : xarray.Dataset, Path
            Dataset.
       variable_attributes : dict, optional
           variable_attributes.

        Returns
        -------
        ds : xarray.Dataset
            Dataset withvariable_attributes.
        """
        return set_attributes(ds, variable_attributes, global_attributes)

    @staticmethod
    def _create_variable_encodings(ds, custom_variable_encodings=None, custom_dtypes=None):
        """
        A wrapper for xarray_io.create_variable_encodings. This can be overridden in a child class
        if different logic for this function is needed.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        custom_variable_encodings : dict, optional
            Custom encodings.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with encodings.
        """
        return create_variable_encodings(ds, custom_variable_encodings, custom_dtypes)


class SwathIOBase(ABC):
    """
    Base class for reading swath data.
    Writes ragged array cell data in indexed or contiguous format.
    """

    beams_vars = []

    @classmethod
    def __init_subclass__(cls):
        required_class_attrs = [
            "fn_pattern",
            "sf_pattern",
            "date_format",
            "grid",
            "grid_cell_size",
            "cell_fn_format",
            "ts_dtype",
        ]
        for var in required_class_attrs:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f"{cls.__name__} must define required class attribute {var}"
                )

    @classmethod
    def chron_files(cls, path):
        """
        Return a ChronFiles object for this class type based on a path.
        """
        return ChronFiles(
            path,
            cls,
            cls.fn_pattern,
            cls.sf_pattern,
            None,
            True,
            cls.fn_read_fmt,
            cls.sf_read_fmt,
        )

    @staticmethod
    @abstractmethod
    def fn_read_fmt():
        """
        TODO: figure out a sane way to describe what this does.
        Also decide if this /needs/ to be enforced. If the user
        doesn't want to use all the filesearch functionality (or
        if they want to use their own filesearch logic), then
        they should still be able to use this class. They could
        of course override this and just return None, but that
        seems like a hack.
        """
        pass

    @staticmethod
    @abstractmethod
    def sf_read_fmt():
        """
        TODO: same as above
        """
        pass

    def __init__(self, source, engine, **kwargs):
        self.source = source
        self.engine = engine
        self._cell_vals = None

        # if filename is an iterable, use open_mfdataset
        # else use open_dataset
        chunks = kwargs.pop("chunks", None)
        if isinstance(source, list):
            self._ds = xr.open_mfdataset(
                source,
                engine=engine,
                preprocess=self._preprocess,
                decode_cf=False,
                concat_dim="obs",
                combine="nested",
                chunks=(chunks or None),
                combine_attrs=self.combine_attributes,
                mask_and_scale=False,
                **kwargs)
            if chunks is not None:
                self._isin = da.isin
            else:
                self._isin = np.in1d

            chunks = None

        elif isinstance(source, (str, Path)):
            self._ds = xr.open_dataset(
                source,
                engine=engine,
                **kwargs,
                decode_cf=False,
                mask_and_scale=False)
            self._ds = self._preprocess(self._ds)

        elif isinstance(source, xr.Dataset):
            self._ds = source
            self._ds = self._preprocess(self._ds)

        self._kwargs = kwargs

        if chunks is not None:
            self._ds = self._ds.chunk(chunks)

        # the swath files have a typo in the calendar attribute
        # this fixes that and lets xarray pick it up when decoding the time variable
        # I'm not sure if it really makes a difference but this is more correct.
        if "calender" in self._ds.time.attrs:
            self._ds.time.attrs["calendar"] = self._ds.time.attrs.pop(
                "calender")

    def read(self,
             cell=None,
             location_id=None,
             mask_and_scale=True,
             lookup_vector=None):
        """
        Returns data for a cell or location_id if specified, or for the entire
        swath file if not specified.

        Parameters
        ----------
        cell : int, optional
            Cell to read data for.
        location_id : int, optional
            Location id to read data for.
        mask_and_scale : bool, optional
            Whether to mask and scale the data. Default is True.
        """
        if location_id is not None:
            out_data = self._read_location_ids(
                location_id, lookup_vector=lookup_vector)
        elif cell is not None:
            out_data = self._read_cell(cell)
        else:
            out_data = self._ds
        return xr.decode_cf(out_data, mask_and_scale=mask_and_scale)

    def write(self, filename, mode="w", **kwargs):
        out_ds = self._ds
        out_ds = out_ds[self._var_order(out_ds)]
        out_ds = self._set_attributes(out_ds)
        # out_encoding = self._create_variable_encodings(out_ds, custom_dtypes = self.ts_dtype)

        if mode == "a":
            if os.path.exists(filename):
                append_to_netcdf(filename, out_ds, unlimited_dim="obs")
                return

        out_encoding = self._create_variable_encodings(
            out_ds, custom_dtypes=self.ts_dtype)
        out_ds.to_netcdf(
            filename, unlimited_dims=["obs"], encoding=out_encoding, **kwargs)

    def _read_location_ids(self, location_ids, lookup_vector=None):
        """
        Read data for a list of location_ids.

        Parameters
        ----------
        location_ids : list or 1D array of int
            Location ids to read data for.
        lookup_vector : 1D array of int, optional
            Lookup vector for location_ids. Should make the search much faster by eliminating
            the need for np.isin. Default is None.
        """
        if lookup_vector is not None:
            return self._ds.sel(obs=lookup_vector[self._ds.location_id.values])

        idxs = self._isin(self._ds.location_id, location_ids)
        idxs = np.array([
            1 if id in location_ids else 0
            for id in self._ds.location_id.values
        ])
        if not np.any(idxs):
            return None
        ds = self._ds.isel(obs=idxs)
        return ds

    def _read_cell(self, cell):
        """
        Read data for a single cell
        """

        if self._cell_vals is None:
            self._cell_vals = self.grid.gpi2cell(self._ds.location_id.values)

        if isinstance(cell, (list, np.ndarray)):
            idxs = np.in1d(self._cell_vals, cell)
        elif isinstance(cell, (int, np.int64, np.int32)):
            idxs = np.where(self._cell_vals == cell)[0]
        else:
            raise ValueError("cell must be int or list of ints")
        if not np.any(idxs):
            return None

        ds = self._ds.isel(obs=idxs)
        return ds

    @staticmethod
    def _var_order(ds):
        """
        A wrapper for var_order, which can be overridden in a child class
        if different logic for this function is needed.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        list of str
            List of dataset variable names in the determined order (result of var_order)
        """

        return var_order(ds)

    @staticmethod
    def _set_attributes(ds, variable_attributes=None, global_attributes=None):
        """
        A wrapper for xarray_io.set_attributes, which can be overriden in a child class
        if different logic for this function is needed.
        Parameters
        ----------
        ds : xarray.Dataset, Path
            Dataset.
        variable_attributes : dict, optional
           variable_attributes.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with variable_attributes and global_attributes.
        """
        return set_attributes(ds, variable_attributes, global_attributes)

    @staticmethod
    def _create_variable_encodings(ds,
                                   custom_variable_encodings=None,
                                   custom_dtypes=None):
        """
        A wrapper for xarray_io.create_variable_encodings. This can be overridden in a child class
        if different logic for this function is needed.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        custom_variable_encodings : dict, optional
            Custom encodings.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with encodings.
        """
        return create_variable_encodings(ds, custom_variable_encodings, custom_dtypes)

    @staticmethod
    def _preprocess(ds):
        """
        To use custom logic for combining attributes in xarray, you need to write a function
        that takes a list of attribute dictionaries as an argument.
        """
        ds.attrs["global_attributes_flag"] = 1
        if "spacecraft" in ds.attrs:
            # Assumption: the spacecraft attribute is something like "metop-a"
            sat_id = {"a": 3, "b": 4, "c": 5}
            sat = ds.attrs["spacecraft"][-1].lower()
            ds["sat_id"] = ("obs",
                            np.repeat(sat_id[sat], ds["location_id"].size))
            del ds.attrs["spacecraft"]
        return ds

    @staticmethod
    def combine_attributes(attrs_list, context):
        """
        Decides which attributes to keep when merging swath files.

        Parameters
        ----------
        attrs_list : list of dict
            List of attributes dictionaries.
        context : None
            This currently is None, but will eventually be passed information about
            the context in which this was called.
            (see https://github.com/pydata/xarray/issues/6679#issuecomment-1150946521)

        Returns
        -------
        """
        # we don't need to pass on anything from global attributes
        if "global_attributes_flag" in attrs_list[0].keys():
            return None

        else:
            variable_attrs = attrs_list
            # this code taken straight from xarray/core/merge.py
            # Replicates the functionality of "drop_conflicts"
            # but just for variable attributes
            result = {}
            dropped_keys = set()
            for attrs in variable_attrs:
                result.update({
                    key: value
                    for key, value in attrs.items()
                    if key not in result and key not in dropped_keys
                })
                result = {
                    key: value
                    for key, value in result.items()
                    if key not in attrs or
                    xr.core.utils.equivalent(attrs[key], value)
                }
                dropped_keys |= {key for key in attrs if key not in result}
            return result

    def contains_location_ids(self, location_ids=None, lookup_vector=None):
        """
        Check if the dataset contains any of the given location_ids.

        Parameters
        ----------
        location_ids : list of int
            Location ids to check.

        Returns
        -------
        bool
            True if the dataset contains any of the given location_ids, False otherwise.
        """
        if lookup_vector is not None:
            return lookup_vector[self._ds.location_id.values].any()

        if location_ids is not None:
            return np.any(
                np.in1d(
                    np.unique(self._ds.location_id.values),
                    location_ids,
                    assume_unique=True))
        else:
            raise ValueError(
                "Must provide either location_ids or lookup_vector")

    def __enter__(self):
        """
        Context manager initialization.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.
        """
        self.close()

    def close(self):
        """
        Close the dataset.
        """
        self._ds.close()


class CellGridCache:
    """
    Cache for CellGrid objects.
    """

    def __init__(self):
        self.grids = {}

    def fetch_or_store(self, key, cell_grid_type=None, *args):
        """
        Fetch a CellGrid object from the cache given a key,
        or store a new one.

        Parameters
        ----------
        """
        if key not in self.grids:
            if cell_grid_type is None:
                raise ValueError(
                    "Key not in cache, please specify cell_grid_type and arguments"
                    " to create a new CellGrid object and add it to the cache under"
                    " the given key.")
            self.grids[key] = dict()
            self.grids[key]["grid"] = cell_grid_type(*args)
            self.grids[key]["possible_cells"] = self.grids[key][
                "grid"].get_cells()
            self.grids[key]["max_cell"] = self.grids[key][
                "possible_cells"].max()
            self.grids[key]["min_cell"] = self.grids[key][
                "possible_cells"].min()

        return self.grids[key]


grid_cache = CellGridCache()


# Define dataset-specific classes.
class AscatH129Cell(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, obs_dim="obs", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None


class AscatH129v1Cell(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, obs_dim="obs", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None


class AscatH121v1Cell(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, obs_dim="obs", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None


class AscatH122Cell(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, obs_dim="obs", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None


class AscatSIG0Cell6250m(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, obs_dim="obs", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None


class AscatSIG0Cell12500m(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, obs_dim="obs", **kwargs)
        self.custom_variable_attrs = None
        self.custom_global_attrs = None
        self.custom_variable_encodings = None


class AscatH129Swath(SwathIOBase):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H129_C_LIIB_{date}_{placeholder}_{placeholder1}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid_sampling_km = 6.25
    grid = grid_cache.fetch_or_store("Fib6.25", FibGrid,
                                     grid_sampling_km)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = ["backscatter", "incidence_angle", "azimuth_angle", "kp"]
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("backscatter_for", np.float32),
        ("backscatter_mid", np.float32),
        ("backscatter_aft", np.float32),
        ("incidence_angle_for", np.float32),
        ("incidence_angle_mid", np.float32),
        ("incidence_angle_aft", np.float32),
        ("azimuth_angle_for", np.float32),
        ("azimuth_angle_mid", np.float32),
        ("azimuth_angle_aft", np.float32),
        ("kp_for", np.float32),
        ("kp_mid", np.float32),
        ("kp_aft", np.float32),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("backscatter40", np.float32),
        ("slope40", np.float32),
        ("curvature40", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("surface_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


class AscatH129v1Swath(SwathIOBase):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H129_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid_sampling_km = 6.25
    grid = grid_cache.fetch_or_store("Fib6.25", FibGrid,
                                     grid_sampling_km)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = []
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("backscatter40", np.float32),
        ("slope40", np.float32),
        ("curvature40", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("backscatter_flag", np.uint8),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("surface_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
        ("subsurface_scattering_probability", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


class AscatH121v1Swath(SwathIOBase):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-12.5km-H121_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid_sampling_km = 12.5
    grid = grid_cache.fetch_or_store("Fib12.5", FibGrid,
                                     grid_sampling_km)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = []
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("backscatter40", np.float32),
        ("slope40", np.float32),
        ("curvature40", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("backscatter_flag", np.uint8),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("surface_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
        ("subsurface_scattering_probability", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


class AscatH122Swath(SwathIOBase):
    fn_pattern = "ascat_ssm_nrt_6.25km_{placeholder}Z_{date}Z_metop-{sat}_h122.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid_sampling_km = 6.25
    grid = grid_cache.fetch_or_store("Fib6.25", FibGrid,
                                     grid_sampling_km)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = []
    ts_dtype = np.dtype([
        ("sat_id", np.int64),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("sigma40", np.float32),
        ("sigma40_noise", np.float32),
        ("slope40", np.float32),
        ("slope40_noise", np.float32),
        ("curvature40", np.float32),
        ("curvature40_noise", np.float32),
        ("dry40", np.float32),
        ("dry40_noise", np.float32),
        ("wet40", np.float32),
        ("wet40_noise", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("surface_soil_moisture_climatology", np.float32),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


class AscatSIG0Swath6250m(SwathIOBase):
    """
    Class for reading ASCAT sigma0 swath data and writing it to cells.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP{sat}-6.25_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid_sampling_km = 6.25
    grid = grid_cache.fetch_or_store("Fib6.25", FibGrid,
                                     grid_sampling_km)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = [
        "backscatter",
        "backscatter_std",
        "incidence_angle",
        "azimuth_angle",
        "kp",
        "n_echos",
        "all_backscatter",
        "all_backscatter_std",
        "all_incidence_angle",
        "all_azimuth_angle",
        "all_kp",
        "all_n_echos",
    ]
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("backscatter_for", np.float32),
        ("backscatter_mid", np.float32),
        ("backscatter_aft", np.float32),
        ("backscatter_std_for", np.float32),
        ("backscatter_std_mid", np.float32),
        ("backscatter_std_aft", np.float32),
        ("incidence_angle_for", np.float32),
        ("incidence_angle_mid", np.float32),
        ("incidence_angle_aft", np.float32),
        ("azimuth_angle_for", np.float32),
        ("azimuth_angle_mid", np.float32),
        ("azimuth_angle_aft", np.float32),
        ("kp_for", np.float32),
        ("kp_mid", np.float32),
        ("kp_aft", np.float32),
        ("n_echos_for", np.int8),
        ("n_echos_mid", np.int8),
        ("n_echos_aft", np.int8),
        ("all_backscatter_for", np.float32),
        ("all_backscatter_mid", np.float32),
        ("all_backscatter_aft", np.float32),
        ("all_backscatter_std_for", np.float32),
        ("all_backscatter_std_mid", np.float32),
        ("all_backscatter_std_aft", np.float32),
        ("all_incidence_angle_for", np.float32),
        ("all_incidence_angle_mid", np.float32),
        ("all_incidence_angle_aft", np.float32),
        ("all_azimuth_angle_for", np.float32),
        ("all_azimuth_angle_mid", np.float32),
        ("all_azimuth_angle_aft", np.float32),
        ("all_kp_for", np.float32),
        ("all_kp_mid", np.float32),
        ("all_kp_aft", np.float32),
        ("all_n_echos_for", np.int8),
        ("all_n_echos_mid", np.int8),
        ("all_n_echos_aft", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        """
        Format a timestamp to search as YYYYMMDD*, for use in a regex
        that will match all files covering a single given date.

        Parameters
        ----------
        timestamp: datetime.datetime
            Timestamp to format

        Returns
        -------
        dict
            Dictionary of formatted strings
        """
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


class AscatSIG0Swath12500m(SwathIOBase):
    """
    Class for reading and writing ASCAT sigma0 swath data.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP{sat}-12.5_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid_sampling_km = 12.5
    grid = grid_cache.fetch_or_store("Fib12.5", FibGrid,
                                     grid_sampling_km)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = [
        "backscatter",
        "backscatter_std",
        "incidence_angle",
        "azimuth_angle",
        "kp",
        "n_echos",
        "all_backscatter",
        "all_backscatter_std",
        "all_incidence_angle",
        "all_azimuth_angle",
        "all_kp",
        "all_n_echos",
    ]
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("backscatter_for", np.float32),
        ("backscatter_mid", np.float32),
        ("backscatter_aft", np.float32),
        ("backscatter_std_for", np.float32),
        ("backscatter_std_mid", np.float32),
        ("backscatter_std_aft", np.float32),
        ("incidence_angle_for", np.float32),
        ("incidence_angle_mid", np.float32),
        ("incidence_angle_aft", np.float32),
        ("azimuth_angle_for", np.float32),
        ("azimuth_angle_mid", np.float32),
        ("azimuth_angle_aft", np.float32),
        ("kp_for", np.float32),
        ("kp_mid", np.float32),
        ("kp_aft", np.float32),
        ("n_echos_for", np.int8),
        ("n_echos_mid", np.int8),
        ("n_echos_aft", np.int8),
        ("all_backscatter_for", np.float32),
        ("all_backscatter_mid", np.float32),
        ("all_backscatter_aft", np.float32),
        ("all_backscatter_std_for", np.float32),
        ("all_backscatter_std_mid", np.float32),
        ("all_backscatter_std_aft", np.float32),
        ("all_incidence_angle_for", np.float32),
        ("all_incidence_angle_mid", np.float32),
        ("all_incidence_angle_aft", np.float32),
        ("all_azimuth_angle_for", np.float32),
        ("all_azimuth_angle_mid", np.float32),
        ("all_azimuth_angle_aft", np.float32),
        ("all_kp_for", np.float32),
        ("all_kp_mid", np.float32),
        ("all_kp_aft", np.float32),
        ("all_n_echos_for", np.int8),
        ("all_n_echos_mid", np.int8),
        ("all_n_echos_aft", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        """
        Format a timestamp to search as YYYYMMDD*, for use in a regex
        that will match all files covering a single given date.

        Parameters
        ----------
        timestamp: datetime.datetime
            Timestamp to format

        Returns
        -------
        dict
            Dictionary of formatted strings
        """
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


cell_io_catalog = {
    "H129": AscatH129Cell,
    "H129_V1.0": AscatH129v1Cell,
    "H121_V1.0": AscatH121v1Cell,
    "H122": AscatH122Cell,
    "SIG0_6.25": AscatSIG0Cell6250m,
    "SIG0_12.5": AscatSIG0Cell12500m,
}

swath_io_catalog = {
    "H129": AscatH129Swath,
    "H129_V1.0": AscatH129v1Swath,
    "H121_V1.0": AscatH121v1Swath,
    "H122": AscatH122Swath,
    "SIG0_6.25": AscatSIG0Swath6250m,
    "SIG0_12.5": AscatSIG0Swath12500m,
}

swath_fname_regex_lookup = {
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-6.25-H129_C_LIIB_.*_.*_.*____.nc":
        "H129",
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-6.25km-H129_C_LIIB_.*_.*_.*____.nc":
        "H129_V1.0",
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-12.5km-H121_C_LIIB_.*_.*_.*____.nc":
        "H121_V1.0",
    "ascat_ssm_nrt_6.25km_.*Z_.*Z_metop-[ABC]_h122.nc":
        "H122",
    "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP[ABC]-6.25_C_LIIB_.*_.*_.*____.nc":
        "SIG0_6.25",
    "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP[ABC]-12.5_C_LIIB_.*_.*_.*____.nc":
        "SIG0_12.5",
    "ascat_h129-v1.0_6.25km_.*_.*_.*.nc": "H129",
    "ascat_h121-v1.0_12.5km_.*_.*_.*.nc": "H121",
    "ascat_h122_6.25km_.*_.*_*.nc": "H122",
    "ascat_h129_6.25km_.*_.*_.*.nc": "H122"
}


def get_swath_product_id(filename):
    """
    Get product identifier from filename.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    product_id : str
        Product identifier.
    """
    product_id = None

    for pattern, swath_product_id in swath_fname_regex_lookup.items():
        if re.match(pattern, filename):
            product_id = swath_product_id

    return product_id
