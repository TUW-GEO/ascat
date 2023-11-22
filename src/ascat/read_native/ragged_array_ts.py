# Copyright (c) 2023, TU Wien, Department of Geodesy and Geoinformation
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
import warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from functools import partial

import xarray as xr
import numpy as np

import tqdm
import dask
import dask.bag as db
from dask.distributed import Client
from fibgrid.realization import FibGrid
from ascat.read_native.xarray_io import ASCAT_NetCDF4
from ascat.file_handling import ChronFiles

int8_nan = np.iinfo(np.int8).max
int64_nan = np.iinfo(np.int64).min
NC_FILL_FLOAT = np.float32(9969209968386869046778552952102584320)


class RAFile:
    """
    Base class used for Ragged Array (RA) time series data.
    """

    def __init__(
        self,
        loc_dim_name="locations",
        obs_dim_name="time",
        loc_ids_name="location_id",
        loc_descr_name="location_description",
        time_units="days since 1900-01-01 00:00:00",
        time_var="time",
        lat_var="lat",
        lon_var="lon",
        alt_var="alt",
        cache=False,
        mask_and_scale=False,
    ):
        """
        Initialize.

        Parameters
        ----------
        loc_dim_name : str, optional
            Location dimension name (default: "locations").
        obs_dim_name : str, optional
            Observation dimension name (default: "time").
        loc_ids_name : str, optional
            Location IDs name (default: "location_id").
        loc_descr_name : str, optional
            Location description name (default: "location_description").
        time_units : str, optional
            Time units definition (default: "days since 1900-01-01 00:00:00").
        time_var : str, optional
            Time variable name (default: "time").
        lat_var : str, optional
            Latitude variable name (default: "lat").
        lon_var : str, optional
            Latitude variable name (default: "lon").
        alt_var : str, optional
            Altitude variable name (default: "alt").
        cache : boolean, optional
            Cache flag (default: False).
        mask_and_scale : boolean, optional
            Mask and scale during reading (default: False).
        """
        # dimension names
        self.dim = {"obs": obs_dim_name, "loc": loc_dim_name}

        # location names
        self.loc = {"ids": loc_ids_name, "descr": loc_descr_name}

        # time, time units and location
        self.var = {
            "time": time_var,
            "time_units": time_units,
            "lat": lat_var,
            "lon": lon_var,
            "alt": alt_var,
        }

        self.cache = cache
        self._cached = False
        self.mask_and_scale = mask_and_scale


class IRANcFile(RAFile):
    """
    Indexed ragged array file reader.
    """

    def __init__(self, filename, **kwargs):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            Filename.
        loc_dim_name : str, optional
            Location dimension name (default: "locations").
        obs_dim_name : str, optional
            Observation dimension name (default: "time").
        loc_ids_name : str, optional
            Location IDs name (default: "location_id").
        loc_descr_name : str, optional
            Location description name (default: "location_description").
        time_units : str, optional
            Time units definition (default: "days since 1900-01-01 00:00:00").
        time_var : str, optional
            Time variable name (default: "time").
        lat_var : str, optional
            Latitude variable name (default: "lat").
        lon_var : str, optional
            Latitude variable name (default: "lon").
        alt_var : str, optional
            Altitude variable name (default: "alt").
        cache : boolean, optional
            Cache flag (default: False).
        mask_and_scale : boolean, optional
            Mask and scale during reading (default: False).
        """
        super().__init__(**kwargs)
        self.filename = filename

        # read location information
        with xr.open_dataset(self.filename,
                             mask_and_scale=self.mask_and_scale) as ncfile:
            var_list = [self.var["lon"], self.var["lat"], self.loc["ids"]]

            if self.cache:
                self.dataset = ncfile.load()
                self.locations = self.dataset[var_list].to_dataframe()
            else:
                self.dataset = None
                self.locations = ncfile[var_list].to_dataframe()

    @property
    def ids(self):
        """
        Location IDs property.

        Returns
        -------
        location_id : numpy.ndarray
            Location IDs.
        """
        return self.locations.location_id

    @property
    def lons(self):
        """
        Longitude coordinates property.

        Returns
        -------
        lon : numpy.ndarray
            Longitude coordinates.
        """
        return self.locations.lon

    @property
    def lats(self):
        """
        Latitude coordinates property.

        Returns
        -------
        lat : numpy.ndarray
            Latitude coordinates.
        """
        return self.locations.lat

    def read(self, location_id, variables=None):
        """
        Read a timeseries for a given location_id.

        Parameters
        ----------
        location_id : int
            Location_id to read.
        variables : list or None
            A list of parameter-names to read. If None, all parameters are read.
            If None, all parameters will be read. The default is None.

        Returns
        -------
        df : pandas.DataFrame
            A pandas.DataFrame containing the timeseries for the location_id.
        """
        pos = self.locations.location_id == location_id

        if not pos.any():
            print(f"location_id not found: {location_id}")
            data = None
        else:
            sel = self.locations[pos]
            i = sel.index.values[0]

            if self.cache:
                j = self.dataset.locationIndex.values == i
                data = self.dataset.sel(locations=i, time=j)
            else:
                with xr.open_dataset(
                        self.filename,
                        mask_and_scale=self.mask_and_scale) as dataset:
                    j = dataset.locationIndex.values == i
                    data = dataset.sel(locations=i, time=j)

            if variables is not None:
                data = data[variables]

        return data


class CRANcFile(RAFile):
    """
    Contiguous ragged array file reader.
    """

    def __init__(self, filename, row_var="row_size", **kwargs):
        """
        Initialize reader.

        Parameters
        ----------
        filename : str
            Filename.
        row_size : str, optional
            Row size variable name (default: "row_size")
        loc_dim_name : str, optional
            Location dimension name (default: "locations").
        obs_dim_name : str, optional
            Observation dimension name (default: "time").
        loc_ids_name : str, optional
            Location IDs name (default: "location_id").
        loc_descr_name : str, optional
            Location description name (default: "location_description").
        time_units : str, optional
            Time units definition (default: "days since 1900-01-01 00:00:00").
        time_var : str, optional
            Time variable name (default: "time").
        lat_var : str, optional
            Latitude variable name (default: "lat").
        lon_var : str, optional
            Latitude variable name (default: "lon").
        alt_var : str, optional
            Altitude variable name (default: "alt").
        cache : boolean, optional
            Cache flag (default: False).
        mask_and_scale : boolean, optional
            Mask and scale during reading (default: False).
        """
        super().__init__(**kwargs)
        self.var["row"] = row_var
        self.filename = filename

        # read location information
        with xr.open_dataset(self.filename,
                             mask_and_scale=self.mask_and_scale) as ncfile:
            var_list = [
                self.var["lon"],
                self.var["lat"],
                self.loc["ids"],
                self.var["row"],
            ]

            if self.cache:
                self.dataset = ncfile.load()

                self.locations = self.dataset[var_list].to_dataframe()
                self.locations[self.var["row"]] = np.cumsum(
                    self.locations[self.var["row"]])
            else:
                self.dataset = None

                self.locations = ncfile[var_list].to_dataframe()
                self.locations[self.var["row"]] = np.cumsum(
                    self.locations[self.var["row"]])

    @property
    def ids(self):
        """
        Location IDs property.

        Returns
        -------
        location_id : numpy.ndarray
            Location IDs.
        """
        return self.locations.location_id

    @property
    def lons(self):
        """
        Longitude coordinates property.

        Returns
        -------
        lon : numpy.ndarray
            Longitude coordinates.
        """
        return self.locations.lon

    @property
    def lats(self):
        """
        Latitude coordinates property.

        Returns
        -------
        lat : numpy.ndarray
            Latitude coordinates.
        """
        return self.locations.lat

    def read(self, location_id, variables=None):
        """
        Read a timeseries for a given location_id.

        Parameters
        ----------
        location_id : int
            Location_id to read.
        variables : list or None
            A list of parameter-names to read. If None, all parameters are read.
            If None, all parameters will be read. The default is None.

        Returns
        -------
        df : pandas.DataFrame
            A pandas.DataFrame containing the timeseries for the location_id.
        """
        pos = self.locations.location_id == location_id

        if not pos.any():
            print(f"location_id not found: {location_id}")
            data = None
        else:
            sel = self.locations[pos]
            i = sel.index.values[0]

            r_to = sel.row_size[i]
            if i > 0:
                r_from = int(self.locations.iloc[[i - 1]].row_size[i - 1])
            else:
                r_from = 0

            if self.cache:
                data = self.dataset.sel(locations=i, obs=slice(r_from, r_to))
            else:
                with xr.open_dataset(
                        self.filename,
                        mask_and_scale=self.mask_and_scale) as dataset:
                    data = dataset.sel(locations=i, obs=slice(r_from, r_to))

            if variables is not None:
                data = data[variables]

        return data

    def read_2d(self, variables=None):
        """
        (Draft!) Read all time series into 2d array.

        1d data: 1, 2, 3, 4, 5, 6, 7, 8
        row_size: 3, 2, 1, 2
        2d data:
        1 2 3 0 0
        4 5 0 0 0
        6 0 0 0 0
        7 8 0 0 0
        """
        row_size = np.array([3, 2, 1, 2])
        y = vrange(np.zeros_like(row_size), row_size)
        x = np.arange(row_size.size).repeat(row_size)
        target = np.zeros((4, 3))
        target[x, y] = np.arange(1, 9)
        print(target)


def vrange(starts, stops):
    """
    Create concatenated ranges of integers for multiple start/stop values.

    Parameters
    ----------
    starts : numpy.ndarray
        Starts for each range.
    stops : numpy.ndarray
        Stops for each range (same shape as starts).

    Returns
    -------
    ranges : numpy.ndarray
        Concatenated ranges.

    Example
    -------
        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])
    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())

def var_order(dataset):
    """
    Returns a reasonable variable order for a ragged array dataset,
    based on that used in existing datasets.

    Puts the count/index variable first depending on the ragged array type,
    then lon, lat, alt, location_id, location_description, and time,
    followed by the rest of the variables in the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Ordered dataset.
    """
    if "row_size" in dataset.data_vars:
        first_var = "row_size"
    elif "locationIndex" in dataset.data_vars:
        first_var = "locationIndex"
    else:
        raise ValueError("No row_size or locationIndex in dataset. \
                          Cannot determine if indexed or ragged")

    order = [
        first_var,
        "lon",
        "lat",
        "alt",
        "location_id",
        "location_description",
        "time",
    ]
    order.extend([v for v in dataset.data_vars if v not in order])

    return dataset[order]


def indexed_to_contiguous(dataset):
    """
    Convert an indexed dataset to a contiguous ragged array dataset.
    Assumes that index variable is named "locationIndex".

    Parameters
    ----------
    dataset : xarray.Dataset, Path
        Dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Converted dataset.
    """
    if isinstance(dataset, (str, Path)):
        with xr.open_dataset(dataset, mask_and_scale=False) as ds:
            return indexed_to_contiguous(ds)

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            "dataset must be an xarray.Dataset or a path to a netCDF file")
    if "locationIndex" not in dataset:
        raise ValueError("dataset must have a locationIndex variable")

    dataset = dataset.sortby(["locationIndex", "time"])

    # # this alone is simpler than what follows if one can assume that the locationIndex
    # # is an integer sequence with no gaps
    # dataset["row_size"] = np.unique(dataset["locationIndex"], return_counts=True)[1]

    idxs, sizes = np.unique(dataset.locationIndex, return_counts=True)
    row_size = np.zeros_like(dataset.location_id.values)
    row_size[idxs] = sizes
    dataset["row_size"] = ("locations", row_size)

    dataset = dataset.drop_vars(["locationIndex"])

    return var_order(dataset)


def contiguous_to_indexed(dataset):
    """
    Convert a contiguous ragged array to an indexed ragged array.
    Assumes count variable is named "row_size".

    Parameters
    ----------
    dataset : xarray.Dataset, Path
        Dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Converted dataset.
    """
    if isinstance(dataset, (str, Path)):
        with xr.open_dataset(dataset, mask_and_scale=False) as ds:
            return contiguous_to_indexed(ds)

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            "dataset must be an xarray.Dataset or a path to a netCDF file")
    if "row_size" not in dataset:
        raise ValueError("dataset must have a row_size variable")

    row_size = np.where(dataset["row_size"].values > 0,
                        dataset["row_size"].values, 0)

    locationIndex = np.repeat(np.arange(row_size.size), row_size)
    dataset["locationIndex"] = ("obs", locationIndex)
    dataset = dataset.drop_vars(["row_size"])

    return dataset


def dataset_ra_type(dataset):
    """
    Determine if a dataset is indexed or contiguous.
    Assumes count variable for contiguous RA is named "row_size".
    Assumes index variable for indexed RA is named "locationIndex".

    Parameters
    ----------
    dataset : xarray.Dataset, Path
        Dataset.
    """
    if "locationIndex" in dataset:
        return "indexed"
    if "row_size" in dataset:
        return "contiguous"

    raise ValueError("Dataset must have either locationIndex or row_size.\
                     Cannot determine if ragged array is indexed or contiguous"
                     )


def set_attributes(dataset, attributes=None):
    """
    Set default attributes for a contiguous or indexed ragged dataset.

    Parameters
    ----------
    dataset : xarray.Dataset, Path
        Dataset.
    attributes : dict, optional
        Attributes.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with attributes.
    """
    if attributes is None:
        attributes = {}

    if dataset_ra_type(dataset) == "contiguous":
        first_var = "row_size"
    elif dataset_ra_type(dataset) == "indexed":
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

    default_attrs = {
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

    attributes = {**default_attrs, **attributes}

    for var, attrs in attributes.items():
        dataset[var] = dataset[var].assign_attrs(attrs)
        if var in [
                "row_size", "locationIndex", "location_id",
                "location_description"
        ]:
            dataset[var].encoding["coordinates"] = None

    date_created = datetime.now().isoformat(" ", timespec="milliseconds")[:-6]
    dataset.attrs["date_created"] = date_created

    return dataset


def create_encoding(dataset, custom_encoding=None):
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
    dataset : xarray.Dataset
        Dataset.
    custom_encoding : dict, optional
        Custom encodings.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with encodings.
    """
    if custom_encoding is None:
        custom_encoding = {}

    if "row_size" in dataset.data_vars:
        first_var = "row_size"
    elif "locationIndex" in dataset.data_vars:
        first_var = "locationIndex"
    else:
        raise ValueError("No row_size or locationIndex in dataset. \
                          Cannot determine if indexed or ragged")

    # default encodings for coordinates and row_size
    default_encoding = {
        first_var: {
            "dtype": "int64",
        },
        "lon": {
            "dtype": "float32",
        },
        "lat": {
            "dtype": "float32",
        },
        "alt": {
            "dtype": "float32",
        },
        "location_id": {
            "dtype": "int64",
        },
        # # for some reason setting this throws an error but
        # # it gets handled properly automatically when left out
        # "location_description": {
        #     "dtype": "str",
        # },
        "time": {
            "dtype": "float64",
            "units": "days since 1900-01-01 00:00:00",
        },
    }

    for _, var_encoding in default_encoding.items():
        var_encoding["_FillValue"] = None
        var_encoding["zlib"] = True
        var_encoding["complevel"] = 4

    default_encoding.update({
        var: {
            "dtype": dtype,
            "zlib": bool(np.issubdtype(dtype, np.number)),
            "complevel": 4,
            "_FillValue": None,
        }
        for var, dtype in dataset.dtypes.items()
    })

    encoding = {**default_encoding, **custom_encoding}

    return encoding


def udunits_name_to_datetime(unit):
    """
    Convert a udunits name to a datetime unit

    Parameters
    ----------
    unit : str
        Unit string.

    Returns
    -------
    dt_unit : str
        Datetime unit.
    """
    lookup = {
        "days": "D",
        "hours": "h",
        "minutes": "m",
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
        "nanoseconds": "ns",
    }

    return lookup[unit]


class RACollection():
    """
    Collection of Ragged Array files.
    """

    def __init__(self, file_list):
        """
        Initialize.

        Parameters
        ----------
        file_list : list
            List of filenames.
        """
        self.file_list = file_list

    def preprocess(self, dataset):
        """
        Pre-processing.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset.
        """
        if dataset_ra_type(dataset) != "indexed":
            dataset = contiguous_to_indexed(dataset)

        if "time" in dataset.dims:
            dataset = dataset.rename_dims({"time": "obs"})
        dataset = dataset.dropna(dim="locations", subset=["location_id"])

        dataset["locationIndex"] = (
            "obs",
            self.sorter[np.searchsorted(
                self.location_vars["location_id"].values,
                dataset["location_id"].values[dataset["locationIndex"]],
                sorter=self.sorter,
            )],
        )

        dataset = dataset.drop_dims("locations")
        for var, var_data in self.location_vars.items():
            dataset[var] = ("locations", var_data.values)
        dataset = dataset.set_coords(["lon", "lat", "alt", "time"])
        try:
            # not sure how to test if time is already an index except like this
            dataset = dataset.reset_index("time")
        except ValueError:
            pass

        return dataset

    def merge(self, out_format="contiguous", dupe_window=None):
        """
        Merge files.

        Parameters
        ----------
        out_format : str, optional
            Output format (default: "contiguous").
        dupe_window : numpy.timedelta64, optional
            Check duplicates for given window (default: None).

        Returns
        -------
        merged_ds
            Merged dataset.
        """
        if dupe_window is None:
            dupe_window = np.timedelta64(10, "m")

        with xr.open_mfdataset(
                self.file_list,
                concat_dim="locations",
                combine="nested",
                preprocess=lambda ds: ds[[
                    var for var in ds.variables
                    if ("locations" in ds[var].dims) and
                    (var not in ["row_size", "locationIndex"])
                ]],
                # parallel=True,
        ) as locs_merged:
            all_location_ids, idxs = np.unique(
                locs_merged["location_id"].values, return_index=True)
            self.location_vars = {
                var: locs_merged[var][idxs]
                for var in locs_merged.variables
            }

            self.sorter = np.argsort(self.location_vars["location_id"].values)

        with xr.open_mfdataset(
                self.file_list,
                concat_dim="obs",
                data_vars="minimal",
                coords="minimal",
                preprocess=self.preprocess,
                combine="nested",
                mask_and_scale=False,
                # parallel=True,
        ) as merged_ds:
            merged_ds.load()

            # deduplicate
            merged_ds = merged_ds.sortby(["sat_id", "locationIndex", "time"])

            dupl = np.insert(
                (abs(merged_ds["time"].values[1:] -
                     merged_ds["time"].values[:-1]) < dupe_window),
                0,
                False,
            )
            merged_ds = merged_ds.sel(obs=~dupl)

            if out_format == "contiguous":
                merged_ds = indexed_to_contiguous(merged_ds)
            # set variable order
            merged_ds = var_order(merged_ds)
            # set dataset ID
            # TODO: should probably change this
            merged_ds.attrs["id"] = ", ".join(
                set([f.name for f in self.file_list]))
            merged_ds.encoding["unlimited_dims"] = []

        return merged_ds


def merge_netCDFs(file_list, out_format="contiguous", dupe_window=None):
    """
    Merge netCDF files.

    Parameters
    ----------
    file_list : list
        List of filenames.
    out_format : str, optional
        Output format (default: "contiguous").
    dupe_window : numpy.timedelta64, optional
        Check duplicates for given window (default: None).

    Returns
    -------
    merged_ds : xarray.Dataset
        Merged dataset.
    """
    return RACollection(file_list).merge(out_format, dupe_window)
    # if dupe_window is None:
    #     dupe_window = np.timedelta64(10, "m")
    # location_vars = {}
    # locationIndex = np.array([], dtype=np.int64)

    # for ncfile in file_list:
    #     with xr.open_dataset(ncfile, mask_and_scale=False) as ds:
    #         if dataset_ra_type(ds) != "indexed":
    #             ds = contiguous_to_indexed(ds)
    #         location_id = ds["location_id"].values[
    #             ds["location_id"].values != int64_nan + 2
    #         ]
    #         index = np.arange(0, len(location_id))
    #         if location_vars.get("location_id") is not None:
    #             common_locations = location_vars["location_id"]
    #         else:
    #             common_locations = np.array([], dtype=ds["location_id"].dtype)
    #         new_loc_indices = index[~np.isin(location_id, common_locations)]

    #         for v in ds.variables:
    #             if "locations" in ds[v].dims:
    #                 if v not in location_vars:
    #                     location_vars[v] = np.array([], dtype=ds[v].dtype)
    #                 location_vars[v] = np.concatenate(
    #                     (location_vars[v], ds[v].values[new_loc_indices])
    #                 )
    #         sorter = np.argsort(location_vars["location_id"])
    #         locationIndex = np.concatenate(
    #             (
    #                 locationIndex,
    #                 sorter[
    #                     np.searchsorted(
    #                         location_vars["location_id"],
    #                         ds["location_id"].values[ds["locationIndex"]],
    #                         sorter=sorter,
    #                     )
    #                 ],
    #             )
    #         )

    # def preprocess(dataset):
    #     if dataset_ra_type(dataset) != "indexed":
    #         dataset = contiguous_to_indexed(dataset)

    #     if "time" in dataset.dims:
    #         dataset = dataset.rename_dims({"time": "obs"})
    #     dataset = dataset.dropna(dim="locations", subset=["location_id"])
    #     dataset = dataset.drop_dims("locations")
    #     for var, var_data in location_vars.items():
    #         dataset[var] = xr.DataArray(var_data, dims="locations")
    #     dataset = dataset.set_coords(["lon", "lat", "alt", "time"])
    #     try:
    #         # not sure how to test if time is already an index except like this
    #         dataset = dataset.reset_index("time")
    #     except ValueError:
    #         pass
    #     return dataset

    # with xr.open_mfdataset(
    #     file_list,
    #     concat_dim="obs",
    #     data_vars="minimal",
    #     coords="minimal",
    #     preprocess=preprocess,
    #     combine="nested",
    #     mask_and_scale=False
    # ) as merged_ds:
    #     # merged_ds.load()
    #     location_id = np.unique(merged_ds["location_id"].data)
    #     merged_ds["locationIndex"] = xr.DataArray(locationIndex, dims="obs")

    #     # deduplicate
    #     merged_ds = merged_ds.sortby(["sat_id", "locationIndex", "time"])

    #     # time_units = merged_ds["time"].attrs["units"].split("since")[0].strip()
    #     # time_units = udunits_name_to_datetime(time_units)
    #     dupl = np.insert(
    #         (
    #             abs(merged_ds["time"].values[1:] - merged_ds["time"].values[:-1])
    #             < dupe_window
    #         ),
    #         0,
    #         False,
    #     )
    #     merged_ds = merged_ds.sel(obs=~dupl)

    #     if out_format == "contiguous":
    #         merged_ds = indexed_to_contiguous(merged_ds)
    #     # set variable order
    #     merged_ds = var_order(merged_ds)
    #     # set dataset ID
    #     # TODO: should probably change this
    #     merged_ds.attrs["id"] = ", ".join(set([f.name for f in file_list]))
    #     merged_ds.encoding["unlimited_dims"] = []
    # return merged_ds


class CellFileCollectionTimeSeries():
    """
    Collection of grid cell file collections
    """

    def __init__(
            self,
            collections,
            ascat_id=None,
            # ioclass=None,
            common_grid=None,
            dask_scheduler="threads",
            **kwargs
    ):
        """
        Initialize.

        Parameters
        ----------
        collections: list of str or CellFileCollection
        """

        if isinstance(collections, (str, Path)):
            collections = [collections]
        if isinstance(collections[0], (str, Path)):
            # all_subdirs = [subdir for c in collections for subdir in Path(c).glob("**/**/")]
            # all_subdirs = {file.parent for c in collections for file in Path(c).glob("**/*.nc")}
            all_subdirs = [
                Path(r) for c in collections for (r, d, f) in os.walk(c) if not d
            ]
            self.collections = [
                CellFileCollection(subdir, ascat_id, ioclass_kws=kwargs)
                for subdir in all_subdirs
            ]
        else:
            self.collections = collections
        self.common_grid = common_grid

        self.grids = []
        for c in self.collections:
            if c.grid not in self.grids:
                self.grids.append(c.grid)

        if dask_scheduler is not None:
            dask.config.set(scheduler=dask_scheduler)
        # self._client = Client(n_workers=1, threads_per_worker=16)


    def _preprocess(self, dataset):
        """
        Pre-processing.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset.
        """
        if dataset_ra_type(dataset) != "indexed":
            dataset = contiguous_to_indexed(dataset)

        # if "time" in dataset.dims:
        #     dataset = dataset.rename_dims({"time": "obs"})
        # if "locations" is in the dimensions, then we have
        # a multi-location dataset.
        if "locations" in dataset.dims:
            dataset = dataset.dropna(dim="locations", subset=["location_id"])

            try:
                dataset["locationIndex"] = (
                    "obs",
                    self.sorter[np.searchsorted(
                        self.location_vars["location_id"].values,
                        dataset["location_id"].values[dataset["locationIndex"]],
                        sorter=self.sorter,
                    )],
                )
                # print(f"{random_int} sorter size:", self.sorter.size)
                # print(f"{random_int}:", self.location_vars["location_id"].values.size)
                # print(f"{random_int}:", dataset["location_id"].values[dataset["locationIndex"]].size)
            except:
                random_int = np.random.randint(0, 100)
                if self.sorter.size != self.location_vars["location_id"].values.size:
                    print(f"{random_int} sorter size:", self.sorter.size)
                    print(f"{random_int}:", self.location_vars["location_id"].values.size)
                    print(f"{random_int}:", dataset["location_id"].values[dataset["locationIndex"]].size)
                #
                # print(np.searchsorted(
                #     self.location_vars["location_id"].values,
                #     dataset["location_id"].values[dataset["locationIndex"]],
                #     sorter=self.sorter,
                # ))
                raise

            dataset = dataset.drop_dims("locations")

        # if not, we just have a single location, and logic is different
        else:
            dataset["locationIndex"] = (
                "obs",
                self.sorter[np.searchsorted(
                    self.location_vars["location_id"].values,
                    np.repeat(dataset["location_id"].values, dataset["locationIndex"].size),
                    sorter=self.sorter,
                )],
            )

        for var, var_data in self.location_vars.items():
            dataset[var] = ("locations", var_data.values)
        dataset = dataset.set_coords(["lon", "lat", "alt", "time"])
        try:
            # can't figure out how to test if time is already an index except like this
            dataset = dataset.reset_index("time")
        except ValueError:
            pass

        return dataset

    def _only_locations(self, ds):
        return ds[[
            var
            for var in ds.variables
            if ("obs" not in ds[var].dims)
            and var not in ["row_size", "locationIndex"]
        ]]

    def _subcollection_cells(self):
        return {c for coll in self.collections for c in coll.cells_in_collection}
    # def _merge_ds(self, ds_list):
    #     locs_merged = xr.combine_nested(
    #         [self._only_locations(ds) for ds in ds_list], concat_dim="locations"
    #     )
    #     all_location_ids, idxs = np.unique(
    #         locs_merged["location_id"].values, return_index=True)
    #     self.location_vars = {
    #         var: locs_merged[var][idxs]
    #         for var in locs_merged.variables
    #     }

    #     self.sorter = np.argsort(self.location_vars["location_id"].values)

    #     locs_merged.close()

    def _read_cells(self, cells, out_grid=None, out_format="contiguous", dupe_window=None,
                    **kwargs):
        cells = cells if isinstance(cells, list) else [cells]

        if dupe_window is None:
            dupe_window = np.timedelta64(10, "m")

        data = [coll.read(cell=cell,
                          new_grid=out_grid,
                          mask_and_scale=False,
                          **kwargs)
                for coll in self.collections
                for cell in cells]

        data = [ds for ds in data if ds is not None]
        # print([self._only_locations(ds) for ds in data])
        locs_merged = xr.combine_nested(
            [self._only_locations(ds) for ds in data], concat_dim="locations"
        )

        all_location_ids, idxs = np.unique(
            locs_merged["location_id"].values, return_index=True)

        self.location_vars = {
            var: locs_merged[var][idxs]
            for var in locs_merged.variables
        }

        self.sorter = np.argsort(self.location_vars["location_id"].values)

        locs_merged.close()

        merged_ds = xr.combine_nested(
            [self._preprocess(ds) for ds in data],
            concat_dim="obs",
            data_vars="minimal",
            coords="minimal",
        )

        return merged_ds

    def _read_locations(self, location_ids, out_grid=None, out_format="contiguous", dupe_window=None,
                        **kwargs):
        location_ids = location_ids if isinstance(location_ids, list) else [location_ids]

        if dupe_window is None:
            dupe_window = np.timedelta64(10, "m")

        # all data here is converted to the SAME GRID within coll.read()
        # before being merged later
        data = [d for d in (coll.read(location_id=location_id,
                                      new_grid=out_grid,
                                      mask_and_scale=False,
                                      **kwargs)
                for coll in self.collections
                            for location_id in location_ids) if d is not None]

        # coords="all" is necessary in case one of the coords has nan values
        # (e.g. altitude, in the case of ASCAT H129)
        locs_merged = xr.combine_nested(
            [self._only_locations(ds) for ds in data], concat_dim="locations",
            coords="all",
        )
        all_location_ids, idxs = np.unique(
            locs_merged["location_id"].values, return_index=True)

        # maybe pass these as args to preprocess instead of setting them as attributes?
        self.location_vars = {
            var: locs_merged[var][idxs]
            for var in locs_merged.variables
        }
        self.sorter = np.argsort(self.location_vars["location_id"].values)

        locs_merged.close()

        merged_ds = xr.combine_nested(
            [self._preprocess(ds) for ds in data],
            concat_dim="obs",
            data_vars="minimal",
            coords="minimal",
        )

        return merged_ds

    def read(self, cell=None, location_id=None, bbox=None, out_grid=None, **kwargs):
        out_grid = out_grid or self.common_grid
        if (len(self.grids) > 1) and out_grid is None:
            raise ValueError("Multiple grids found, need to specify out_grid\
                            as argument to read function or common_grid as\
                            argument to __init__")

        if cell is not None:
            data = self._read_cells(cell, out_grid, **kwargs)
        elif location_id is not None:
            data = self._read_locations(location_id, out_grid, **kwargs)
        elif bbox is not None:
            raise NotImplementedError
        else:
            raise ValueError("Need to specify either cell, location_id or bbox")

        # check if we have multiple grid systems and run regrid function on whole
        # dataset if so
        # grids = []
        # for c in self.collections:
        #     if c.grid not in grids:
        #         grids.append(c.grid)

        # if len(grids) > 1:
        #     out_grid = (out_grid or self.common_grid)
        #     if out_grid is None:
        #         raise ValueError("Multiple grids found, need to specify out_grid\
        #                         as argument to read function or common_grid as\
        #                         argument to __init__")
        #     data = self._regrid(data, out_grid)

        return data

    # def write(self, cell=None, location_id=None, bbox=None, **kwargs):
    #     if cell == "all":
    #         cells = self.get_cells()
    #     elif cell is not None:
    #         cells = cell
    #     data = self.read(cell=cells, location_id=location_id, bbox=bbox, **kwargs)

    def _write_single_cell(self,out_dir, ioclass, cell, out_grid, **kwargs):
        data = self.read(cell=cell)
        writer = ioclass(data)
        writer.write(out_dir/writer.fn_format.format(cell), **kwargs)
        data.close()
        writer.close()

    def _read_single_cell(self, cell, **kwargs):
        data = self.read(cell=cell)
        return data

    def write_cells(self, out_dir, ioclass, cells=None, out_grid=None, **kwargs):
        from time import time
        out_grid = out_grid or self.common_grid
        if (len(self.grids) > 1) and out_grid is None:
            raise ValueError("Multiple grids found, need to specify out_grid\
                            as argument to read function or common_grid as\
                            argument to __init__")

        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        cells = self._subcollection_cells()

        for cell in tqdm.tqdm(cells):
            self._write_single_cell(out_dir, ioclass, cell, out_grid, **kwargs)

    def add_collection(self, collections, ascat_id=None, ioclass=None):
        """
        Add a cell file collection to the collection,
        based on file path.
        """
        new_idx = len(self.collections)
        if isinstance(collections[0], str):
            self.collections.extend(CellFileCollection(c, ascat_id) for c in collections)
        else:
            self.collections.extend(collections)

        for c in self.collections[new_idx:]:
            if c.grid not in self.grids:
                self.grids.append(c.grid)


ascat_id_dict = {
    "H129": {
        "ioclass": ASCAT_NetCDF4,
        "fn_pattern": "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-6.25-H129_C_LIIB_{date}_*_*____.nc",
        "fn_read_fmt": lambda timestamp: {"date": timestamp.strftime("%Y%m%d*")},
        "sf_pattern": {"year_folder": "{year}"},
        "sf_read_fmt": lambda timestamp: {"year_folder": {"year": f"{timestamp.year}"}},
        "date_format": "%Y%m%d%H%M%S",
        "grid": FibGrid(6.25),
        "cell_fn_format": "{:04d}.nc",
    }
}


class ASCAT_ID_Metadata():
    def __init__(self,
                 ascat_id
                 ):
        ascat_id = ascat_id.upper()
        self.ioclass = ascat_id_dict[ascat_id]["ioclass"]
        self.fn_pattern = ascat_id_dict[ascat_id]["fn_pattern"]
        self.fn_read_fmt = ascat_id_dict[ascat_id]["fn_read_fmt"]
        self.sf_pattern = ascat_id_dict[ascat_id]["sf_pattern"]
        self.sf_read_fmt = ascat_id_dict[ascat_id]["sf_read_fmt"]
        self.cell_fn_format = ascat_id_dict[ascat_id]["cell_fn_format"]
        self.date_format = ascat_id_dict[ascat_id]["date_format"]
        self.grid = ascat_id_dict[ascat_id]["grid"]
        possible_cells = self.grid.get_cells()
        self.max_cell = possible_cells.max()
        self.min_cell = possible_cells.min()


class CellFileCollection:

    """
    Grid cell files.
    """

    def __init__(self,
                 path,
                 ascat_id,
                 # ioclass,
                 cache=False,
                 # fn_format="{:04d}.nc",
                 ioclass_kws=None,
                 ):
        """
        Initialize.
        """
        self.path = path
        self.ascat_id = ASCAT_ID_Metadata(ascat_id)
        self.ioclass = self.ascat_id.ioclass
        self.grid = self.ascat_id.grid

        # possible_cells = self.grid.get_cells()
        # self.cell_max = possible_cells.max()
        # self.cell_min = possible_cells.min()

        self.fn_format = self.ascat_id.cell_fn_format
        # ASSUME THE IOCLASS RETURNS XARRAY
        # self.ioclass = ioclass
        # self.fn_format = fn_format
        self.previous_cell = None
        self.fid = None
        self.min_time = None
        self.max_time = None

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws

    @property
    def cells_in_collection(self):
        return [int(p.stem) for p in self.path.glob("*")]

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

    def _open(self, location_id=None, cell=None):
        """
        Open cell file.

        Parameters
        ----------
        location_id : int
            Location identifier.

        Returns
        -------
        success : boolean
            Flag if opening the file was successful.
        """
        success = True
        if location_id is not None:
            cell = self.grid.gpi2cell(location_id)
        filename = self._get_cell_path(cell)

        if self.previous_cell != cell:
            self.close()

            try:
                self.fid = self.ioclass(filename, **self.ioclass_kws)
            except IOError as e:
                success = False
                self.fid = None
                msg = f"I/O error({e.errno}): {e.strerror}, {filename}"
                warnings.warn(msg, RuntimeWarning)
                self.previous_cell = None
            else:
                self.previous_cell = cell

        return success

    def _read_cell(self, cell=None, **kwargs):
        """
        Read data from the entire cell.
        """
        # if there are kwargs, use them instead of self.ioclass_kws

        data = None
        if self._open(cell=cell):
            data = self.fid.read(**kwargs)

        return data

    def _read_lonlat(self, lon, lat, **kwargs):
        """
        Reading data for given longitude and latitude coordinate.

        Parameters
        ----------
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.

        Returns
        -------
        data : dict of values
            data record.
        """
        location_id, _ = self.grid.find_nearest_gpi(lon, lat)

        return self._read_location_id(location_id, **kwargs)

    def _read_location_id(self, location_id, **kwargs):
        """
        Read data for given grid point.

        Parameters
        ----------
        location_id : int
            Location identifier.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        data = None

        if self._open(location_id=location_id):
            data = self.fid.read(location_id=location_id, **kwargs)

        return data

    def _index_time(self):
        # open just the time var as mf dataset to get min/max time
        # with self.ioclass(list(self.path.glob("*")),
        #                   concat_dim="time",
        #                   combine="nested",
        #                   data_vars="minimal",
        #                   preprocess=lambda ds: xr.decode_cf(ds[["time"]]),
        #                   # drop_vars=
        #                   # keep_vars=["time"]
        #                   decode_cf=False,
        #                   ) as f:
        #     self.min_time, self.max_time = f.date_range()
        for f in self.path.glob("*"):
            with self.ioclass(f, **self.ioclass_kws) as f:
                # print(self.min_time, self.max_time)
                if self.min_time is None:
                    (self.min_time, self.max_time) = f.date_range
                else:
                    (min_time, max_time) = f.date_range
                    self.min_time = min(self.min_time, min_time)
                    self.max_time = max(self.max_time, max_time)

        return self.min_time, self.max_time

    def _get_cell_path(self, cell=None, location_id=None):
        """
        Get path to cell file given cell number or location id.
        Returns a path whether the file exists or not, as long
        as the cell number or location id is within the grid.
        """
        if location_id is not None:
            cell = self.grid.gpi2cell(location_id)
        elif cell is None:
            raise ValueError("Either location_id or cell must be given")

        if (cell > self.ascat_id.max_cell) or (cell < self.ascat_id.min_cell):
            raise ValueError(f"Cell {cell} is not in grid")

        return self.path / self.fn_format.format(cell)

    def _convert_to_grid(self, data, new_grid, old_grid=None):
        """
        Convert the data to a new grid.
        ACTUALLY YOU DON'T NEED THIS JUST DO IT WHEN MERGING
        ALL YOU NEED IS THE LONS AND LATS
        """
        if old_grid is None:
            old_grid = self.grid
        if (new_grid == old_grid) or (data is None):
            return data
        # old_lon = np.atleast_1d(data["lon"].values)
        # old_lat = np.atleast_1d(data["lat"].values)
        # lookup = grids.BasicGrid(old_lon, old_lat).calc_lut(new_grid)
        # new_loc_ids, _ = np.unique(lookup, return_inverse=True)
        # perhaps this could be cached to prevent calculating each time
        lookup = old_grid.calc_lut(new_grid)
        # if self.fid.ra_type == "contiguous":
        #     all_lids = np.repeat(data["location_id"].values, data["row_size"].values)
        #     new_lids = lookup[all_lids]
        #     location_id, row_size = np.unique(new_lids, return_counts=True)
        #     lon, lat = new_grid.gpi2lonlat(location_id)
        #     data["new_lids"] = ("obs", new_lids)
        #     data = data.sortby(["new_lids", "time"])
        #     data["row_size"] = ("obs", row_size)
        #     #### needs more
        #     #### possibly drop this and just assume indexed
        #     #### or rather
        #     #### ask if I can get rid of altitude and location_description
        #     data = data.drop_vars("new_lids")

        # elif self.fid.ra_type == "indexed":
        if "locations" in data.dims:
            all_lids = data["location_id"].values[data["locationIndex"].values]
            new_lids = lookup[all_lids]
            location_id, locationIndex = np.unique(new_lids, return_inverse=True)
            lon, lat = new_grid.gpi2lonlat(location_id)
            alt = np.repeat(np.atleast_1d(data["alt"].values)[0], location_id.size)
            location_description = np.repeat(
                np.atleast_1d(data["location_description"].values)[0], location_id.size
            )
            data = data.drop_dims("locations")
            # no need to overwrite these in the single-lcoation case
            data["alt"] = ("locations", alt)
            data["location_description"] = ("locations", location_description)
        else:
            # case when data is just a single location (won't have a locations dim)
            all_lids = np.repeat(data["location_id"].values, data["locationIndex"].size)
            new_lids = lookup[all_lids]
            # the below will be a tuple of the single new location id and the index
            # 0 repeated as many times as there are observations
            location_id, locationIndex = np.unique(new_lids, return_inverse=True)
            lon, lat = new_grid.gpi2lonlat(location_id)
            # alt = data["alt"].values
            # location_description = data["location_description"].values
        data["lon"] = ("locations", lon)
        data["lat"] = ("locations", lat)
        data["location_id"] = ("locations", location_id)
        data["locationIndex"] = ("obs", locationIndex)
        data = data.set_coords(["lon", "lat", "alt", "location_id"])

        return data

    def read(self, cell=None, location_id=None, coords=None, new_grid=None, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it
        """
        # new_grid = kwargs.pop("new_grid", False)
        kwargs = {**self.ioclass_kws, **kwargs}
        if cell is not None:
            data = self._read_cell(cell, **kwargs)
        elif location_id is not None:
            if new_grid is not False:
                warnings.warn("You have specified a new_grid but are searching for a location_id.\
                Currently, the location_id argument searches the original grid. The returned data\
                will be converted to the new grid and will probably have different location_id values\
                from those you searched for.")
            data = self._read_location_id(location_id, **kwargs)
        elif coords is not None:
            data = self._read_lonlat(coords[0], coords[1], **kwargs)
        else:
            raise ValueError("Either cell, location_id or coords (lon, lat) must be given")

        if new_grid is not None:
            data = self._convert_to_grid(data, new_grid)

        return data

    # def iter_locations(self, **kwargs):
    #     """
    #     Yield all values for all locations.

    #     Yields
    #     ------
    #     data : numpy.ndarray
    #         Data
    #     location_id : int
    #         Location identifier.
    #     """
    #     if "ll_bbox" in kwargs:
    #         latmin, latmax, lonmin, lonmax = kwargs["ll_bbox"]
    #         location_ids = self.grid.get_bbox_grid_points(
    #             latmin, latmax, lonmin, lonmax)
    #         kwargs.pop("ll_bbox", None)
    #     elif "gpis" in kwargs:
    #         subgrid = self.grid.subgrid_from_gpis(kwargs["gpis"])
    #         gp_info = list(subgrid.grid_points())
    #         location_ids = np.array(gp_info, dtype=np.int32)[:, 0]
    #         kwargs.pop("gpis", None)
    #     else:
    #         gp_info = list(self.grid.grid_points())
    #         location_ids = np.array(gp_info, dtype=np.int32)[:, 0]

    #     for location_id in location_ids:
    #         try:
    #             data = self._read_location_id(location_id, **kwargs)
    #         except IOError as e:
    #             msg = f"I/O error({e.errno}): {e.strerror}, {location_id}"
    #             warnings.warn(msg, RuntimeWarning)
    #             data = None

    #         yield data, location_id

    # def flush(self):
    #     """
    #     Flush data.
    #     """
    #     if self.fid is not None:
    #         self.fid.flush()

    def close(self):
        """
        Close file.
        """
        if self.fid is not None:
            self.fid.close()
            self.fid = None



class SwathFileCollection:
    """
    Swath file collection.
    """

    def __init__(self,
                 path,
                 ascat_id,
                 # ioclass,
                 # start_dt=None,
                 # end_dt=None,
                 # delta_dt=None,
                 # dtype_dt="datetime64[ns]",
                 cache=False,
                 # fn_format="{:04d}.nc",
                 ioclass_kws=None,
                 ):
        self.path = path
        self.ascat_id = ASCAT_ID_Metadata(ascat_id)
        self.ioclass = self.ascat_id.ioclass
        self.grid = self.ascat_id.grid

        # possible_cells = self.grid.get_cells()
        # self.cell_max = possible_cells.max()
        # self.cell_min = possible_cells.min()

        self.fn_pattern = self.ascat_id.fn_pattern
        self.sf_pattern = self.ascat_id.sf_pattern
        self.fn_read_fmt = self.ascat_id.fn_read_fmt
        self.sf_read_fmt = self.ascat_id.sf_read_fmt
        self.date_format = self.ascat_id.date_format
        self.chron_files = ChronFiles(self.path,
                                      dummy_filesearch,
                                      self.fn_pattern,
                                      self.sf_pattern,
                                      None,
                                      True,
                                      self.fn_read_fmt,
                                      self.sf_read_fmt,)

        # self._start_dt = start_dt or np.datetime64("1970-01-01")
        # self._end_dt = end_dt or np.datetime64("2100-01-01")
        # self._delta_dt = delta_dt or np.timedelta64(7, "D")
        # self._time_array = np.arange(self._start_dt, self._end_dt, self._delta_dt, dtype=dtype_dt)
        # # the last datetime may be before the end of the period depending on delta_dt,
        # # so we adjust it to the end of the period. Therefore the last set of files may cover
        # # an interval longer than delta_dt.
        # self._time_array[-1] = self._end_dt


        # self.fn_format = fn_format
        self.previous_cell = None
        self.fid = None
        self.min_time = None
        self.max_time = None

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws

    @property
    def cells_in_collection(self):
        # return [int(p.stem) for p in self.path.glob("*")]
        pass

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

    def _get_filenames(self, start_dt, end_dt):
        """
        Get filenames for the given time range.

        Parameters
        ----------
        start_dt : datetime.datetime
            Start time.
        end_dt : datetime.datetime
            End time.

        Returns
        -------
        fnames : list of pathlib.Path
            List of filenames.
        """

        fnames = self.chron_files.search_period(start_dt, end_dt, date_str=self.date_format)
        return fnames

    def _open(self, location_id=None, cell=None):
        """
        Open swath file.

        Parameters
        ----------
        location_id : int
            Location identifier.

        Returns
        -------
        success : boolean
            Flag if opening the file was successful.
        """
        success = False
        return success

    def _read_cell(self, cell=None, **kwargs):
        """
        Read data from the entire cell.
        """
        # if there are kwargs, use them instead of self.ioclass_kws

        data = None
        if self._open(cell=cell):
            data = self.fid.read(**kwargs)

        return data

    def _read_lonlat(self, lon, lat, **kwargs):
        """
        Reading data for given longitude and latitude coordinate.

        Parameters
        ----------
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.

        Returns
        -------
        data : dict of values
            data record.
        """
        location_id, _ = self.grid.find_nearest_gpi(lon, lat)

        return self._read_location_id(location_id, **kwargs)

    def _read_location_id(self, location_id, **kwargs):
        """
        Read data for given grid point.

        Parameters
        ----------
        location_id : int
            Location identifier.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        data = None

        if self._open(location_id=location_id):
            data = self.fid.read(location_id=location_id, **kwargs)

        return data

    def _get_cell_path(self, cell=None, location_id=None):
        """
        Get path to cell file given cell number or location id.
        Returns a path whether the file exists or not, as long
        as the cell number or location id is within the grid.
        """
        if location_id is not None:
            cell = self.grid.gpi2cell(location_id)
        elif cell is None:
            raise ValueError("Either location_id or cell must be given")

        if (cell > self.ascat_id.max_cell) or (cell < self.ascat_id.min_cell):
            raise ValueError(f"Cell {cell} is not in grid")

        return self.path / self.fn_format.format(cell)

    def _convert_to_grid(self, data, new_grid, old_grid=None):
        """
        Convert the data to a new grid.
        """
        # if old_grid is None:
        #     old_grid = self.grid
        # if (new_grid == old_grid) or (data is None):
        #     return data
        # lookup = old_grid.calc_lut(new_grid)

        # if "locations" in data.dims:
        #     all_lids = data["location_id"].values[data["locationIndex"].values]
        #     new_lids = lookup[all_lids]
        #     location_id, locationIndex = np.unique(new_lids, return_inverse=True)
        #     lon, lat = new_grid.gpi2lonlat(location_id)
        #     alt = np.repeat(np.atleast_1d(data["alt"].values)[0], location_id.size)
        #     location_description = np.repeat(
        #         np.atleast_1d(data["location_description"].values)[0], location_id.size
        #     )
        #     data = data.drop_dims("locations")
        #     # no need to overwrite these in the single-lcoation case
        #     data["alt"] = ("locations", alt)
        #     data["location_description"] = ("locations", location_description)
        # else:
        #     # case when data is just a single location (won't have a locations dim)
        #     all_lids = np.repeat(data["location_id"].values, data["locationIndex"].size)
        #     new_lids = lookup[all_lids]
        #     # the below will be a tuple of the single new location id and the index
        #     # 0 repeated as many times as there are observations
        #     location_id, locationIndex = np.unique(new_lids, return_inverse=True)
        #     lon, lat = new_grid.gpi2lonlat(location_id)

        # data["lon"] = ("locations", lon)
        # data["lat"] = ("locations", lat)
        # data["location_id"] = ("locations", location_id)
        # data["locationIndex"] = ("obs", locationIndex)
        # data = data.set_coords(["lon", "lat", "alt", "location_id"])

        return data

    def read(self, cell=None, location_id=None, coords=None, new_grid=None, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it
        """
        # new_grid = kwargs.pop("new_grid", False)
        # kwargs = {**self.ioclass_kws, **kwargs}
        # if cell is not None:
        #     data = self._read_cell(cell, **kwargs)
        # elif location_id is not None:
        #     if new_grid is not False:
        #         warnings.warn("You have specified a new_grid but are searching for a location_id.\
        #         Currently, the location_id argument searches the original grid. The returned data\
        #         will be converted to the new grid and will probably have different location_id values\
        #         from those you searched for.")
        #     data = self._read_location_id(location_id, **kwargs)
        # elif coords is not None:
        #     data = self._read_lonlat(coords[0], coords[1], **kwargs)
        # else:
        #     raise ValueError("Either cell, location_id or coords (lon, lat) must be given")

        # if new_grid is not None:
        #     data = self._convert_to_grid(data, new_grid)

        return data

    def stack_cells(self, cells):
        """
        Stack data from a list of cells into one dataset for
        each cell.
        """
        pass

    # def flush(self):
    #     """
    #     Flush data.
    #     """
    #     if self.fid is not None:
    #         self.fid.flush()

    def close(self):
        """
        Close file.
        """
        if self.fid is not None:
            self.fid.close()
            self.fid = None

class dummy_filesearch:
    """
    dummy class for ChronFiles
    """

    def __init__(self, filename, mode="r"):
        pass


    def read(self):
        return None

    def read_period(self, dt_start, dt_end):
        return None

    def write(self, data):
        pass

    @staticmethod
    def merge(data):
        return None
