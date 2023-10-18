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
from datetime import datetime
from pathlib import Path

import xarray as xr
import numpy as np

int8_nan = np.iinfo(np.int8).max
int64_nan = np.iinfo(np.int64).min
NC_FILL_FLOAT = np.float32(9969209968386869046778552952102584320)


class RAFile:
    """
    Ragged array representation

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
        super().__init__(**kwargs)
        self.filename = filename

        with xr.open_dataset(
            self.filename, mask_and_scale=self.mask_and_scale
        ) as ncfile:
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
        Location ids.
        """
        return self.locations.location_id

    @property
    def lons(self):
        """
        Longitude coordinates.
        """
        return self.locations.lon

    @property
    def lats(self):
        """
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
                    self.filename, mask_and_scale=self.mask_and_scale
                ) as dataset:
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
        """
        super().__init__(**kwargs)
        self.var["row"] = row_var
        self.filename = filename

        with xr.open_dataset(
            self.filename, mask_and_scale=self.mask_and_scale
        ) as ncfile:
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
                    self.locations[self.var["row"]]
                )
            else:
                self.dataset = None

                self.locations = ncfile[var_list].to_dataframe()
                self.locations[self.var["row"]] = np.cumsum(
                    self.locations[self.var["row"]]
                )

    @property
    def ids(self):
        """
        Location ids.
        """
        return self.locations.location_id

    @property
    def lons(self):
        """
        Longitude coordinates.
        """
        return self.locations.lon

    @property
    def lats(self):
        """
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
                    self.filename, mask_and_scale=self.mask_and_scale
                ) as dataset:
                    data = dataset.sel(locations=i, obs=slice(r_from, r_to))

            if variables is not None:
                data = data[variables]

        return data


def var_order(dataset):
    """
    Returns a reasonable variable order for a ragged array dataset,
    based on that used in existing datasets.

    Puts the count/index variable first depending on the ragged array type,
    then lon, lat, alt, location_id, location_description, and time,
    followed by the rest of the variables in the dataset.
    """
    if "row_size" in dataset.data_vars:
        first_var = "row_size"
    elif "locationIndex" in dataset.data_vars:
        first_var = "locationIndex"
    else:
        raise ValueError(
            "No row_size or locationIndex in dataset. \
                          Cannot determine if indexed or ragged"
        )

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
    """
    if isinstance(dataset, (str, Path)):
        with xr.open_dataset(dataset, mask_and_scale=False) as ds:
            return indexed_to_contiguous(ds)

    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray Dataset or a path to a netCDF file")
    if "locationIndex" not in dataset:
        raise ValueError("dataset must have a locationIndex variable")

    dataset = dataset.sortby(["locationIndex", "time"])

    # # this alone is simpler than what follows if one can assume that the locationIndex
    # # is an integer sequence with no gaps
    # dataset["row_size"] = np.unique(dataset["locationIndex"], return_counts=True)[1]

    idxs, sizes = np.unique(dataset.locationIndex, return_counts=True)
    row_size = np.zeros_like(dataset.location_id.values)
    row_size[idxs] = sizes
    dataset["row_size"] = xr.DataArray(row_size, dims=["locations"])

    dataset = dataset.drop_vars(["locationIndex"])

    return var_order(dataset)


def contiguous_to_indexed(dataset):
    """
    Convert a contiguous ragged array to an indexed ragged array.
    Assumes count variable is named "row_size".
    """
    if isinstance(dataset, (str, Path)):
        with xr.open_dataset(dataset, mask_and_scale=False) as ds:
            return contiguous_to_indexed(ds)

    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray Dataset or a path to a netCDF file")
    if "row_size" not in dataset:
        raise ValueError("dataset must have a row_size variable")

    row_size = np.where(dataset["row_size"].values > 0, dataset["row_size"].values, 0)

    locationIndex = np.repeat(np.arange(len(row_size)), row_size)
    dataset["locationIndex"] = xr.DataArray(locationIndex, dims=["obs"])
    dataset = dataset.drop_vars(["row_size"])
    return dataset


def dataset_ra_type(dataset):
    """
    Determine if a dataset is indexed or contiguous.
    Assumes count variable for contiguous RA is named "row_size".
    Assumes index variable for indexed RA is named "locationIndex".
    """
    if "locationIndex" in dataset:
        return "indexed"
    if "row_size" in dataset:
        return "contiguous"
    raise ValueError("Dataset must have either locationIndex or row_size.\
                     Cannot determine if ragged array is indexed or contiguous")


def set_attributes(dataset, attributes=None):
    """
    Set attributes for a contiguous or indexed ragged array.
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
        "location_id": {
        },
        "location_description": {
        },
    }

    attributes = {**default_attrs, **attributes}

    for var, attrs in attributes.items():
        dataset[var] = dataset[var].assign_attrs(attrs)
        if var in ["row_size", "locationIndex", "location_id", "location_description"]:
            dataset[var].encoding["coordinates"] = None

    dataset.attrs["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    """
    if custom_encoding is None:
        custom_encoding = {}

    if "row_size" in dataset.data_vars:
        first_var = "row_size"
    elif "locationIndex" in dataset.data_vars:
        first_var = "locationIndex"
    else:
        raise ValueError(
            "No row_size or locationIndex in dataset. \
                          Cannot determine if indexed or ragged"
        )
    # hard code the default encodings for coordinates and
    # row_size stuff
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

    default_encoding.update(
        {
            var: {
                "dtype": dtype,
                "zlib": bool(np.issubdtype(dtype, np.number)),
                "complevel": 4,
                "_FillValue": None,
            }
            for var, dtype in dataset.dtypes.items()
        }
    )

    encoding = {**default_encoding, **custom_encoding}

    return encoding


def udunits_name_to_datetime(unit):
    """
    Convert a udunits name to a datetime unit
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


def merge_netCDFs(file_list, out_format="contiguous", dupe_window=None):
    if dupe_window is None:
        dupe_window = np.timedelta64(10, "m")
    location_vars = {}
    locationIndex = np.array([], dtype=np.int64)
    # # uncomment these to open all files at once and allow for
    # # passing in a list of paths or a list of open datasets
    # if isinstance(file_list[0], (str, Path)):
    #    with contextlib.ExitStack() as stack:
    #        datasets = [stack.enter_context(xr.open_dataset(f)) for f in file_list]
    #        return merge_netCDFs(datasets)
    # for ds in file_list:
    for ncfile in file_list:
        with xr.open_dataset(ncfile, mask_and_scale=False) as ds:
            if dataset_ra_type(ds) != "indexed":
                ds = contiguous_to_indexed(ds)
            location_id = ds["location_id"].values[
                ds["location_id"].values != int64_nan + 2
            ]
            index = np.arange(0, len(location_id))
            if location_vars.get("location_id") is not None:
                common_locations = location_vars["location_id"]
            else:
                common_locations = np.array([], dtype=ds["location_id"].dtype)
            new_loc_indices = index[~np.isin(location_id, common_locations)]

            for v in ds.variables:
                if "locations" in ds[v].dims:
                    if v not in location_vars:
                        location_vars[v] = np.array([], dtype=ds[v].dtype)
                    location_vars[v] = np.concatenate(
                        (location_vars[v], ds[v].values[new_loc_indices])
                    )
            sorter = np.argsort(location_vars["location_id"])
            locationIndex = np.concatenate(
                (
                    locationIndex,
                    sorter[
                        np.searchsorted(
                            location_vars["location_id"],
                            ds["location_id"].values[ds["locationIndex"]],
                            sorter=sorter,
                        )
                    ],
                )
            )

    def preprocess(dataset):
        if dataset_ra_type(dataset) != "indexed":
            dataset = contiguous_to_indexed(dataset)
        if "time" in dataset.dims:
            dataset = dataset.rename_dims({"time": "obs"})
        dataset = dataset.dropna(dim="locations", subset=["location_id"])
        dataset = dataset.drop_dims("locations")
        for var, var_data in location_vars.items():
            dataset[var] = xr.DataArray(var_data, dims="locations")
        dataset = dataset.set_coords(["lon", "lat", "alt", "time"])
        try:
            # not sure how to test if time is already an index except like this
            dataset = dataset.reset_index("time")
        except ValueError:
            pass
        return dataset

    with xr.open_mfdataset(
        file_list,
        concat_dim="obs",
        data_vars="minimal",
        coords="minimal",
        preprocess=preprocess,
        combine="nested",
        mask_and_scale=False
    ) as merged_ds:
        merged_ds.load()
        merged_ds["locationIndex"] = xr.DataArray(locationIndex, dims="obs")

        # deduplicate
        merged_ds = merged_ds.sortby(["sat_id", "locationIndex", "time"])

        # time_units = merged_ds["time"].attrs["units"].split("since")[0].strip()
        # time_units = udunits_name_to_datetime(time_units)
        dupl = np.insert(
            (
                abs(merged_ds["time"].values[1:] - merged_ds["time"].values[:-1])
                < dupe_window
            ),
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
        merged_ds.attrs["id"] = ", ".join(set([f.name for f in file_list]))
    return merged_ds


class GridCellFiles:
    def __init__(
        self, path, grid, ioclass, cache=False, fn_format="{:04d}.nc", ioclass_kws=None
    ):
        self.path = path
        self.grid = grid
        self.ioclass = ioclass
        self.fn_format = fn_format
        self.previous_cell = None
        self.fid = None

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws

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

    def _open(self, location_id):
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
        cell = self.grid.gpi2cell(location_id)
        filename = os.path.join(self.path, self.fn_format.format(cell))

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

        if self._open(location_id):
            data = self.fid.read(location_id, **kwargs)

        return data

    def read(self, *args, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it
        """
        if len(args) == 1:
            data = self._read_location_id(args[0], **kwargs)
        if len(args) == 2:
            data = self._read_lonlat(args[0], args[1], **kwargs)
        if len(args) < 1 or len(args) > 2:
            raise ValueError(f"Wrong number of arguments: {len(args)}")

        return data

    def iter_locations(self, **kwargs):
        """
        Yield all values for all locations.

        Yields
        ------
        data : numpy.ndarray
            Data
        location_id : int
            Location identifier.
        """
        if "ll_bbox" in kwargs:
            latmin, latmax, lonmin, lonmax = kwargs["ll_bbox"]
            location_ids = self.grid.get_bbox_grid_points(
                latmin, latmax, lonmin, lonmax
            )
            kwargs.pop("ll_bbox", None)
        elif "gpis" in kwargs:
            subgrid = self.grid.subgrid_from_gpis(kwargs["gpis"])
            gp_info = list(subgrid.grid_points())
            location_ids = np.array(gp_info, dtype=np.int32)[:, 0]
            kwargs.pop("gpis", None)
        else:
            gp_info = list(self.grid.grid_points())
            location_ids = np.array(gp_info, dtype=np.int32)[:, 0]

        for location_id in location_ids:
            try:
                data = self._read_location_id(location_id, **kwargs)
            except IOError as e:
                msg = f"I/O error({e.errno}): {e.strerror}, {location_id}"
                warnings.warn(msg, RuntimeWarning)
                data = None

            yield data, location_id

    def flush(self):
        """
        Flush data.
        """
        if self.fid is not None:
            self.fid.flush()

    def close(self):
        """
        Close file.
        """
        if self.fid is not None:
            self.fid.close()
            self.fid = None
