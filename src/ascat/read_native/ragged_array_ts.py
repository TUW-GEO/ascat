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

import xarray as xr
import numpy as np

NC_FILL_FLOAT = np.float32(9969209968386869046778552952102584320)


def var_order(dataset):
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

    def read_compat(self, grid):
        with xr.open_dataset(
            self.filename, mask_and_scale=self.mask_and_scale
        ) as dataset:
            cell = int(dataset.attrs["id"].split(".")[0])
            grid_gpis, grid_lons, grid_lats = grid.grid_points_for_cell(cell)

            # alt and location_description generally are completely filled with
            # NaN values/empty strings, and there is no way to derive them from
            # the gpis yet, but in case they do have data for some reason, this
            # maintains the existing data in alignment with the relevant
            # location_ids and fills in the rest with NaNs/empty strings.
            #
            # NC_FILL_FLOAT is the NaN used for alt, since that's what's used by
            # the previous writers/mergers, but it's against CF conventions to
            # have fill values in dataset coordinates, so I'm not sure if this
            # is appropriate
            alt = dataset[self.var["alt"]].values
            alt = np.append(alt, np.repeat(NC_FILL_FLOAT, (len(grid_gpis) - len(alt))))
            location_description = dataset[self.loc["descr"]].values
            location_description = np.append(
                location_description,
                np.repeat("", len(grid_gpis) - len(location_description)),
            )

            # get the array of location ids from dataset and append any missing location
            # ids from the same cell to the end of the array
            location_id = dataset[self.loc["ids"]].values[
                np.asarray(dataset[self.loc["ids"]].values > 0).nonzero()
            ]
            location_id = np.concatenate(
                (location_id, np.setdiff1d(grid_gpis, location_id))
            )

            # calculate what locationIndex will be after sorting the location_ids
            sorter = np.argsort(np.sort(location_id))
            locationIndex = sorter[
                np.searchsorted(
                    np.sort(location_id),
                    location_id[dataset.locationIndex.values],
                    sorter=sorter,
                )
            ]

            # Need to drop locations dim and associated variables in order to
            # add new versions of those variables with more entries.
            dataset = dataset.drop_dims([self.dim["loc"]])

            # Add back vars with locations dimension, which will then be sorted along
            # with location_id
            dataset[self.loc["ids"]] = xr.DataArray(location_id, dims=[self.dim["loc"]])
            dataset[self.var["alt"]] = xr.DataArray(alt, dims=[self.dim["loc"]])
            dataset[self.loc["descr"]] = xr.DataArray(
                location_description, dims=[self.dim["loc"]]
            )
            dataset = dataset.sortby(self.loc["ids"])
            assert np.all(dataset.location_id.values == np.array(grid_gpis))

            # Add back the lons and lats, which were already properly sorted when
            # retrieved from the grid
            dataset[self.var["lon"]] = xr.DataArray(grid_lons, dims=[self.dim["loc"]])
            dataset[self.var["lat"]] = xr.DataArray(grid_lats, dims=[self.dim["loc"]])

            # Add the locationIndex
            dataset["locationIndex"] = xr.DataArray(
                locationIndex, dims=[self.dim["obs"]]
            )

            dataset = dataset.set_coords(
                [self.var["lon"], self.var["lat"], self.var["alt"], self.var["time"]]
            )

            # Make sure the obs dimension is named obs
            dataset = dataset.rename_dims({self.dim["obs"]: "obs"})
            dataset = var_order(dataset)

        return dataset


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

    def read_compat(self, grid):
        with xr.open_dataset(
            self.filename, mask_and_scale=self.mask_and_scale
        ) as dataset:
            cell = int(dataset.attrs["id"].split(".")[0])
            grid_gpis, grid_lons, grid_lats = grid.grid_points_for_cell(cell)
            # alt = dataset[self.var["alt"]].values
            # alt = np.append(alt, [NC_FILL_FLOAT]*(len(grid_gpis) - len(alt)))
            # replace nans in row_size with zeros
            row_size = np.where(
                dataset[self.var["row"]].values > 0, dataset[self.var["row"]].values, 0
            )

            # get the array of location ids from dataset and append any missing location
            # ids from the same cell to the end of the array
            location_id = dataset[self.loc["ids"]].values[
                np.asarray(dataset[self.loc["ids"]].values > 0).nonzero()
            ]
            location_id = np.concatenate(
                (location_id, np.setdiff1d(grid_gpis, location_id))
            )

            # calculate locationIndex for the unsorted dataset
            locationIndex = np.repeat(np.arange(len(row_size)), row_size)

            # calculate what locationIndex will be after sorting the location_ids
            sorter = np.argsort(np.sort(location_id))
            locationIndex = sorter[
                np.searchsorted(
                    np.sort(location_id),
                    location_id[locationIndex],
                    sorter=sorter,
                )
            ]

            dataset = dataset.drop_vars(["row_size"])
            dataset[self.loc["ids"]] = xr.DataArray(location_id, dims=[self.dim["loc"]])
            dataset["locationIndex"] = xr.DataArray(locationIndex, dims=["obs"])

            # sort all vars with locations dim by location_ids
            dataset = dataset.sortby(self.loc["ids"])
            assert np.all(dataset.location_id.values == np.array(grid_gpis))
            dataset[self.var["lon"]] = xr.DataArray(grid_lons, dims=[self.dim["loc"]])
            dataset[self.var["lat"]] = xr.DataArray(grid_lats, dims=[self.dim["loc"]])

            dataset = var_order(dataset)

        return dataset


default_attrs = {
    "row_size": {
        "long_name": "number of observations at this location",
        "sample_dimension": "obs",
    },
    "lon": {
        "standard_name": "longitude",
        "long_name": "location longitude",
        "units": "degrees_east",
        "valid_range": np.array([-180, 180], dtype=np.float),
    },
    "lat": {
        "standard_name": "latitude",
        "long_name": "location latitude",
        "units": "degrees_north",
        "valid_range": np.array([-90, 90], dtype=np.float),
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
}


default_encoding = {
    "row_size": {
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
    "location_description": {
        "dtype": "string",
    },
    "time": {
        "dtype": "float64",
        "units": "days since 1900-01-01 00:00:00",
        "calendar": None,
    },
}

for var in default_encoding:
    default_encoding[var]["_FillValue"] = None
    default_encoding[var]["zlib"] = True
    default_encoding[var]["complevel"] = 4


def compatible_write(compat_dataset, output_name, attributes=None, encoding=None):
    """
    Write a compatible dataset in contiguous ragged array format.
    The output from IRANcFile.read_compat() and CRANcFile.read_compat()
    are compatible, as well as any dataset produced as a concatenation of these two
    along the obs dimension with `data_vars="minimal"`.
    """
    if attributes is None:
        attributes = {}
    if encoding is None:
        encoding = {}

    compat_dataset = compat_dataset.sortby(["locationIndex", "time"])

    idxs, sizes = np.unique(compat_dataset.locationIndex, return_counts=True)
    row_size = np.zeros_like(compat_dataset.location_id.values)
    row_size[idxs] = sizes
    compat_dataset["row_size"] = xr.DataArray(row_size, dims=["locations"])

    compat_dataset = compat_dataset.drop_vars(["locationIndex"])
    compat_dataset.attrs["date_created"] = str(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    encoding = {**default_encoding, **encoding}
    attributes = {**default_attrs, **attributes}
    if "locationIndex" in encoding:
        encoding.pop("locationIndex")
    if "locationIndex" in attributes:
        attributes.pop("locationIndex")

    for var, attrs in attributes.items():
        compat_dataset[var] = compat_dataset[var].assign_attrs(attrs)
        if var in ["row_size", "location_id", "location_description"]:
            compat_dataset[var].encoding["coordinates"] = None

    var_order(compat_dataset).to_netcdf(output_name, encoding=encoding)


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
