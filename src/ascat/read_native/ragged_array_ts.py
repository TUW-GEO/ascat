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

import xarray as xr
import numpy as np


class RAFile:
    """
    Ragged array representation

    """

    def __init__(self,
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
                 mask_and_scale=False):

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
            "alt": alt_var
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
        """
        super().__init__(**kwargs)
        self.var["row"] = row_var
        self.filename = filename

        with xr.open_dataset(self.filename,
                             mask_and_scale=self.mask_and_scale) as ncfile:

            var_list = [
                self.var["lon"], self.var["lat"], self.loc["ids"],
                self.var["row"]
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

class GridCellFiles:

    def __init__(self,
                 path,
                 grid,
                 ioclass,
                 cache=False,
                 fn_format="{:04d}.nc",
                 ioclass_kws=None):

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
                latmin, latmax, lonmin, lonmax)
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
