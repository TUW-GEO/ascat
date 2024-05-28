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

import xarray as xr
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygeogrids

from ascat.utils import get_grid_gpis
from ascat.read_native.product_info import grid_cache


@xr.register_dataset_accessor("swath")
class SwathAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        # add a grid based on product information
        try:
            self.grid = grid_cache.fetch_or_store(xarray_obj.attrs["grid_name"])["grid"]
            assert isinstance(self.grid, pygeogrids.BasicGrid)
        except KeyError as exc:
            raise ValueError(
                "A grid_name which has been registered in the grid_cache"
                "must exist in the dataset attributes."
            ) from exc

    def sel_temporal(
            self,
            time_min,
            time_max,
            end_inclusive=False,
    ):
        ds = self._obj
        if end_inclusive:
            mask = (ds["time"] >= time_min) & (ds["time"] <= time_max)
        else:
            mask = (ds["time"] >= time_min) & (ds["time"] < time_max)
        ds = ds.sel(obs=mask.values)
        return ds

    def sel_spatial(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            max_coord_dist=np.Inf,
            # k_nearest_coords=1
    ):
        """
        Select data based on spatial criteria.

        Parameters
        ----------
        cell : int or list of int
            The cell number(s) to select.
        location_id : int or list of int
            The location ID(s) to select.
        coords : tuple of float
            The coordinates to select.
        bbox : tuple of float
            The bounding box to select.
        max_coord_dist : float
            The maximum distance a coordinate's nearest grid point can be from it to be
            selected.
        """
        ds = self._obj
        _, lookup_vector = get_grid_gpis(
            self.grid,
            cell,
            location_id,
            coords,
            bbox,
            max_coord_dist,
            return_lookup=True
        )
        ds = ds.sel(obs=lookup_vector[ds["location_id"].values])

        return ds

    def plot(self, var_name):
        """Plot the desired variable on a map.

        Parameters
        ----------
        var_name : str
            The name of the variable to plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """

        ds = self._obj
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        fig = plt.scatter(ds["longitude"], ds["latitude"], c=ds[var_name])

        return fig, ax


@xr.register_dataset_accessor("idx_ragged")
class IndexedRaggedAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        try:
            self.grid = grid_cache.fetch_or_store(xarray_obj.attrs["grid_name"])["grid"]
            assert isinstance(self.grid, pygeogrids.BasicGrid)
        except KeyError as exc:
            raise ValueError(
                "A grid_name which has been registered in the grid_cache"
                "must exist in the dataset attributes."
            ) from exc

    def sel_temporal(
            self,
            time_min,
            time_max,
            end_inclusive=False,
    ):
        ds = self._obj
        if end_inclusive:
            mask = (ds["time"] >= time_min) & (ds["time"] <= time_max)
        else:
            mask = (ds["time"] >= time_min) & (ds["time"] < time_max)
        ds = ds.sel(obs=mask.values)
        return ds

    def sel_spatial(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            max_coord_dist=np.Inf,
    ):
        """
        Select data based on spatial criteria without trimming the locations dimension.
        """
        ds = self._obj
        _, lookup_vector = get_grid_gpis(
            self.grid,
            cell,
            location_id,
            coords,
            bbox,
            max_coord_dist,
            return_lookup=True
        )
        ds = self._select_gpis(ds, lookup_vector=lookup_vector)

        return ds

    def extract_spatial(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            max_coord_dist=np.Inf,
    ):
        """
        Select data based on spatial criteria and trim the locations dimension to only
        include the selected locations.
        """
        ds = self._obj
        _, lookup_vector = get_grid_gpis(
            self.grid,
            cell,
            location_id,
            coords,
            bbox,
            max_coord_dist,
            return_lookup=True
        )
        ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)

        return ds

    def plot(self, var_name):
        """Plot the desired variable on a map.

        Parameters
        ----------
        var_name : str
            The name of the variable to plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """

        ds = self._obj
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        lons = ds["lon"][ds["locationIndex"]]
        lats = ds["lat"][ds["locationIndex"]]
        fig = plt.scatter(lons, lats, c=ds[var_name])

        return fig, ax

    @staticmethod
    def _trim_to_gpis(ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None
        if gpis is None and lookup_vector is None:
            return ds
        if gpis is None:
            ds_location_ids = ds["location_id"].data[ds["locationIndex"].data]
            obs_idx = lookup_vector[ds_location_ids]
            locations_idx = da.unique(ds["locationIndex"].data[obs_idx]).compute()

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
            new_locationIndex = np.searchsorted(ds["location_id"].data, ds_location_ids[obs_idx])
            # and add the new locationIndex
            ds["locationIndex"] = ("obs", new_locationIndex)

        else:
            # first trim out any gpis not in the dataset from the gpi list
            gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

            # this is a list of the locationIndex values that correspond to the gpis we're keeping
            locations_idx = np.searchsorted(ds["location_id"].values, gpis)
            # this is the indices of the observations that have any of those locationIndex values
            obs_idx = da.isin(ds["locationIndex"], locations_idx).compute()

            # now we need to figure out what the new locationIndex vector will be once we drop all the other location_ids
            old_locationIndex = ds["locationIndex"].values
            new_locationIndex = np.searchsorted(
                locations_idx,
                old_locationIndex[da.isin(old_locationIndex, locations_idx)]
            )

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
            # and add the new locationIndex
            ds["locationIndex"] = ("obs", new_locationIndex)

        return ds

    @staticmethod
    def _select_gpis(ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None
        if gpis is None and lookup_vector is None:
            return ds
        if gpis is None:
            ds_location_ids = ds["location_id"].data[ds["locationIndex"].data]
            obs_idx = lookup_vector[ds_location_ids]
            locations_idx = da.unique(ds["locationIndex"].data[obs_idx]).compute()

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx})

        else:
            # first trim out any gpis not in the dataset from the gpi list
            gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

            # this is a list of the locationIndex values that correspond to the gpis we're keeping
            locations_idx = np.searchsorted(ds["location_id"].values, gpis)
            # this is the indices of the observations that have any of those locationIndex values
            obs_idx = da.isin(ds["locationIndex"], locations_idx).compute()

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx})

        return ds


@xr.register_dataset_accessor("orthomulti")
class OrthoMultiAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        try:
            self.grid = grid_cache.fetch_or_store(xarray_obj.attrs["grid_name"])["grid"]
            assert isinstance(self.grid, pygeogrids.BasicGrid)
        except KeyError as exc:
            raise ValueError(
                "A grid_name which has been registered in the grid_cache"
                "must exist in the dataset attributes."
            ) from exc

    def sel_temporal(
            self,
            time_min,
            time_max,
            end_inclusive=False,
    ):
        ds = self._obj
        if end_inclusive:
            mask = (ds["time"] >= time_min) & (ds["time"] <= time_max)
        else:
            mask = (ds["time"] >= time_min) & (ds["time"] < time_max)
        ds = ds.sel(obs=mask.values)
        return ds

    def sel_spatial(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            max_coord_dist=np.Inf,
    ):
        """
        Select data based on spatial criteria without trimming the locations dimension.
        """
        ds = self._obj
        _, lookup_vector = get_grid_gpis(
            self.grid,
            cell,
            location_id,
            coords,
            bbox,
            max_coord_dist,
            return_lookup=True
        )
        ds = self._select_gpis(ds, lookup_vector=lookup_vector)

        return ds

    def extract_spatial(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            max_coord_dist=np.Inf,
    ):
        """
        Select data based on spatial criteria and trim the locations dimension to only
        include the selected locations.
        """
        ds = self._obj
        _, lookup_vector = get_grid_gpis(
            self.grid,
            cell,
            location_id,
            coords,
            bbox,
            max_coord_dist,
            return_lookup=True
        )
        ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)

        return ds

    def plot(self, var_name):
        """Plot the desired variable on a map.

        Parameters
        ----------
        var_name : str
            The name of the variable to plot.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """

        ds = self._obj
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        lons = ds["lon"][ds["locationIndex"]]
        lats = ds["lat"][ds["locationIndex"]]
        fig = plt.scatter(lons, lats, c=ds[var_name])

        return fig, ax

    @staticmethod
    def _trim_to_gpis(ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None
        if gpis is None and lookup_vector is None:
            return ds
        if gpis is None:
            ds_location_ids = ds["location_id"].data[ds["locationIndex"].data]
            obs_idx = lookup_vector[ds_location_ids]
            locations_idx = da.unique(ds["locationIndex"].data[obs_idx]).compute()

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
            new_locationIndex = np.searchsorted(ds["location_id"].data, ds_location_ids[obs_idx])
            # and add the new locationIndex
            ds["locationIndex"] = ("obs", new_locationIndex)

        else:
            # first trim out any gpis not in the dataset from the gpi list
            gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

            # this is a list of the locationIndex values that correspond to the gpis we're keeping
            locations_idx = np.searchsorted(ds["location_id"].values, gpis)
            # this is the indices of the observations that have any of those locationIndex values
            obs_idx = da.isin(ds["locationIndex"], locations_idx).compute()

            # now we need to figure out what the new locationIndex vector will be once we drop all the other location_ids
            old_locationIndex = ds["locationIndex"].values
            new_locationIndex = np.searchsorted(
                locations_idx,
                old_locationIndex[da.isin(old_locationIndex, locations_idx)]
            )

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
            # and add the new locationIndex
            ds["locationIndex"] = ("obs", new_locationIndex)

        return ds

    @staticmethod
    def _select_gpis(ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None
        if gpis is None and lookup_vector is None:
            return ds
        if gpis is None:
            ds_location_ids = ds["location_id"].data[ds["locationIndex"].data]
            obs_idx = lookup_vector[ds_location_ids]
            locations_idx = da.unique(ds["locationIndex"].data[obs_idx]).compute()

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx})

        else:
            # first trim out any gpis not in the dataset from the gpi list
            gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

            # this is a list of the locationIndex values that correspond to the gpis we're keeping
            locations_idx = np.searchsorted(ds["location_id"].values, gpis)
            # this is the indices of the observations that have any of those locationIndex values
            obs_idx = da.isin(ds["locationIndex"], locations_idx).compute()

            # then trim out any gpis in the dataset not in gpis
            ds = ds.isel({"obs": obs_idx})

        return ds
