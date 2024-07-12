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
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from pyresample import kd_tree
from pyresample.geometry import AreaDefinition
from pyresample.geometry import SwathDefinition

from ascat.file_handling import ChronFiles
from ascat.read_native.product_info import grid_cache
from ascat.read_native.product_info import swath_io_catalog
from ascat.utils import get_grid_gpis


class SwathFile:
    """
    Class to read and merge swath files.
    """
    def __init__(self, filename, chunks=1_000_000):
        self.filename = filename
        self.chunks = chunks
        self.ds = None

    def read(self, date_range=None, valid_gpis=None, lookup_vector=None, mask_and_scale=True):
        """
        Read the file or a subset of it.
        """
        ds = xr.open_dataset(
            self.filename,
            mask_and_scale=mask_and_scale,
            engine="netcdf4",
        )
        ds["location_id"] = ds["location_id"].astype(np.int32)
        if date_range is not None:
            ds = self._trim_var_range(ds, "time", *date_range)
        if lookup_vector is not None:
            ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)
        elif valid_gpis is not None:
            ds = self._trim_to_gpis(ds, gpis=valid_gpis)
        # ds = self._ensure_obs(ds)

        ds = ds.chunk({"obs": self.chunks})

        # should I do it this way or just return the ds without having it be a class attribute?
        self.ds = ds
        return self.ds

    def merge(self, data):
        """
        Merge datasets with different locations dimensions.

        Parameters
        ----------
        data : list of xarray.Dataset
            Datasets to merge.

        Returns
        -------
        xarray.Dataset
            Merged dataset.
        """
        if data == []:
            return None

        # [self._preprocess(ds) for ds in data if ds.obs.size > 0]
        # merged_ds = xr.concat(
        #     [self._preprocess(ds) for ds in data if ds.obs.size > 0],
        #     dim="obs",
        #     combine_attrs=self.combine_attributes,
        # )
        ds_to_merge = [self._preprocess(ds) for ds in data if ds.obs.size > 0]

        # if all the datasets are empty in the obs dimension, just return the first one
        if ds_to_merge == []:
            return data[0]

        merged_ds = xr.concat(
            ds_to_merge,
            dim="obs",
            combine_attrs=self.combine_attributes,
            data_vars="minimal",
            coords="minimal",
        )

        return merged_ds

    @staticmethod
    def _preprocess(ds):
        """Pre-processing to be done on a component dataset so it can be merged with others.

        Assumes `ds` is an indexed ragged array. (Re)-calculates the `locationIndex`
        values for `ds` with respect to the `location_id` variable for the merged
        dataset, which may include locations not present in `ds`.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset with pre-processing applied.
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
    def _ensure_obs(ds):
        # basic heuristic - if obs isn't present, assume it's instead "time"
        if "obs" not in ds.dims:
            ds = ds.rename_dims({"time": "obs"})
        # other possible heuristics:
        # - if neither "obs" nor "time" is present, assume the obs dim is the one that's
        #  not "locations".
        return ds

    @staticmethod
    def _trim_var_range(ds, var_name, var_min, var_max, end_inclusive=False):
        # if var_name in ds:
        if end_inclusive:
            mask = (ds[var_name] >= var_min) & (ds[var_name] <= var_max)
        else:
            mask = (ds[var_name] >= var_min) & (ds[var_name] < var_max)
        return ds.sel(obs=mask.compute())

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
            ds_location_ids = ds["location_id"].data
            obs_idx = lookup_vector[ds_location_ids]
            ds = ds.sel(obs=obs_idx)

        # TODO Need to add this case!!!
        # else:
        #     ds = ds.sel(obs=(da.isin(ds["location_id"], gpis).compute())) #
            # # first trim out any gpis not in the dataset from the gpi list
            # gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

            # # this is a list of the locationIndex values that correspond to the gpis we're keeping
            # locations_idx = np.searchsorted(ds["location_id"].values, gpis)
            # # this is the indices of the observations that have any of those locationIndex values
            # obs_idx = da.isin(ds["locationIndex"], locations_idx).compute()

            # # now we need to figure out what the new locationIndex vector will be once we drop all the other location_ids
            # old_locationIndex = ds["locationIndex"].values
            # new_locationIndex = np.searchsorted(
            #     locations_idx,
            #     old_locationIndex[da.isin(old_locationIndex, locations_idx)]
            # )

            # # then trim out any gpis in the dataset not in gpis
            # ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
            # # and add the new locationIndex
            # ds["locationIndex"] = ("obs", new_locationIndex)

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

    def _close(self):
        """
        Close the file.
        """
        if self.ds is not None:
            self.ds.close()

    def __enter__(self):
        """
        Context manager initialization.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object. The file will be
        closed. The parameters describe the exception that caused the
        context to be exited.
        """
        self._close()


class SwathGridFiles(ChronFiles):
    """
    Class to read and merge multiple swath files.

    TODO
    ----
    - Override all existing methods to make sense, even if they're not
        very useful in this case.
    """

    def __init__(
        self,
        root_path,
        file_class,
        fn_templ,
        sf_templ,
        grid_name,
        date_field_fmt,
        cls_kwargs=None,
        err=True,
        fn_read_fmt=None,
        sf_read_fmt=None,
        fn_write_fmt=None,
        sf_write_fmt=None,
        cache_size=0,
    ):
        """
        Initialize SwathFiles class.

        Parameters
        ----------
        root_path : str
            Root path.
        file_class : class
            Class reading/writing files.
        fn_templ : str
            Filename template (e.g. "{date}_ascat.nc").
        sf_templ : dict, optional
            Subfolder template defined as dictionary (default: None).
        cls_kwargs : dict, optional
            Class keyword arguments (default: None).
        err : bool, optional
            Set true if a file error should be re-raised instead of
            reporting a warning.
            Default: False
        fn_read_fmt : str or function, optional
            Filename format for read operation.
        sf_read_fmt : str or function, optional
            Subfolder format for read operation.
        fn_write_fmt : str or function, optional
            Filename format for write operation.
        sf_write_fmt : str or function, optional
            Subfolder format for write operation.
        cache_size : int, optional
            Number of files to keep in memory (default=0).
        """
        # first check if any files directly under root_path contain the ending (make
        # sure not to iterate through every file - just stop after the first one).
        # This allows the user to set the root path either at the place necessitated by
        # the sf_templ or directly at the level of the files. However, the user still
        # cannot set the root path anywhere else in the directory structure (e.g. within
        # a satellite but above a year). In order to choose a specific satellite, must
        # pass that as a fmt_kwarg
        ending = fn_templ.split(".")[-1]
        for f in Path(root_path).glob(f"*.{ending}"):
            if f.is_file():
                sf_templ = None
                sf_read_fmt = None
                break

        super().__init__(root_path, file_class, fn_templ, sf_templ, cls_kwargs, err,
                         fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt,
                         cache_size)

        self.date_field_fmt = date_field_fmt
        grid_info = grid_cache.fetch_or_store(grid_name)
        self.grid_name = grid_name
        self.grid = grid_info["grid"]
        if "grid_sampling_km" in grid_info["attrs"]:
            self.grid_sampling_km = grid_info["attrs"]["grid_sampling_km"]
        else:
            self.grid_sampling_km = None

    @classmethod
    def from_product_id(
            cls,
            path,
            product_id,
    ):
        """Create a SwathGridFiles object based on a product_id.

        Returns a SwathGridFiles object initialized with an io_class specified
        by `product_id` (case-insensitive).

        Parameters
        ----------
        path : str or Path
            Path to the swath file collection.
        product_id : str
            Identifier for the specific ASCAT product the swath files are part of.

        Raises
        ------
        ValueError
            If product_id is not recognized.

        Examples
        --------
        >>> my_swath_collection = SwathFileCollection.from_product_id(
        ...     "/path/to/swath/files",
        ...     "H129",
        ... )

        """
        product_id = product_id.upper()
        if product_id in swath_io_catalog:
            io_class = swath_io_catalog[product_id]
        else:
            error_str = f"Product {product_id} not recognized. Valid products are"
            error_str += f" {', '.join(swath_io_catalog.keys())}."
            raise ValueError(error_str)

        return cls(
            path,
            SwathFile,
            io_class.fn_pattern,
            io_class.sf_pattern,
            # grid=io_class.grid,
            # grid_sampling_km=io_class.grid_sampling_km,
            grid_name=io_class.grid_name,
            date_field_fmt=io_class.date_field_fmt,
            fn_read_fmt=io_class.fn_read_fmt,
            sf_read_fmt=io_class.sf_read_fmt,
            # fn_write_fmt=io_class.fn_write_fmt,
            # sf_write_fmt=io_class.sf_write_fmt,
            # cache_size=io_class.cache_size,
        )

    @classmethod
    def from_io_class(
            cls,
            path,
            io_class,
    ):
        """Create a SwathGridFiles from a given io_class.

        Returns a SwathGridFiles object initialized with the given io_class.

        Parameters
        ----------
        path : str or Path
            Path to the swath file collection.
        io_class : class
            Class to use for reading and writing the swath files.

        Examples
        --------
        >>> my_swath_collection = SwathFileCollection.from_io_class(
        ...     "/path/to/swath/files",
        ...     AscatH129Swath,
        ... )

        """
        # enforce the presence of certain class attributes
        return cls(
            path,
            SwathFile,
            io_class.fn_pattern,
            io_class.sf_pattern,
            # grid=io_class.grid,
            # grid_sampling_km=io_class.grid_sampling_km,
            grid_name=io_class.grid_name,
            date_field_fmt=io_class.date_field_fmt,
            fn_read_fmt=io_class.fn_read_fmt,
            sf_read_fmt=io_class.sf_read_fmt,
            # fn_write_fmt=io_class.fn_write_fmt,
            # sf_write_fmt=io_class.sf_write_fmt,
            # cache_size=io_class.cache_size,
        )

    def _spatial_filter(
            self,
            filenames,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            # geom=None,
            # mask_and_scale=True,
            # date_range=None,
            # **kwargs,
            # timestamp,
            # search_date_fmt="%Y%m%d*",
            # date_field="date",
            # date_field_fmt="%Y%m%d",
            # return_date=False
    ):
        """
        Filter a search result for cells matching a spatial criterion.

        Parameters
        ----------
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.

        Returns
        -------
        filenames : list of str
            Filenames.
        """

        if cell is not None:
            gpis = get_grid_gpis(self.grid, cell=cell)
            spatial = SwathDefinition(
                lats=self.grid.arrlat[gpis],
                lons=self.grid.arrlon[gpis],
            )
        elif location_id is not None:
            gpis = get_grid_gpis(self.grid, location_id=location_id)
            spatial = SwathDefinition(
                lats=self.grid.arrlat[gpis],
                lons=self.grid.arrlon[gpis],
            )
        elif coords is not None:
            spatial = SwathDefinition(
                lats=[coords[1]],
                lons=[coords[0]],
            )
        elif bbox is not None:
            bbox = (bbox[2], bbox[0], bbox[3], bbox[1])
            spatial = AreaDefinition(
                "bbox",
                "",
                "EPSG:4326",
                {"proj": "latlong", "datum": "WGS84"},
                1000,
                1000,
                bbox,
            )
        else:
            spatial = None

        if spatial is None:
            return filenames

        filtered_filenames = []
        for filename in filenames:
            lazy_result = dask.delayed(self._check_intersection)(filename, spatial)
            filtered_filenames.append(lazy_result)

        def none_filter(fname_list):
            return [l for l in fname_list if l is not None]

        filtered_filenames = dask.delayed(none_filter)(filtered_filenames).compute()

        return filtered_filenames

    def _check_intersection(self, filename, spatial):
        """
        Check if a file intersects with a pyresample SwathDefinition or AreaDefinition.

        Parameters
        ----------
        filename : str
            Filename.
        gpis : list of int
            List of gpis.

        Returns
        -------
        bool
            True if the file intersects with the gpis.
        """
        with self.cls(filename) as f:
            f.read()
            lons, lats = f.ds["longitude"].values, f.ds["latitude"].values
            swath_def = SwathDefinition(lats=lats, lons=lons)
            n_info = kd_tree.get_neighbour_info(
                swath_def,
                spatial,
                radius_of_influence=15000,
                neighbours=1,
            )
            valid_input_index, _, _ = n_info[:3]
            if np.any(valid_input_index):
                return filename
        return None

    def swath_search(
        self,
        dt_start,
        dt_end,
        dt_delta=None,
        search_date_fmt="%Y%m%d*",
        date_field="date",
        end_inclusive=True,
        cell=None,
        location_id=None,
        coords=None,
        bbox=None,
        **fmt_kwargs,
    ):
        """
        Search for swath files within a time range and spatial criterion.

        Parameters
        ----------
        dt_start : datetime
            Start date.
        dt_end : datetime
            End date.
        dt_delta : timedelta
            Time delta.
        search_date_fmt : str
            Search date format.
        date_field : str
            Date field.
        end_inclusive : bool
            End date inclusive.
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.

        Returns
        -------
        list of str
            Filenames.
        """
        dt_delta = dt_delta or timedelta(days=1)

        filenames = self.search_period(
            dt_start,
            dt_end,
            dt_delta,
            search_date_fmt,
            date_field,
            date_field_fmt=self.date_field_fmt,
            end_inclusive=end_inclusive,
            **fmt_kwargs,
        )

        filtered_filenames = self._spatial_filter(
            filenames,
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
        )

        return filtered_filenames

    def extract(
        self,
        dt_start,
        dt_end,
        dt_delta=None,
        search_date_fmt="%Y%m%d*",
        date_field="date",
        end_inclusive=True,
        cell=None,
        location_id=None,
        coords=None,
        bbox=None,
        processes=None,
        **fmt_kwargs,
    ):
        """
        Extract data from swath files within a time range and spatial criterion.

        Parameters
        ----------
        dt_start : datetime
            Start date.
        dt_end : datetime
            End date.
        dt_delta : timedelta
            Time delta.
        search_date_fmt : str
            Search date format.
        date_field : str
            Date field.
        end_inclusive : bool
            End date inclusive.
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.

        Returns
        -------
        xarray.Dataset
            Dataset.
        """
        filenames = self.swath_search(
            dt_start, dt_end, dt_delta, search_date_fmt, date_field,
            end_inclusive, cell, location_id, coords, bbox, **fmt_kwargs,
        )
        valid_gpis = get_grid_gpis(
            self.grid,
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox
        )
        lookup_vector = np.zeros(self.grid.gpis.max()+1, dtype=bool)
        lookup_vector[valid_gpis] = 1

        data = []
        for filename in filenames:
            self._open(filename)
            date_range = (np.datetime64(dt_start), np.datetime64(dt_end))

            # filter by date here or below?
            d = self.fid.read(
                # date_range=date_range,
            )
            if d is not None:
                data.append(d)

        if data:
            data = self._merge_data(data)

            data_location_ids = data["location_id"].values
            obs_idx = lookup_vector[data_location_ids]
            data = data.sel(obs=obs_idx)

            # still not clear if it's better to filter date here or during fid.read
            if date_range is not None:
                mask = (data["time"] >= date_range[0]) & (data["time"] <= date_range[1])
                data = data.sel(obs=mask.compute())

            data.attrs["grid_name"] = self.grid_name

            return data
        return None
