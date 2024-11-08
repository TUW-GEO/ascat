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

import multiprocessing as mp

from datetime import timedelta
from functools import partial
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr

from pyresample import kd_tree
from pyresample.geometry import AreaDefinition
from pyresample.geometry import SwathDefinition

from ascat.read_native.grid_registry import grid_registry

from ascat.file_handling import Filenames
from ascat.file_handling import ChronFiles
from ascat.utils import get_grid_gpis
from ascat.utils import create_variable_encodings



class Swath(Filenames):
    """
    Class to read and merge swath files given one or more file paths.
    """
    # def __init__(self, filename, chunks=1_000_000):
    #     self.filename = filename
    #     self.chunks = chunks
    #     self.ds = None

    def _read(self, filename, generic=True, preprocessor=None, **xarray_kwargs):
        """
        Open one swath file as an xarray.Dataset and preprocess it if necessary.

        Parameters
        ----------
        filename : str
            File to read.
        generic : bool, optional
            Not yet implemented.
        preprocessor : callable, optional
            Function to preprocess the dataset.
        xarray_kwargs : dict
            Additional keyword arguments passed to xarray.open_dataset.

        Returns
        -------
        ds : xarray.Dataset
            Dataset.
        """
        ds = xr.open_dataset(
            filename,
            engine="netcdf4",
            **xarray_kwargs,
        )
        ds["location_id"] = ds["location_id"].astype(np.int32)

        return ds

    def read(
        self,
        mask_and_scale=True,
        max_nbytes=None,
        parallel=False,
        **kwargs
    ):
        """
        Read the file or a subset of it.
        """

        ds, closers = super().read(closer_attr="_close",
                                   parallel=parallel,
                                   mask_and_scale=mask_and_scale,
                                   **kwargs)
        if ds is not None:
            ds.set_close(partial(super()._multi_file_closer, closers))
            return ds

    @staticmethod
    def _nbytes(ds):
        return ds.nbytes

    def _merge(self, data):
        """
        Merge datasets.

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

        merged_ds = xr.concat(
            [ds for ds in data if ds is not None],
            dim="obs",
            combine_attrs=self.combine_attributes,
            data_vars="minimal",
            coords="minimal",
        )

        return merged_ds

    @staticmethod
    def _ensure_obs(ds):
        # TODO make reg func.
        # basic heuristic - if obs isn't present, assume it's instead "time"
        if "obs" not in ds.dims:
            ds = ds.rename_dims({"time": "obs"})
        # other possible heuristics:
        # - if neither "obs" nor "time" is present, assume the obs dim is the one that's
        #  not "locations".
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

class SwathGridFiles(ChronFiles):
    """
    Class to manage chronological swath files with a date field in the filename.
    """
    def __init__(
        self,
        root_path,
        file_class,
        fn_templ,
        sf_templ,
        grid_name,
        date_field_fmt,
        cell_fn_format=None,
        beams_vars=None,
        ts_dtype=None,
        cls_kwargs=None,
        err=True,
        fn_read_fmt=None,
        sf_read_fmt=None,
        fn_write_fmt=None,
        sf_write_fmt=None,
        preprocessor=None,
        postprocessor=None,
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
        grid_info = grid_registry.get(grid_name)
        self.grid_name = grid_name
        self.grid = grid_info["grid"]
        if "grid_sampling_km" in grid_info["attrs"]:
            self.grid_sampling_km = grid_info["attrs"]["grid_sampling_km"]
        else:
            self.grid_sampling_km = None

        self.cell_fn_format = cell_fn_format
        self.beams_vars = beams_vars
        self.ts_dtype = ts_dtype
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

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
        from ascat.read_native.product_info import swath_io_catalog
        product_id = product_id.upper()
        if product_id in swath_io_catalog:
            product_class = swath_io_catalog[product_id]
        else:
            error_str = f"Product {product_id} not recognized. Valid products are"
            error_str += f" {', '.join(swath_io_catalog.keys())}."
            raise ValueError(error_str)

        return cls.from_product_class(path, product_class)

    @classmethod
    def from_product_class(
        cls,
        path,
        product_class,
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
        return cls(
            path,
            Swath,
            product_class.fn_pattern,
            product_class.sf_pattern,
            grid_name=product_class.grid_name,
            cell_fn_format=product_class.cell_fn_format,
            date_field_fmt=product_class.date_field_fmt,
            beams_vars=product_class.beams_vars,
            ts_dtype=product_class.ts_dtype,
            fn_read_fmt=product_class.fn_read_fmt,
            sf_read_fmt=product_class.sf_read_fmt,

            # fn_write_fmt=io_class.fn_write_fmt,
            # sf_write_fmt=io_class.sf_write_fmt,
        )


    def _spatial_filter(
            self,
            filenames,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
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
        elif (bbox or geom) is not None:
            if bbox is not None:
                # AreaDefinition expects (lonmin, latmin, lonmax, latmax)
                # but bbox is (latmin, latmax, lonmin, lonmax)
                bbox = (bbox[2], bbox[0], bbox[3], bbox[1])
            else:
                # If we get a geometry just take its bounding box and check
                # that intersection.
                #
                # shapely.geometry.bounds is already in the correct order
                bbox = geom.bounds
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

    @staticmethod
    def _trim_var_range(ds, var_name, var_min, var_max, end_inclusive=False):
        # TODO make reg func.
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
        else:
            # this is a case where I really do want the answer of `isin` to be fully
            # loaded no matter what, so I use dask.array's version of isin to make sure
            # I can call compute on it.
            ds = ds.sel(obs=(da.isin(ds["location_id"], gpis).compute()))

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
        f = self.cls(filename)
        ds = f.read()
        lons, lats = ds["longitude"].values, ds["latitude"].values
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
        geom=None,
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
        geom : shapely.geometry
            Geometry.

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
            geom=geom,
        )

        return filtered_filenames

    def extract(
        self,
        date_range,
        dt_delta=None,
        search_date_fmt="%Y%m%d*",
        date_field="date",
        end_inclusive=True,
        cell=None,
        location_id=None,
        coords=None,
        bbox=None,
        geom=None,
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
        dt_start, dt_end = date_range
        filenames = self.swath_search(
            dt_start, dt_end, dt_delta, search_date_fmt, date_field,
            end_inclusive, cell, location_id, coords, bbox, geom, **fmt_kwargs,
        )

        date_range = (np.datetime64(dt_start), np.datetime64(dt_end))
        data = self.cls(filenames).read()

        if data:
            if any(v is not None for v in (cell, location_id, coords, bbox, geom)):
                valid_gpis = get_grid_gpis(
                    self.grid,
                    cell=cell,
                    location_id=location_id,
                    coords=coords,
                    bbox=bbox,
                    geom=geom,
                )
                lookup_vector = np.zeros(self.grid.gpis.max()+1, dtype=bool)
                lookup_vector[valid_gpis] = 1

                data_location_ids = data["location_id"].values
                obs_idx = lookup_vector[data_location_ids]
                data = data.sel(obs=obs_idx)

            if date_range is not None:
                mask = (data["time"] >= date_range[0]) & (data["time"] <= date_range[1])
                data = data.sel(obs=mask.compute())

            data.attrs["grid_name"] = self.grid_name

            return data

    def stack_to_cell_files(self,
                            out_dir,
                            max_nbytes,
                            n_processes,
                            date_range=None,
                            fmt_kwargs=None,
                            cells=None):
        """
        Stack all swath files to cell files, writing them in parallel.
        """
        from ascat.read_native.cell_collection import RaggedArrayCell

        fmt_kwargs = fmt_kwargs or {}
        if date_range is not None:
            dt_start, dt_end = date_range
            filenames = self.swath_search(dt_start, dt_end, cell=cells, **fmt_kwargs)
        else:
            filenames = list(Path(self.root_path).glob("**/*.nc"))

        swath = self.cls(filenames)

        for ds in swath.iter_read_nbytes(max_nbytes):
            ds_cells = self.grid.gpi2cell(ds["location_id"]).compressed()
            ds_cells = xr.DataArray(ds_cells, dims="obs", name="cell")

            # sorting here enables us to manually select each cell's data much faster
            # than using a .groupby
            ds = ds.sortby(ds_cells)

            unique_cells, cell_counts = np.unique(ds_cells, return_counts=True)
            cell_counts = np.hstack([0, np.cumsum(cell_counts)])

            # for each cell in unique cells, isel the slice from the dataarray corresponding to it
            ds_list = []
            cell_fnames = []
            for i, c in enumerate(unique_cells):
                if c in cells:
                    cell_ds = ds.isel(obs=slice(cell_counts[i], cell_counts[i+1]))
                    if len(cell_ds) == 0:
                        continue
                    ds_list.append(cell_ds)
                    cell_fname = Path(out_dir)/self.cell_fn_format.format(c)
                    cell_fnames.append(cell_fname)

            writer_class = RaggedArrayCell(cell_fnames)
            writer_class.write(ds_list, parallel=True,
                                postprocessor=self.postprocessor, mode="a")


class SwathGridFilesOld(ChronFiles):
    """
    Class to read and merge multiple swath files.
    """

    def __init__(
        self,
        root_path,
        file_class,
        fn_templ,
        sf_templ,
        grid_name,
        date_field_fmt,
        cell_fn_format=None,
        beams_vars=None,
        ts_dtype=None,
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
        grid_info = grid_registry.get(grid_name)
        self.grid_name = grid_name
        self.grid = grid_info["grid"]
        if "grid_sampling_km" in grid_info["attrs"]:
            self.grid_sampling_km = grid_info["attrs"]["grid_sampling_km"]
        else:
            self.grid_sampling_km = None

        self.cell_fn_format = cell_fn_format
        self.beams_vars = beams_vars
        self.ts_dtype = ts_dtype

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
        from ascat.read_native.product_info import swath_io_catalog
        product_id = product_id.upper()
        if product_id in swath_io_catalog:
            io_class = swath_io_catalog[product_id]
        else:
            error_str = f"Product {product_id} not recognized. Valid products are"
            error_str += f" {', '.join(swath_io_catalog.keys())}."
            raise ValueError(error_str)

        return cls(
            path,
            Swath,
            io_class.fn_pattern,
            io_class.sf_pattern,
            grid_name=io_class.grid_name,
            date_field_fmt=io_class.date_field_fmt,
            fn_read_fmt=io_class.fn_read_fmt,
            sf_read_fmt=io_class.sf_read_fmt,
            beams_vars=io_class.beams_vars,
            ts_dtype=io_class.ts_dtype,
            # fn_write_fmt=io_class.fn_write_fmt,
            # sf_write_fmt=io_class.sf_write_fmt,
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
        return cls(
            path,
            SwathFile,
            io_class.fn_pattern,
            io_class.sf_pattern,
            grid_name=io_class.grid_name,
            date_field_fmt=io_class.date_field_fmt,
            beams_vars=io_class.beams_vars,
            ts_dtype=io_class.ts_dtype,
            fn_read_fmt=io_class.fn_read_fmt,
            sf_read_fmt=io_class.sf_read_fmt,
            # fn_write_fmt=io_class.fn_write_fmt,
            # sf_write_fmt=io_class.sf_write_fmt,
        )

    def _spatial_filter(
            self,
            filenames,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
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
        elif (bbox or geom) is not None:
            if bbox is not None:
                # AreaDefinition expects (lonmin, latmin, lonmax, latmax)
                # but bbox is (latmin, latmax, lonmin, lonmax)
                bbox = (bbox[2], bbox[0], bbox[3], bbox[1])
            else:
                # If we get a geometry just take its bounding box and check
                # that intersection.
                #
                # shapely.geometry.bounds is already in the correct order
                bbox = geom.bounds
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
        geom=None,
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
        geom : shapely.geometry
            Geometry.

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
            geom=geom,
        )

        return filtered_filenames

    def _cell_data_as_indexed_ra(self, ds, cell, dupe_window=None):
        """Convert swath data for a single cell to indexed format.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset.

        Returns
        -------
        ds : xarray.Dataset
            Output dataset.
        """
        dupe_window = dupe_window or np.timedelta64(10, "m")
        # location_id = self.grid
        ds = ds.drop_vars(["latitude", "longitude", "cell"], errors="ignore")
        location_id, lon, lat = self.grid.grid_points_for_cell(cell)
        sorter = np.argsort(location_id)
        locationIndex = sorter[np.searchsorted(location_id,
                                               ds["location_id"].values,
                                               sorter=sorter)]

        ds = ds.assign_coords({"lon": ("locations", lon),
                               "lat": ("locations", lat),
                               "alt": ("locations", np.repeat(np.nan,
                                                              len(location_id)))})

        ds = ds.set_coords(["time"])
        ds["location_id"] = ("locations", location_id)
        ds["location_description"] = ("locations", np.repeat("", len(location_id)))
        ds["locationIndex"] = ("obs", locationIndex)

        # set _FillValue to missing_value (which has been deprecated)
        for var in ds.variables:
            if "missing_value" in ds[var].attrs and "_FillValue" not in ds[var].attrs:
                ds[var].attrs["_FillValue"] = ds[var].attrs.get("missing_value")

        ds = ds.sortby(["locationIndex", "time"])

        # dedupe
        dupl = np.insert(
            (abs(ds["time"].values[1:] -
                 ds["time"].values[:-1]) < dupe_window),
            0,
            False,
        )
        ds = ds.sel(obs=~dupl)

        return ds

    def extract(
        self,
        # dt_start,
        # dt_end,
        date_range,
        dt_delta=None,
        search_date_fmt="%Y%m%d*",
        date_field="date",
        end_inclusive=True,
        cell=None,
        location_id=None,
        coords=None,
        bbox=None,
        geom=None,
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
        dt_start, dt_end = date_range
        filenames = self.swath_search(
            dt_start, dt_end, dt_delta, search_date_fmt, date_field,
            end_inclusive, cell, location_id, coords, bbox, geom, **fmt_kwargs,
        )
        valid_gpis = get_grid_gpis(
            self.grid,
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            geom=geom,
        )
        lookup_vector = np.zeros(self.grid.gpis.max()+1, dtype=bool)
        lookup_vector[valid_gpis] = 1

        date_range = (np.datetime64(dt_start), np.datetime64(dt_end))

        data = self._yield_data_from_files(filenames, processes=processes).__next__()

        if data:
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

    def _yield_data_from_files(self, filenames, max_size_mb=None, processes=None):
        """
        Yield data from swath files. If max_size_mb is set, yield data in chunks
        of at most max_size_mb in size.
        """
        data = []
        if max_size_mb is None:
            for filename in filenames:
                self._open(filename)

                d = self.fid.read()
                if d is not None:
                    data.append(d)
            if data:
                yield self._merge_data(data, processes=processes)

        else:
            d_size = 0
            for filename in filenames:
                self._open(filename)

                d = self.fid.read()
                if d is not None:
                    d_size += d.nbytes

                if d_size > max_size_mb * 1024**2:
                    if data:
                        yield self._merge_data(data, load=True, processes=processes)
                    data = []
                    d_size = d.nbytes
                data.append(d)
            if data:
                yield self._merge_data(data, load=True, processes=processes)

    def _write_cell(self, in_q):
        while True:
            item = in_q.get()
            if item is None:
                break
            cell, cell_obj, dupe_window = item
            cell_obj.ds = self._cell_data_as_indexed_ra(cell_obj.ds, cell, dupe_window)

            out_encoding = create_variable_encodings(
                cell_obj.ds, custom_dtypes=self.ts_dtype)
            cell_obj.write(mode="a", encoding=out_encoding)
            # cell_obj.write(mode="a")

    def _process_stack(self, ds):
        # if there are kwargs, use them instead of self.ioclass_kws
        # beam_idx = {"for": 0, "mid": 1, "aft": 2}

        # if any beam has backscatter ds for a record, the record is valid. Drop
        # observations that don't have any backscatter data.
        # if "backscatter" in ds.variables:
        #     if ds["obs"].size > 0:
        #         ds = ds.sel(
        #             obs=~np.all(
        #                 ds["backscatter"].isnull(),
        #                 axis=1
        #             )
        #         )

        # # break the beams dimension variables into separate variables for
        # # the fore, mid, and aft beams
        # if ds["obs"].size > 0:
        #     if "beams" in ds.dims:
        #         # if the dataset has a beams dimension and beams_vars is set,
        #         # break the beams dimension variables into separate variables
        #         # for the fore, mid, and aft beams and remove the beams dimension
        #         # along with any beams-dimensional variables that aren't in beams_vars
        #         if self.beams_vars:
        #             for var in self.beams_vars:
        #                 if var in ds.variables:
        #                     for ending, idx in beam_idx.items():
        #                         ds[var + "_" + ending] = ds[var].sel(beams=idx)

        #             # drop the variables on the beams dimension
        #             ds = ds.drop_dims("beams")

        ds = ds.assign_coords(
            {"cell": ("obs", self.grid.gpi2cell(ds["location_id"].values))}
        )
        return ds

    def stack_to_cell_files(
            self,
            out_dir,
            cell_class,
            # dt_start,
            # dt_end,
            date_range,
            dt_delta=None,
            search_date_fmt="%Y%m%d*",
            date_field="date",
            end_inclusive=True,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            fnames=None,
            mode="w",
            processes=1,
            buffer_memory_mb=None,
            dupe_window=None,
            **fmt_kwargs
        ):
        """
        Stack swath files into cell files.
        This first gets the relevant filenames, then creates one large xarray dataset
        out of them. The dataset is then split up into separate datasets according to cell.
        In separate processes, each cell dataset is converted to indexed ragged array format
        (function _cell_data_as_indexed_ra) and then written to disk (function _cell_to_disk).
        """

        # is there a better way to do this that doesn't break tests?
        if mode == "w":
            # ask user if they want to delete existing files
            # TODO probably could figure out something different for this.
            print("Using write mode ('w').")
            input_str = input(f"Overwrite all existing files in {out_dir}? (y/n) ")
        elif mode == "a":
            print("Using append mode ('a').")
            input_str = input(f"Append data to existing files in {out_dir}? (y/n) ")
        else:
            raise ValueError("Invalid mode. Valid modes are 'w' and 'a'.")
        if input_str.lower() == "n":
            print("Exiting.")
            return
        elif input_str.lower() != "y":
            raise ValueError("Invalid input. Please enter 'y' or 'n'.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode == "w":
            [f.unlink() for f in out_dir.glob("*.nc")]

        if fnames is None:
            fnames = self.swath_search(
                dt_start, dt_end, dt_delta, search_date_fmt, date_field,
                end_inclusive, cell, location_id, coords, bbox, **fmt_kwargs,
            )

        if processes == 1:
            for ds in self._yield_data_from_files(
                    fnames,
                    max_size_mb=buffer_memory_mb,
                    processes=processes
            ):
                ds = self._process_stack(ds)
                ds = ds.chunk({"obs": 1_000_000})
                ds.load()
                for cell in np.unique(ds.cell):
                    cell_fname = out_dir / f"{cell:04d}.nc"
                    cell_ds = ds.isel(obs=np.where(ds.cell.values == cell)[0])
                    cell_obj = cell_class(cell_fname, cell_ds)
                    cell_obj.ds = self._cell_data_as_indexed_ra(cell_obj.ds, cell, dupe_window)
                    out_encoding = create_variable_encodings(
                        cell_obj.ds, custom_dtypes=self.ts_dtype)
                    cell_obj.write(mode="a", encoding=out_encoding)
            return

        processor_ps = []
        num_processors = max(1, processes - 2)  # Ensure at least one processor

        # We could use "fork" here but since we have threads in the subprocesses, it will
        # throw a warning and eventually stop working in Python 3.14. Instead we set up
        # a forkserver which is slower to start up but doesn't have this issue.
        ctx = mp.get_context("forkserver")

        reader_q = ctx.Queue(maxsize=processes)

        # is it bad to have a bunch of processors here when I'm gonna use some more in
        # _yield_data_from_files?
        for _ in range(num_processors):
            p = ctx.Process(target=self._write_cell, args=(reader_q,))
            p.start()
            processor_ps.append(p)

        for ds in self._yield_data_from_files(fnames, max_size_mb=buffer_memory_mb):
            ds = ds.chunk({"obs": 1_000_000})
            ds = self._process_stack(ds)
            ds.load()
            for cell in np.unique(ds.cell):
                cell_fname = out_dir / f"{cell:04d}.nc"
                cell_ds = ds.isel(obs=np.where(ds.cell.values == cell)[0])
                cell_obj = cell_class(cell_fname, cell_ds)
                reader_q.put((cell, cell_obj, dupe_window))

            # Wait until the processes have finished with this batch before
            # moving on - ensures that there are no processors trying to
            # write to the same file at the same time.

            # This is unnecessary in most situations since we're passing cell
            # data to reader_q in order, but if there are a lot of processors
            # and relatively few files it would be possible to run into this
            # issue.

            # We don't use .join() because the forkserver takes a bit to set up
            # and we would like to just leave it open for the next batch instead
            # of reopening it.

            while reader_q.qsize() > 0:
                pass

        [reader_q.put(None) for p in processor_ps]
        [p.join() for p in processor_ps]
