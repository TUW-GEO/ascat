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

from functools import partial
from pathlib import Path

import xarray as xr
import numpy as np

import ascat.accessors
from ascat.grids import GridRegistry

from ascat.file_handling import Filenames
from ascat.utils import get_grid_gpis
from ascat.utils import append_to_netcdf
import warnings


class RaggedArrayTs(Filenames):
    """
    Class to read and merge ragged array cell files.
    """
    def _read(
        self,
        filename,
        location_id=None,
        lookup_vector=None,
        date_range=None,
        preprocessor=None,
        **xarray_kwargs
    ):
        """
        Open one Ragged Array file as an xarray.Dataset and preprocess it if necessary.

        Parameters
        ----------
        filename : str
            File to read.
        location_id : int or list of int
            Location id. Only used for selecting points from contiguous ragged arrays.
            Not used for indexed ragged arrays or point arrays.
        lookup_vector : np.ndarray
            Lookup vector for faster selection.
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.
        preprocessor : callable, optional
            Function to preprocess the dataset.
        xarray_kwargs : dict
            Additional keyword arguments passed to xarray.open_dataset.

        Returns
        -------
        ds : xarray.Dataset
            Dataset.
        """
        if ds := self.cache.get(filename):
            pass
        else:
            ds = xr.open_dataset(filename, engine = "h5netcdf", **xarray_kwargs)
            self.cache[filename] = ds

        if preprocessor:
            ds = preprocessor(ds)
        ds = self._ensure_obs(ds)

        if ds.cf_geom.array_type == "contiguous":
            ds = self._trim_to_gpis(ds, gpis=location_id, lookup_vector=lookup_vector)

        elif ds.cf_geom.array_type == "indexed":
            ds.time.load()
            # we need to make sure the time variable is in memory before converting to
            # a point array, since reordering this as a dask array will explode the process
            # graph later on
            ds = self._ensure_point(ds)
            if date_range is not None:
                ds = self._trim_var_range(ds, "time", *date_range)

        return ds


    def read(self,
             date_range=None,
             location_id=None,
             lookup_vector=None,
             preprocessor=None,
             return_format=None,
             parallel=False,
             **kwargs):
        """
        Read data from Ragged Array Cell files.

        Parameters
        ----------
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.
        location_id : list of int
            List of timeseries IDs to read.
        lookup_vector : np.ndarray
            Lookup vector.
        preprocessor : callable, optional
            Function to preprocess the dataset.
        return_format : str, optional
            CF discrete geometry format to return data as. Can be "point", "indexed", or "contiguous".
        parallel : bool, optional
            Whether or not to read/preprocess in parallel. Default is False.
        **kwargs : dict
        """
        ds, closers = super().read(date_range=date_range,
                                   location_id=location_id,
                                   lookup_vector=lookup_vector,
                                   preprocessor=preprocessor,
                                   closer_attr="_close",
                                   parallel=parallel,
                                   **kwargs)

        if ds is not None:
            ds.set_close(partial(super()._multi_file_closer, closers))

            if ds.cf_geom.array_type == "contiguous" and date_range is not None:
                ds = self._dim_safe_rechunk(ds)
                ds = self._trim_var_range(ds, "time", *date_range)
                ds = self._dim_safe_rechunk(ds)

            if ds.cf_geom.array_type != "contiguous":
                ds = self._trim_to_gpis(ds.chunk({"obs": 1000000}), gpis=location_id, lookup_vector=lookup_vector)

            if return_format is not None:
                if return_format == "point":
                    return ds.cf_geom.to_point_array()

                elif return_format == "indexed":
                    return ds.cf_geom.to_indexed_ragged()

                elif return_format == "contiguous":
                    return ds.cf_geom.to_contiguous_ragged()
                else:
                    raise ValueError("return_format must be 'point', 'indexed', "
                                     "'contiguous', or None (default, returns as read)")
            return ds


    @staticmethod
    def _indexed_or_contiguous(ds):
        if "locationIndex" in ds:
            return "indexed"
        return "contiguous"

    @staticmethod
    def _array_type(ds):
        """
        Determine the type of ragged array.
        """
        if "locationIndex" in ds:
            return "indexed"
        if "row_size" in ds:
            return "contiguous"

    @staticmethod
    def _trim_var_range(ds, var_name, var_min, var_max, end_inclusive=False):
        if end_inclusive:
            mask = (ds[var_name] >= var_min) & (ds[var_name] <= var_max)
        else:
            mask = (ds[var_name] >= var_min) & (ds[var_name] < var_max)

        mask = mask.compute()
        if ds.cf_geom.array_type == "contiguous" and "locations" in ds.dims:
            group_indices = np.repeat(np.arange(ds.row_size.size), ds.row_size)
            kept_counts = np.bincount(group_indices,
                                      weights=mask.astype(int),
                                      minlength=ds.row_size.size).astype(np.int32)
            ds = ds.sel(obs=mask)
            ds["row_size"].values = kept_counts
            return ds
        else:
            return ds.sel(obs=mask)


    @staticmethod
    def _ensure_obs(ds):
        ds = ds.cf_geom.set_sample_dimension("obs")
        return ds

    @staticmethod
    def _ensure_point(ds):
        return ds.cf_geom.to_point_array()

    @staticmethod
    def _ensure_indexed(ds):
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
        if ds is not None:
            return ds.cf_geom.to_indexed_ragged()


    @staticmethod
    def _ensure_contiguous(ds):
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
        if ds is not None:
            return ds.cf_geom.to_contiguous_ragged(sort_vars=["time"])

    @staticmethod
    def _only_locations(ds):
        """Return a dataset with only the variables that aren't in the obs-dimension.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset with only the locations-dimensional variables.
        """
        return ds[[
            var
            for var in ds.variables
            if "obs" not in ds[var].dims
            and var not in ["row_size", "locationIndex"]
        ]]

    @staticmethod
    def _dim_safe_rechunk(data):
        if "locations" in data.dims:
            return data.chunk({"obs": 1000000, "locations": -1})
        return data.chunk({"obs": 1000000})


    def _merge(self, data):
        """
        Merge datasets with potentially different locations dimensions.

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

        if data[0].cf_geom.array_type == "indexed":
            return self._merge_indexed(data)
        elif data[0].cf_geom.array_type == "point":
            return self._merge_point(data)
        elif data[0].cf_geom.array_type == "contiguous":
            return self._merge_contiguous(data)
        else:
            raise ValueError("Array type must be 'contiguous', 'indexed', or 'point'")


    def _merge_indexed(self, data):
        preprocessed_point_arrays_to_merge = [(self._preprocess(ds)
                                               .chunk({"obs":-1})
                                               .cf_geom.to_point_array())
                                              for ds in data if ds is not None]
        merged_point_ds = xr.concat(
            preprocessed_point_arrays_to_merge,
            dim="obs",
            data_vars="minimal",
            compat="equals",
            combine_attrs="drop_conflicts",
        )
        merged_indexed_ds = merged_point_ds.cf_geom.to_indexed_ragged()
        return merged_indexed_ds

    def _merge_point(self, data):
        merged_ds = xr.concat(
            [self._preprocess(ds).chunk({"obs":-1})
                for ds in data if ds is not None],
            dim="obs",
            combine_attrs="drop_conflicts",
        )
        return merged_ds

    @staticmethod
    def _only_obs_vars(ds):
        return ds[[var for var in ds.variables if "obs" in ds[var].dims]]

    def _merge_contiguous(self, data):
        preprocessed = [self._preprocess(ds).chunk({"obs": -1})
                        for ds in data if ds is not None]
        obs_dim_ds_to_concat = [self._only_obs_vars(ds) for ds in preprocessed]
        non_obs_dim_ds_to_concat = [ds.drop_dims("obs") for ds in preprocessed]
        obs_dim_ds = xr.concat(obs_dim_ds_to_concat, dim="obs")
        # any variables without dims will be in non_obs_dim_ds, we handle them
        # by assuming they should be equal on all merged data - not sure if this
        # is the right choice. We could also override.
        non_obs_dim_ds = xr.concat(non_obs_dim_ds_to_concat,
                                   dim="locations",
                                   data_vars="minimal",
                                   compat="equals")
        merged_ds = xr.merge([obs_dim_ds, non_obs_dim_ds])
        merged_ds = self._dim_safe_rechunk(merged_ds)
        return merged_ds

    def _preprocess(self, ds):
        """Pre-processing to be done on a component dataset so it can be merged with others.

        Nothing here at the moment.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset with pre-processing applied.
        """
        return ds

    def _trim_to_gpis(self, ds, gpis=None, lookup_vector=None):
        """Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep. One of gpis or lookup_vector
            must be provided.
        lookup_vector : np.ndarray
            Lookup vector from gpi numbers to bools for inclusion. One of gpis or lookup_vector
            must be provided.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return

        if (gpis is None or len(gpis) == 0) and (lookup_vector is None or len(lookup_vector)==0):
            return ds

        return ds.cf_geom.sel_instances(gpis, lookup_vector)

    def _write(self,
               data,
               filename,
               ra_type="indexed",
               mode="w",
               postprocessor=None,
               **kwargs):
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
        if ra_type == "contiguous":
            data = self._ensure_contiguous(data)
        if ra_type == "indexed":
            data = self._ensure_indexed(data)
        if ra_type == "point":
            data = self._ensure_point(data)

        if postprocessor is not None:
            data = postprocessor(data)

        data.encoding["unlimited_dims"] = ["obs"]

        if mode == "a" and ra_type in ["indexed", "point"]:
            if Path(filename).exists():
                append_to_netcdf(filename, data, unlimited_dim="obs")
                data.close()
                return

        import warnings
        with warnings.catch_warnings():
            # NetCDF will sometimes warn us about endianness for some reason and idk why
            warnings.filterwarnings("ignore",
                                    message="endian-ness of dtype and endian kwarg do not match, using endian kwarg",
                                    category=UserWarning)
            data.to_netcdf(filename, **kwargs)
        data.close()


class OrthoMultiTimeseriesCell(Filenames):
    """
    Class to read and merge orthomulti cell files.
    """
    def _read(self, filename, preprocessor=None, **xarray_kwargs):
        """
        Open one OrthoMulti file as an xarray.Dataset and preprocess it if necessary.

        Parameters
        ----------
        filename : str
            File to read.
        preprocessor : callable, optional
            Function to preprocess the dataset.
        xarray_kwargs : dict
            Additional keyword arguments passed to xarray.open_dataset.

        Returns
        -------
        ds : xarray.Dataset
            Dataset.
        """
        ds = xr.open_dataset(filename, **xarray_kwargs)
        if preprocessor:
            ds = preprocessor(ds)
        return ds

    def read(self,
             date_range=None,
             location_id=None,
             lookup_vector=None,
             preprocessor=None,
             parallel=False,
             **kwargs):
        """
        Read data from OrthoMulti Cell files.

        Parameters
        ----------
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.
        location_id : list of int
            List of timeseries IDs to read.
        lookup_vector : np.ndarray
            Lookup vector.
        preprocessor : callable, optional
            Function to preprocess the dataset.
        parallel : bool, optional
            Whether or not to read/preprocess in parallel. Default is False.
        """
        ds = super().read(preprocessor=preprocessor, **kwargs)
        if date_range is not None:
            ds = ds.sel(time=slice(*date_range))
        ds = self._trim_to_gpis(ds, gpis=location_id, lookup_vector=lookup_vector)

        return ds

    def _merge(self, data):
        """
        Merge datasets with different location and/or time dimensions.

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

        # ensures that time dimensions are aligned in case they are not
        data = xr.align(*data, join="outer")

        data = [ds.chunk(-1).set_index(locations="location_id")
                for ds in data if ds is not None]
        merged_ds = xr.concat(
            data,
            dim="locations",
            combine_attrs="drop_conflicts",
        ).rename_vars({"locations": "location_id"})
        merged_ds["location_id"].attrs["cf_role"] = "timeseries_id"

        return merged_ds

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
        lookup_vector : np.ndarray
            Lookup vector from gpi numbers to bools for inclusion.

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return

        if (gpis is None or len(gpis) == 0) and (lookup_vector is None or len(lookup_vector)==0):
            return ds

        return ds.cf_geom.sel_instances(gpis, lookup_vector)


class CellGridFiles():

    def __init__(
        self,
        root_path,
        file_class,
        grid,
        fn_format="{cell:04d}.nc",
        sf_format=None,
        preprocessor=None,
    ):
        """
        Initialize cell grid files.

        Parameters
        ----------
        root_path : str or Path
            Root path where the cell files are stored.
        file_class : class
            Class to use for file handling (e.g., RaggedArrayTs or OrthoMultiTimeseriesCell).
        grid : str or Grid
            Grid name or object defining the spatial grid structure.
        fn_format : str, optional
            Format string for cell file names, default is "{cell:04d}.nc".
        sf_format : str, optional
            Format string for subdirectories, if any.
        preprocessor : callable, optional
            Function to preprocess datasets when reading.
        """
        self.root_path = Path(root_path)
        self.file_class = file_class
        self.grid = grid
        self.fn_format = fn_format
        self.sf_format = sf_format
        self._preprocessor = preprocessor
        self._active_reader = None


    @classmethod
    def from_product_id(cls, root_path, product_id, **kwargs):
        from ascat.product_info import cell_io_catalog
        product_id = product_id.upper()
        if product_id in cell_io_catalog:
            product_class = cell_io_catalog[product_id]
            return cls.from_product_class(root_path, product_class, **kwargs)
        error_str = f"Product {product_id} not recognized. Valid products are"
        error_str += f" {', '.join(cell_io_catalog.keys())}."
        raise ValueError(error_str)

    @classmethod
    def from_product_class(cls, root_path, product_class, **kwargs):
        grid_name = product_class.grid_name
        init_options = {
            "root_path": root_path,
            "file_class": product_class.file_class,
            "grid": GridRegistry().get(grid_name),
            "fn_format": product_class.fn_format,
            "preprocessor": product_class.preprocessor,
            **kwargs
        }
        init_options = {**init_options}
        return cls(**init_options)

    def fn_search(self, cell):
        # get the paths to files matching a cell if the files exist
        filename = self.fn_format.format(cell)
        if self.sf_format is not None:
            subfolder = self.sf_format
            files = list(self.root_path.glob(subfolder + "/" + filename))
            return files
        else:
            return list(self.root_path.glob("**/" + filename))

    def convert_to_contiguous(self, out_dir, print_progress=True, **kwargs):
        """
        Convert all files in the collection to contiguous format and write to disk.

        Parameters
        ----------
        out_dir : str
            Output directory.
        print_progress : bool, optional
            Whether to print progress messages to console. Default is True.
        kwargs : dict
            Keyword arguments passed to the reprocess method.
        """
        self.reprocess(out_dir, lambda ds: ds, ra_type="contiguous", print_progress=True, **kwargs)

    def reprocess(self, out_dir, func, parallel=True, **kwargs):
        """
        Use Filenames.reprocess to apply a function to all files in the collection and
        save the results to `out_dir`.

        Parameters
        ----------
        out_dir : str
            Output directory.
        func : callable
            Function to apply to each file.
        parallel : bool, optional
            Whether to process files in parallel. Default is True.
        kwargs : dict
            Keyword arguments passed to func.
        """

        # get list of every filename in the collection
        filenames = self.spatial_search()
        files = self.file_class(filenames)
        files.reprocess(out_dir,
                        func,
                        parallel=parallel,
                        read_kwargs={"preprocessor": self._preprocessor},
                        **kwargs)


    def spatial_search(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
    ):
        """
        Search files for cells matching a spatial criterion. All args are declared
        as optional; but one and only one should be passed.

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
        geom : shapely.geometry
            Geometry object.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        if cell is not None:
            # guarantee cell is a list
            matched_cells = cell
            if not isinstance(matched_cells, list):
                matched_cells = [matched_cells]
        elif location_id is not None:
            # guarantee location_id is a list
            if not isinstance(location_id, list):
                location_id = [location_id]
            matched_cells = self._cells_for_location_id(location_id)
        elif coords is not None:
            matched_cells = self._cells_for_coords(coords)
        elif bbox is not None:
            matched_cells = self._cells_for_bbox(bbox)
        elif geom is not None:
            matched_cells = self._cells_for_geom(geom)
        else:
            matched_cells = self.grid.arrcell

        matched_cells = np.unique(matched_cells)

        filenames = []
        for cell in matched_cells:
            filenames += self.fn_search(cell)

        return filenames

    def _cells_for_location_id(self, location_id):
        """
        Get cells for location_id.

        Parameters
        ----------
        location_id : int
            Location id.

        Returns
        -------
        cells : list of int
            Cells.
        """
        cells = self.grid.gpi2cell(location_id)
        return cells

    def _cells_for_coords(self, coords):
        """
        Get cells for coordinates.

        Parameters
        ----------
        coords : tuple
            Coordinates (lon, lat)

        Returns
        -------
        cells : list of int
            Cells.
        """
        gpis = get_grid_gpis(self.grid, coords=coords)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _cells_for_bbox(self, bbox):
        """
        Get cells for bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box.

        Returns
        -------
        cells : list of int
            Cells.
        """
        gpis = get_grid_gpis(self.grid, bbox=bbox)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _cells_for_geom(self, geom):
        """
        Get cells for geometry.

        Parameters
        ----------
        geom : shapely.geometry
            Geometry object.

        Returns
        -------
        cells : list of int
            Cells.
        """
        gpis = get_grid_gpis(self.grid, geom=geom)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _fn(self, cell):
        return self.root_path / self.fn_format.format(cell)

    def read(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
            max_coord_dist=np.inf,
            date_range=None,
            **kwargs,
    ):
        """
        Read data matching a spatial and temporal criterion.

        Parameters
        ----------
        cell : int or list of int
            Grid cell number to read.
        location_id : int or list of int
            Location id.
        coords : tuple of numeric or tuple of iterable of numeric
            Tuple of (lon, lat) coordinates. lon and lat could each be numpy arrays in
            order to read multiple coordinates. For each coordinate the nearest grid point
            within `max_coord_dist` (in spherical cartesian coordinates) will be selected.

            Note that if any passed coordinates share the same nearest grid point, that grid
            point will only be represented once in the output dataset.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        geom : shapely.geometry
            Geometry object.
        max_coord_dist : float
            The maximum distance a coordinate's nearest grid point can be from it to be
            selected (in spherical cartesian coordinates). Default is np.inf.
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.

        Returns
        -------
        xarray.Dataset
            Filtered and merged data for the specified spatiotemporal region.

        """
        filenames = self.spatial_search(
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            geom=geom,
        )
        if ((self._active_reader is None)
            or not
            all(filename in self._active_reader.cache for filename in filenames)):
            self._active_reader = self.file_class(filenames)

        if all(criterion is None for criterion in [cell, location_id, coords, bbox, geom]):
            valid_gpis = None
            lookup_vector = None
        elif cell is not None:
            valid_gpis = None
            lookup_vector = None
        else:
            valid_gpis, lookup_vector = get_grid_gpis(
                self.grid,
                cell,
                location_id,
                coords,
                bbox,
                geom,
                max_coord_dist,
                return_lookup=True
            )

        out_ds = self._active_reader.read(date_range=date_range,
                                          location_id=valid_gpis,
                                          lookup_vector=lookup_vector,
                                          preprocessor=self._preprocessor,
                                          **kwargs)

        if out_ds is None:
            warning_str = ("No data found for specified criteria, returning None:\n"
                           f"cell={cell}, location_id={location_id}, coords={coords}, bbox={bbox}, geom={geom},\n"
                           f"max_coord_dist={max_coord_dist}, date_range={date_range}")
            warnings.warn(warning_str, UserWarning, 2)
            return out_ds
        return out_ds
