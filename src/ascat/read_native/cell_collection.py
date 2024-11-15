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
from functools import partial
from pathlib import Path

from tqdm import tqdm

import xarray as xr
import numpy as np

from ascat.read_native.xarray_accessors import *
from ascat.read_native.grid_registry import GridRegistry

from ascat.file_handling import MultiFileHandler
from ascat.file_handling import Filenames
from ascat.utils import get_grid_gpis
from ascat.utils import append_to_netcdf

class RaggedArrayCell(Filenames):
    """
    Class to read and merge ragged array cell files.
    """
    # def __init__(self, filenames):
        # self.filename = filename
        # self.ds = data          #
        # self.chunks = chunks

    def _read(
        self,
        filename,
        location_id=None,
        lookup_vector=None,
        date_range=None,
        generic=True,
        preprocessor=None,
        **xarray_kwargs
    ):
        """
        Open one Ragged Array file as an xarray.Dataset and preprocess it if necessary.

        Parameters
        ----------
        filename : str
            File to read.
        generic : bool, optional
            If True, the data is returned as a generic Indexed Ragged Array file for
            easy merging. If False, the file is returned as its native ragged array type.
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
        ds = self._ensure_obs(ds)

        if location_id is not None:
            ds = self._trim_to_gpis(ds, gpis=location_id)
        elif lookup_vector is not None:
            ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)
        # ds = self._ensure_indexed(ds)
        ds = self._ensure_point(ds)
        if date_range is not None:
            ds = self._trim_var_range(ds, "time", *date_range)

        return ds


    def read(self,
             date_range=None,
             location_id=None,
             lookup_vector=None,
             preprocessor=None,
             return_format="indexed",
             parallel=False,
             **kwargs):
        """
        Read data from Ragged Array Cell files.

        Parameters
        ----------
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.
        valid_gpis : list of int
            List of valid gpis.
        lookup_vector : np.ndarray
            Lookup vector.
        preprocessor : callable, optional
            Function to preprocess the dataset.
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

            if return_format == "point":
                return ds

            if return_format == "indexed":
                return ds.cf_geom.to_indexed_ragged()

            if return_format == "contiguous":
                return ds.cf_geom.to_contiguous_ragged()

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
        # if var_name in ds:
        if end_inclusive:
            mask = (ds[var_name] >= var_min) & (ds[var_name] <= var_max)
        else:
            mask = (ds[var_name] >= var_min) & (ds[var_name] < var_max)
        return ds.sel(obs=mask.compute())


    @staticmethod
    def _ensure_obs(ds):
        # basic heuristic - if obs isn't present, assume it's instead "time"
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
            return ds.cf_geom.to_contiguous_ragged()

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

    def _merge(self, data, **kwargs):
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

        merged_ds = xr.concat(
            [self._preprocess(ds).chunk({"obs":-1})
                for ds in data if ds is not None],
            dim="obs",
            # data_vars="minimal",
            # coords="minimal",
            combine_attrs="drop_conflicts",
            **kwargs,
        )

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
            List of gpis to keep.

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
        if ra_type not in ["contiguous", "indexed"]:
            raise ValueError("ra_type must be 'contiguous' or 'indexed'")
        if ra_type == "contiguous":
            data = self._ensure_contiguous(data)
        else:
            data = self._ensure_indexed(data)

        if postprocessor is not None:
            data = postprocessor(data)

        # data = data[self._var_order(data)]

        # custom_variable_attrs = self._kwargs.get(
        #     "attributes", None) or self.custom_variable_attrs
        # custom_global_attrs = self._kwargs.get(
        #     "global_attributes", None) or self.custom_global_attrs
        # data = self._set_attributes(data, custom_variable_attrs,
        #                               custom_global_attrs)

        # custom_variable_encodings = kwargs.pop(
        #     "encoding", None) or self.custom_variable_encodings
        # out_encoding = kwargs.pop("encoding", {})
        # out_encoding = create_variable_encodings(data, out_encoding)
        #
        data.encoding["unlimited_dims"] = ["obs"]

        # for var, var_encoding in out_encoding.items():
        #     if "_FillValue" in var_encoding and "_FillValue" in data[
        #             var].attrs:
        #         del data[var].attrs["_FillValue"]

        if mode == "a" and ra_type == "indexed":
            if not Path(filename).exists():
                data.to_netcdf(filename, **kwargs)
            else:
                append_to_netcdf(filename, data, unlimited_dim="obs")
            return

        data.to_netcdf(filename,
                       # encoding=out_encoding,
                       **kwargs)


class OrthoMultiCell(Filenames):
    """
    Class to read and merge orthomulti cell files.
    """
    # def __init__(self, filename, chunks=None):
    #     self.filename = filename
    #     if chunks is None:
    #         chunks = {"time": 1000, "locations": 1000}
    #     self.chunks = chunks
    #     self.ds = None

    def _read(self, filename, generic=True, preprocessor=None, **xarray_kwargs):
        """
        Open one OrthoMulti file as an xarray.Dataset and preprocess it if necessary.

        Parameters
        ----------
        filename : str
            File to read.
        generic : bool, optional
            If True, the data is returned as a generic Indexed Ragged Array file for
            easy merging. If False, the file is returned as its native ragged array type.
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


    def read(self, date_range=None, valid_gpis=None, lookup_vector=None, preprocessor=None, **kwargs):
        ds = super().read(preprocessor=preprocessor, **kwargs)
        # ds = ds.set_index(locations="location_id")
        # ds = ds.chunk(self.chunks)
        if date_range is not None:
            ds = ds.sel(time=slice(*date_range))
        if lookup_vector is not None:
            ds = self._trim_to_gpis(ds, lookup_vector=lookup_vector)
        elif valid_gpis is not None:
            # ds = ds.sel(locations=valid_gpis)
            ds = self._trim_to_gpis(ds, gpis=valid_gpis)
        # should I do it this way or just return the ds without having it be a class attribute?
        # self.ds = ds
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

        # This handles the case where the datasets have different time dimensions but
        # I'm not sure if I actually need to handle that case since the datasets I've
        # seen have all the years in each cell and just need to be combined by location.
        # This is robust, so I'll leave it for now, but consider some logic
        # to do just a standard concat by "locations" dim if the time dims are the same.
        #
        # Another note - need to chunk here before combining or else it tries to load
        # everything into memory at once. This isn't necessary if doing a regular concat
        # along a single dimension.

        merged_ds = xr.combine_by_coords(
            [ds.chunk(-1) for ds in data if ds is not None],
            combine_attrs="drop_conflicts",
        )
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

        Returns
        -------
        xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None
        if gpis is None and lookup_vector is None:
            pass

        elif gpis is None:
            ds_location_ids = ds["location_id"].data
            locs_idx = lookup_vector[ds_location_ids]
            ds = ds.sel(locations=locs_idx)
        else:
            ds = ds.where(ds["location_id"].isin(gpis).compute(), drop=True)

        return ds


class CellGridFiles():

    def __init__(
        self,
        root_path,
        file_class,
        grid,
        fn_format="{cell:04d}.nc",
        sf_format=None,
    ):
        self.root_path = Path(root_path)
        self.file_class = file_class
        self.grid = grid
        self.fn_format = fn_format
        self.sf_format = sf_format


    @classmethod
    def from_product_id(cls, root_path, product_id, **kwargs):
        from ascat.read_native.product_info import cell_io_catalog
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
            "grid": grid_registry.get(grid_name)["grid"],
            "fn_format": product_class.fn_format,
            **kwargs
        }
        init_options = {**init_options}
        return cls(**init_options)

    def fn_search(self, cell, sf_args=None):
        # get the paths to files matching a cell if the files exist

        filename = self.fn_format.format(cell)
        if sf_args is not None:
            subfolder = self.sf_format.format(**sf_args)
            return list(self.root_path.glob(subfolder / filename))
        else:
            # Should it return all below root path if no subfolder is specified?
            #return list(self.root_path.glob("**/" + filename))
            return list(self.root_path.glob(filename))


    def spatial_search(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
    ):
        """
        Search files for cells matching a spatial criterion.

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

        Notes
        -----
        TODO maybe can get rid of all the _cells_for_* methods and just do them here
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
        # self.grid.allpoints

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
        # gpis, _ = self.grid.find_nearest_gpi(*coords)
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
        # gpis = self.grid.get_bbox_grid_points(*bbox)
        gpis = get_grid_gpis(self.grid, bbox=bbox)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _cells_for_geom(self, geom):
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
        gpis = get_grid_gpis(self.grid, geom=geom)
        cells = self._cells_for_location_id(gpis)
        return cells

    def _fn(self, cell):
        return self.root_path / self.fn_format.format(cell=cell)

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
            Tuple of (lon, lat) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        max_coord_dist : float
            The maximum distance a coordinate's nearest grid point can be from it to be
            selected.
        date_range : tuple of np.datetime64
            Tuple of (start, end) dates.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        filenames = self.spatial_search(
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            geom=geom,
        )
        if cell is not None:
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

        return self.file_class(filenames).read(date_range=date_range,
                                               lookup_vector=lookup_vector,
                                               **kwargs)
