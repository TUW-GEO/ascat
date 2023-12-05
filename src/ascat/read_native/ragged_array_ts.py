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
import gc
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from functools import partial

import xarray as xr
import numpy as np

import psutil
import tqdm
import dask
import dask.array as da
import dask.bag as db
from dask.distributed import Client
from fibgrid.realization import FibGrid
from ascat.file_handling import ChronFiles
from ascat.read_native.xarray_io import ascat_io_metadata

# int8_nan = np.iinfo(np.int8).max
# int64_nan = np.iinfo(np.int64).min
# NC_FILL_FLOAT = np.float32(9969209968386869046778552952102584320)


class CellFileCollectionStack():
    """
    Collection of grid cell file collections
    """

    def __init__(
            self,
            collections,
            ascat_id=None,
            # ioclass=None,
            common_grid=None,
            dupe_window=None,
            dask_scheduler="threads",
            **kwargs
    ):
        """
        Initialize.

        Parameters
        ----------
        collections: list of str or CellFileCollection
        """

        metadata = ascat_io_metadata(ascat_id)
        if isinstance(collections, (str, Path)):
            collections = [collections]

        if isinstance(collections[0], (str, Path)):
            all_subdirs = [
                Path(r) for c in collections for (r, d, f) in os.walk(c) if not d
            ]
            self.collections = [
                CellFileCollection(subdir, metadata=metadata, ioclass_kws=kwargs)
                for subdir in all_subdirs
            ]
        else:
            self.collections = collections

        self.common_grid = common_grid
        self.dupe_window = dupe_window or np.timedelta64(10, "m")
        self.grids = []
        common_cell_size = None
        self._different_cell_sizes = False

        for coll in self.collections:
            if coll.grid not in self.grids:
                self.grids.append(coll.grid)
            if common_cell_size is None:
                common_cell_size = coll.grid_cell_size
            elif coll.grid_cell_size != common_cell_size:
                self._different_cell_sizes = True

        if dask_scheduler is not None:
            dask.config.set(scheduler=dask_scheduler)
        # self._client = Client(n_workers=1, threads_per_worker=16)

    def add_collection(self, collections, ascat_id=None):
        """
        Add a cell file collection to the stack,
        based on file path.

        Parameters
        ----------
        collections : str or list of str or CellFileCollection
            Path to the cell file collection to add, or a list of paths.
        ascat_id : str, optional
            ASCAT ID of the collections to add. Needed if collections is a string or
            list of strings.
        """
        new_idx = len(self.collections)
        if isinstance(collections, (string, Path)):
            collections = [collections]
        if isinstance(collections[0], str):
            self.collections.extend(
                CellFileCollection(c, ascat_id) for c in collections
            )
        elif isinstance(collections[0], CellFileCollection):
            self.collections.extend(collections)
        else:
            raise ValueError(
                "collections must be a list of strings or CellFileCollection objects"
            )

        common_cell_size = None
        for coll in self.collections[new_idx:]:
            if coll.grid not in self.grids:
                self.grids.append(coll.grid)
            if common_cell_size is None:
                common_cell_size = coll.grid_cell_size
            elif coll.grid_cell_size != common_cell_size:
                self._different_cell_sizes = True

    def read(self, cell=None, location_id=None, bbox=None, out_grid=None, **kwargs):
        """
        Read data for a cell, location_id or bbox.
        If multiple grids are present, then out_grid must be specified.

        Parameters
        ----------
        cell : int
            Cell number to read data for
        location_id : int
            Location ID to read data for
        bbox : tuple of floats
            Bounding box to read data for (lonmin, latmin, lonmax, latmax)
        out_grid : pygeogrids.CellGrid
            Grid to regrid data to
        **kwargs : dict
            Keyword arguments to pass to the read function of the collection
        """
        out_grid = out_grid or self.common_grid
        if (len(self.grids) > 1) and out_grid is None:
            raise ValueError("Multiple grids found, need to specify out_grid"
                             + "as argument to read function or common_grid as"
                             + "argument to __init__")

        if cell is not None:
            data = self._read_cells(cell, out_grid, **kwargs)
        elif location_id is not None:
            data = self._read_locations(location_id, out_grid, **kwargs)
        elif bbox is not None:
            raise NotImplementedError
        else:
            raise ValueError("Need to specify either cell, location_id or bbox")

        data = data.sortby(["sat_id", "locationIndex", "time"])

        #### DEDUPE
        dupl = np.insert(
            (abs(data["time"].values[1:] -
                 data["time"].values[:-1]) < self.dupe_window),
            0,
            False,
        )
        data = data.sel(obs=~dupl)

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
        #     data = self._reasample_to_grid(data, out_grid)

        return data

    def write_cells(self, out_dir, ioclass, cells=None, out_cell_size=None, **kwargs):
        """
        Merge the data in all the collection by cell, and write each cell to disk.

        Parameters
        ----------
        out_dir : str or Path
            Path to output directory.
        ioclass : class
            IO class to use for the writer.
        cells : list of int, optional
            Cells to write. If None, write all cells.
        out_cell_size : tuple, optional
            Size of the output cells in degrees (assumes they are square).
            If None, and the component collections all have the same cell size,
            use that.
        **kwargs
            Keyword arguments to pass to the ioclass write function.
        """
        # out_grid = out_grid or self.common_grid
        # if (len(self.grids) > 1) and out_grid is None:
        #     raise ValueError("Multiple grids found, need to specify out_grid"
        #                      + "as argument to read function or common_grid as"
        #                      + "argument to __init__")

        if out_cell_size is None and self._different_cell_sizes is True:
            raise ValueError("Different cell sizes found, need to specify out_cell_size"
                             + " as argument to write_cells function")
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        cells = self._subcollection_cells(out_cell_size)

        for cell in tqdm.tqdm(cells):
            self._write_single_cell(out_dir, ioclass, cell, out_cell_size, **kwargs)

    def _subcollection_cells(self, out_cell_size=None):
        """
        Get the cells that are covered by all the subcollections.
        If out_cell_size is passed, then it returns the cells in the new
        cell-scheme that are covered by the subcollections.

        Parameters
        ----------
        out_cell_size : int, optional
            The size of the cells in the new cell-scheme.

        Returns
        -------
        cells : set
            Cells covered by all subcollections.
        """
        if out_cell_size is None:
            return {c for coll in self.collections for c in coll.cells_in_collection}

        new_cells = set()
        # if we assume the collections all have the same grid etc then we only
        # really need to check the first collection
        # But would be better to redesign so we just pull from a grid or somethin
        # Or return a list for each collection but then the reading logic should
        # be different.
        for coll in self.collections:
            coll.create_cell_lookup(out_cell_size)
            new_cells.update(k for k,v in coll.cell_lut.items()
                             if np.any(np.isin(v, coll.cells_in_collection)))
        return new_cells

    def _read_cells(
        self,
        cells,
        # out_grid=None,
        out_format="contiguous",
        dupe_window=None,
        search_cell_size=None,
        **kwargs
    ):
        """
        Read data for a list of cells.
        If search_cell_size is passed, then it interprets "cells" with respect to the
        new cell-scheme, determines the cells from the original cell-scheme that are
        covered by the new cells, reads those, and trims the extraneous grid points from
        each new cell.
        """
        search_cells = cells if isinstance(cells, list) else [cells]

        if dupe_window is None:
            dupe_window = np.timedelta64(10, "m")

        if search_cell_size is not None:
            data = []
            for coll in self.collections:
                coll.create_cell_lookup(search_cell_size)
                old_cells = [c
                             for cell in search_cells
                             for c in coll.cell_lut[cell]
                             if c in coll.cells_in_collection]

                data.extend(
                    [
                        self._trim_to_gpis(
                            coll.read(
                                cell=cell,
                                # new_grid=out_grid,
                                mask_and_scale=False,
                                **kwargs
                            ),
                            coll.grid.grid_points_for_cell(cell)[0].compressed(),
                        )
                        for cell in old_cells
                    ]
                )

        else:
            data = [coll.read(cell=cell,
                              # new_grid=out_grid,
                              mask_and_scale=False,
                              **kwargs)
                    for coll in self.collections
                    for cell in search_cells]

        data = [ds for ds in data if ds is not None]
        # print([self._only_locations(ds) for ds in data])
        if data == []:
            return None

        locs_merged = xr.combine_nested(
            [self._only_locations(ds) for ds in data], concat_dim="locations"
        )

        _, idxs = np.unique(
            locs_merged["location_id"].values, return_index=True)

        location_vars = {
            var: locs_merged[var][idxs]
            for var in locs_merged.variables
        }

        location_sorter = np.argsort(location_vars["location_id"].values)

        locs_merged.close()

        merged_ds = xr.combine_nested(
            [self._preprocess(ds, location_vars, location_sorter) for ds in data],
            concat_dim="obs",
            data_vars="minimal",
            coords="minimal",
            combine_attrs="drop_conflicts",
        )


        return merged_ds

    def _read_locations(
        self,
        location_ids,
        # out_grid=None,
        out_format="contiguous",
        dupe_window=None,
        **kwargs
    ):
        location_ids = (
            location_ids
            if isinstance(location_ids, (list, np.ndarray))
            else [location_ids]
        )

        if dupe_window is None:
            dupe_window = np.timedelta64(10, "m")

        # all data here is converted to the SAME GRID CELL SIZE within coll.read()
        # before being merged later
        data = [d for d in
                (coll.read(
                    location_id=location_id,
                    # new_grid=out_grid,
                    mask_and_scale=False,
                    **kwargs)
                 for coll in self.collections
                 for location_id in location_ids)
                if d is not None]
        # data = []
        # for coll in self.collections:
        #     for location_id in location_ids:
        #         ds = coll.read(location_id=location_id,
        #                        new_grid=out_grid,
        #                        mask_and_scale=False,
        #                        **kwargs)
        #         if ds is not None:
        #             data.append(ds)

        # merge all the locations-dimensional data vars into one dataset,
        # in order to determine the grid points and indexing for the locations
        # dimension of the merged dataset.

        # coords="all" is necessary in case one of the coords has nan values
        # (e.g. altitude, in the case of ASCAT H129)
        locs_merged = xr.combine_nested(
            [self._only_locations(ds) for ds in data], concat_dim="locations",
            coords="all",
        )

        _, idxs = np.unique(locs_merged["location_id"].values, return_index=True)

        location_vars = {
            var: locs_merged[var][idxs]
            for var in locs_merged.variables
        }

        location_sorter = np.argsort(location_vars["location_id"].values)

        locs_merged.close()

        # merge the data variables
        merged_ds = xr.combine_nested(
            [self._preprocess(ds, location_vars, location_sorter) for ds in data],
            concat_dim="obs",
            data_vars="minimal",
            coords="minimal",
        )

        return merged_ds

    def _write_single_cell(self, out_dir, ioclass, cell, out_cell_size, **kwargs):
        """
        Write data for a single cell from the stack to disk.

        Parameters
        ----------
        out_dir : str or Path
            Path to output directory.
        ioclass : class
            IO class to use for the writer.
        cell : tuple
            Cell to write.
        out_cell_size : tuple
            Size of the output cell.
        **kwargs
            Keyword arguments to pass to the ioclass write function.
        """

        data = self.read(cell=cell, search_cell_size=out_cell_size)
        # data = data[self._var_order(data)]
        writer = ioclass(data)
        writer.write(out_dir / writer.fn_format.format(cell), **kwargs)
        data.close()
        writer.close()

    @staticmethod
    def _preprocess(ds, location_vars, location_sorter):
        """
        Pre-processing to be done on a component dataset before merging it with others.
        Assumes ds is an indexed ragged array. (Re)-calculates the locationIndex values
        for ds with respect to the location_id variable for the merged dataset, which
        may include locations not present in ds.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        location_vars : dict
            Dictionary of ordered location variable DataArrays for the merged data.
        location_sorter : np.ndarray
            Result of np.argsort(location_vars["location_id"]), used to calculate
            the locationIndex variable. Calculated outside this function to avoid
            re-calculating it for every dataset being merged.

        Returns
        -------
        xarray.Dataset
            Dataset with pre-processing applied.
        """
        # First, we need to calculate the locationIndex variable, based
        # on the location_id variable that will go on the final merged dataset.
        # This should have been stored in self.location_vars["location_id"] at some
        # point before reaching this function, along with all the other
        # locations-dimensional variables in the combined dataset.

        # if "locations" is in the dataset dimensions, then we have
        # a multi-location dataset.
        if "locations" in ds.dims:
            ds = ds.dropna(dim="locations", subset=["location_id"])

            ds["locationIndex"] = (
                "obs",
                location_sorter[np.searchsorted(
                    location_vars["location_id"].values,
                    ds["location_id"].values[ds["locationIndex"]],
                    sorter=location_sorter,
                )],
            )

            ds = ds.drop_dims("locations")

        # if not, we just have a single location, and logic is different
        else:
            ds["locationIndex"] = (
                "obs",
                location_sorter[np.searchsorted(
                    location_vars["location_id"].values,
                    np.repeat(ds["location_id"].values, ds["locationIndex"].size),
                    sorter=location_sorter,
                )],
            )

        # Next, we put the locations-dimensional variables on the dataset,
        # and set them as coordinates.
        for var, var_data in location_vars.items():
            ds[var] = ("locations", var_data.values)
        ds = ds.set_coords(["lon", "lat", "alt", "time"])

        try:
            # Need to reset the time index if it's already there, but I can't
            # figure out how to test if time is already an index except like this
            ds = ds.reset_index("time")
        except ValueError:
            pass

        return ds

    @staticmethod
    def _only_locations(ds):
        """
        Returns a dataset with only the locations-dimensional variables.
        (technically, the non-obs-dimensional variables, to handle the case
        where we only have one location and therefore no locations dimension)

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
    def _trim_to_gpis(ds, gpis):
        """
        Trim a dataset to only the gpis in the given list.
        If any gpis are passed which are not in the dataset, they are ignored.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        gpis : list or list-like
            List of gpis to keep.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with only the gpis in the list.
        """
        if ds is None:
            return None

        # first trim out any gpis not in the dataset from the gpi list
        gpis = np.intersect1d(gpis, ds.location_id.values, assume_unique=True)
        # then trim out any gpis in the dataset not in gpis
        locations_idx = np.searchsorted(ds.location_id.values, gpis)
        obs_idx = np.in1d(ds.locationIndex, locations_idx)

        return ds.isel({"obs": obs_idx, "locations": locations_idx})


class CellFileCollection:

    """
    Grid cell files.
    """

    def __init__(self,
                 path,
                 ascat_id=None,
                 metadata=None,
                 # ioclass,
                 cache=False,
                 # fn_format="{:04d}.nc",
                 ioclass_kws=None,
                 ):
        """
        Initialize.
        """
        self.path = path
        self.metadata = metadata or ascat_io_metadata(ascat_id)
        if self.metadata is None:
            raise ValueError("Either ascat_id or metadata must be given")
        self.ioclass = self.metadata.cell_ioclass
        self.grid = self.metadata.grid
        self.grid_cell_size = self.metadata.grid_cell_size

        # possible_cells = self.grid.get_cells()
        # self.cell_max = possible_cells.max()
        # self.cell_min = possible_cells.min()

        self.fn_format = self.metadata.cell_fn_format
        # ASSUME THE IOCLASS RETURNS XARRAY
        # self.ioclass = ioclass
        # self.fn_format = fn_format
        self.previous_cell = None
        self.fid = None
        self.min_time = None
        self.max_time = None
        self.out_cell_size = None
        self.cell_lut = None

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws

    @property
    def cells_in_collection(self):
        return [int(p.stem) for p in self.path.glob("*")]

    def read(self, cell=None, location_id=None, coords=None, new_grid=None, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it

        Parameters
        ----------
        cell : int
            Grid cell number to read.
        location_id : int
            Location id.
        coords : tuple
            Tuple of (lon, lat) coordinates.
        new_grid : pygeogrids.CellGrid
            New grid to convert to.
        **kwargs : dict
            Keyword arguments passed to the ioclass.
        """
        # new_grid = kwargs.pop("new_grid", False)
        kwargs = {**self.ioclass_kws, **kwargs}
        if cell is not None:
            data = self._read_cell(cell, **kwargs)
        elif location_id is not None:
            if new_grid is not None:
                warnings.warn(
                    "You have specified a new_grid but are searching for a location_id."
                    " Currently, the location_id argument searches the original grid."
                    " The returned data will be converted to the new grid and will"
                    " likely have different location_id values from those you searched"
                    " for.", stacklevel=2)
            data = self._read_location_id(location_id, **kwargs)
        elif coords:
            data = self._read_lonlat(coords[0], coords[1], **kwargs)
        else:
            raise ValueError("Either cell, location_id or coords (lon, lat)"
                             " must be given")

        if new_grid is not None:
            data = self._convert_to_grid(data, new_grid)

        return data

    def create_cell_lookup(self, out_cell_size):
        """
        Create a lookup table to map the new cell grid to the old cell grid,
        and store it in self.cell_lut

        Parameters
        ----------
        out_cell_size : int
            Cell size of the new grid.
        """
        if out_cell_size != self.out_cell_size:
            self.out_cell_size = out_cell_size
            new_grid = self.grid.to_cell_grid(out_cell_size)
            old_cells = self.grid.arrcell
            new_cells = new_grid.arrcell
            self.cell_lut = {new_cell:
                             np.unique(old_cells[np.where(new_cells == new_cell)[0]])
                             for new_cell in np.unique(new_cells)}

    def _open(self, location_id=None, cell=None, grid=None):
        """
        Open cell file using the given location_id (will open the cell file containing
        the location_id) or cell number. Sets self.fid to an xarray dataset representing
        the open cell file.

        Parameters
        ----------
        location_id : int (optional)
            Location identifier.
        cell : int (optional)
            Cell number.
        grid : pygeogrids.CellGrid (optional)
            Grid object.

        Returns
        -------
        success : boolean
            Flag if opening the file was successful.
        """
        success = True
        if location_id is not None:
            grid = grid or self.grid
            cell = grid.gpi2cell(location_id)
        filename = self._get_cell_path(cell)

        if self.previous_cell != cell:
            self.close()

            try:
                self.fid = self.ioclass(filename, **self.ioclass_kws)
            except IOError as e:
                success = False
                self.fid = None
                msg = f"I/O error({e.errno}): {e.strerror}, {filename}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self.previous_cell = None
            else:
                self.previous_cell = cell

        return success

    def _read_cell(self, cell, **kwargs):
        """
        Read data from the entire cell.

        Parameters
        ----------
        cell : int
            Cell number.

        Returns
        -------
        data : xarray.Dataset
            Data for the cell.
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
        data : xarray.Dataset
            Data at the given coordinates.
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
        data : xarray.Dataset
            Data for the given grid point.
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

        Parameters
        ----------
        cell : int (optional)
            Cell number.
        location_id : int (optional)
            Location identifier.

        Returns
        -------
        path : pathlib.Path
            Path to cell file.
        """
        if location_id is not None:
            cell = self.grid.gpi2cell(location_id)
        elif cell is None:
            raise ValueError("Either location_id or cell must be given")

        if (cell > self.metadata.max_cell) or (cell < self.metadata.min_cell):
            raise ValueError(f"Cell {cell} is not in grid")

        return self.path / self.fn_format.format(cell)

    def _convert_to_grid(self, data, new_grid, old_grid=None):
        """
        Convert the data to a new grid.
        NEEDS TO BE REWRITTEN TO RESAMPLE INSTEAD
        """
        raise NotImplementedError

    def close(self):
        """
        Close file.
        """
        if self.fid is not None:
            self.fid.close()
            self.fid = None

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
                 metadata=None,
                 ioclass_kws=None,
                 dask_scheduler=None,
                 ):
        self.path = path
        self.metadata = metadata or ascat_io_metadata(ascat_id)
        self.ioclass = self.metadata.swath_ioclass
        self.grid = self.ioclass.grid
        self.ts_dtype = self.ioclass.ts_dtype

        # possible_cells = self.grid.get_cells()
        # self.cell_max = possible_cells.max()
        # self.cell_min = possible_cells.min()

        self.fn_pattern = self.ioclass.fn_pattern
        self.sf_pattern = self.ioclass.sf_pattern
        self.fn_read_fmt = self.ioclass.fn_read_fmt
        self.sf_read_fmt = self.ioclass.sf_read_fmt
        self.date_format = self.ioclass.date_format
        self.fn_format = self.ioclass.cell_fn_format
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
        self._open_fnames = None
        self.fid = None
        self.min_time = None
        self.max_time = None
        self.max_process_memory_mb = 8*1024

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws

        if dask_scheduler is not None:
            dask.config.set(scheduler=dask_scheduler)

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

        fnames = self.chron_files.search_period(start_dt,
                                                end_dt,
                                                date_str=self.date_format)
        return fnames

    def _open(self, fnames):
        """
        Open swath files

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
        if fnames != self._open_fnames:
            self.fid = None

        if self.fid is None:
            try:
                self.fid = self.ioclass(fnames, self.grid, **self.ioclass_kws)
            except IOError as e:
                success = False
                self.fid = None
                msg = f"I/O error({e.errno}): {e.strerror}"
                warnings.warn(msg, RuntimeWarning)
        # else:
        #     # handling for adding new extra data to the open dataset
        #     pass

        return success

    def _convert_to_indexed_ra(self, ds):
        """
        Convert dataset to indexed format.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset.

        Returns
        -------
        ds : xarray.Dataset
            Output dataset.
        """
        ds = ds.drop_vars(["latitude", "longitude", "cell"], errors="ignore")
        location_id, locationIndex = np.unique(ds.location_id.values,
                                               return_inverse=True)
        lon, lat = self.grid.gpi2lonlat(location_id)
        ds = ds.assign_coords({"lon": ("locations", lon),
                               "lat": ("locations", lat),
                               "alt": ("locations", np.repeat(np.nan, len(location_id)))})
        ds = ds.set_coords(["time"])
        ds["location_id"] = ("locations", location_id)
        ds["location_description"] = ("locations", np.repeat("", len(location_id)))
        return ds

    def write_cell_ds(self, ds, out_path):
        print(ds)
        # writer = self.ioclass(self.convert_to_indexed_ra(ds), self.grid)
        # writer.write(out_path)
        # writer.close()

    def stack(self, fnames, out_path, chunks=None):
        buffer = []
        buffer_size = 0
        process = psutil.Process()
        print(len(fnames))
        for iter, f in enumerate(fnames):
            try:
                self._open(f)
            except:
                # print warning?
                continue
            ds = self.fid.read(chunks=chunks).load()
            buffer.append(ds)
            buffer_size += ds.nbytes / 1e6
            # buffer_size  = process.memory_info().rss / 1e6
            # mem_use = sum([x.nbytes for x in buffer]) / 1e6
            if buffer_size > self.max_process_memory_mb:
                print(buffer_size)
                print(f)
                out_dir = out_path / f"{iter:05d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                print(out_dir)
                combined_ds = self.process(xr.combine_nested(buffer,
                                                concat_dim="obs",
                                                combine_attrs="drop_conflicts"))
                print(combined_ds)
                # for cell in np.unique(combined_ds.cell.values):
                #     cell_ds = combined_ds.sel(cell=cell)
                with mp.Pool(processes=8) as pool:
                    pool.starmap(self.write_cell_ds,
                                 [(cell_ds.to_dataframe(), out_dir / self.fn_format.format(cell))
                                  for cell, cell_ds
                                  in combined_ds.groupby("cell")])
                # for cell, cell_ds in combined_ds.groupby("cell"):
                #     self.write_cell_ds(cell_ds, out_dir / self.fn_format.format(cell))
                    # writer = self.ioclass(self._convert_to_indexed_ra(cell_ds), self.grid)
                    # writer.write(out_dir / self.fn_format.format(cell))
                    # writer.close()
                combined_ds.close()
                buffer_size = 0
                buffer = []
                # self.process(xr.combine_nested(buffer, dim="obs"))

        if len(buffer) > 0:
            iter += len(buffer)
            out_dir = out_path / f"{iter:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for cell, cell_ds in self.process(
                    xr.combine_nested(buffer,
                                      concat_dim="obs",
                                      combine_attrs="drop_conflicts")
            ).groupby("cell"):
                writer = self.ioclass(self._convert_to_indexed_ra(cell_ds), self.grid)
                writer.write(out_dir / self.fn_format.format(cell))
                writer.close()
            # self.process(xr.combine_nested(buffer, dim="obs"))

    def process(self, data, **kwargs):
        """
        Read data from the entire cell.
        """
        # if there are kwargs, use them instead of self.ioclass_kws

        # data = None
        # if self._open(start_dt, end_dt):
        #     data = self.fid.read()
        # print("read")

        #could add cell here or after processing ...
        # data = data.assign_coords({"cell": ("obs", self.grid.gpi2cell(data["location_id"].values))})
        # data["cell"] = ("obs", self.grid.gpi2cell(data["location_id"].values))
        # data = data.set_xindex("cell")
        # print("added cell")

        beam_idx = {"for": 0, "mid": 1, "aft": 2}
        sat_id = {"a": 3, "b": 4, "c": 5}

        # if any beam has backscatter data for a record, the record is valid
        print("dropping nans from backscatter if all three beams are nan")
        if data["obs"].size > 0:
            # valid = ~da.isnan(data["backscatter"]).all(axis=(1))
            data = data.dropna(dim="obs", how="all", subset=["backscatter"])

        print("creating beam vars")
        if data["obs"].size > 0:
            for var in self.ts_dtype.names:
                if var[:-4] in [
                    "backscatter",
                    "incidence_angle",
                    "azimuth_angle",
                    "kp",
                ]:
                    print(f"creating beam vars for {var}")
                    ending = var[-3:]
                    data[var] = data.sel(beams=beam_idx[ending])[var[:-4]]
                # if data[var].dtype != ts_dtypes[var]:
                #     data[var] = data[var].astype(ts_dtypes[var])
                #     data[var].attrs["dtype"] = ts_dtypes[var]
                #     data[var].attrs["_FillValue"] = dtype_to_nan(ts_dtypes[var])

            print("dropping beam var sources")
            data = data.drop_vars(["backscatter", "incidence_angle", "azimuth_angle", "kp"])
            sat = data.attrs["spacecraft"][-1].lower()
            print("adding sat_id")
            data["sat_id"] = ("obs", np.repeat(sat_id[sat], data["location_id"].size))

        print("adding cells")
        data = data.assign_coords({"cell": ("obs", self.grid.gpi2cell(data["location_id"].values))})
        # data["cell"] = ("obs", self.grid.gpi2cell(data["location_id"].values))
        print("setting xindex")
        data = data.set_xindex("cell")

        print("selecting valid obs")
        # data = data.isel(obs=valid)
        # data = data.sel(valid=True)
        # print("valid obs selected")

        return data

    def _read_cell(self, start_dt, end_dt, cell, **kwargs):
        """
        Read data from the entire cell.
        """
        # if there are kwargs, use them instead of self.ioclass_kws

        data = None
        if self._open(start_dt, end_dt):
            data = self.fid.read(cell=cell, **kwargs)

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

        if (cell > self.ioclass.max_cell) or (cell < self.ioclass.min_cell):
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

    def read(self, start_dt, end_dt, cell=None, location_id=None, coords=None, new_grid=None, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it
        """
        if cell is not None:
            data = self._read_cell(start_dt, end_dt, cell)

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

    def stack_cells(self, start_dt, end_dt, cells):
        """
        Stack data from a list of cells into one dataset for
        each cell.
        """
        for fname in self._get_filenames(start_dt, end_dt):
            try:
                self._open(fname)
            except:
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


####################################################
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
        raise ValueError("No row_size or locationIndex in dataset."
                          + "Cannot determine if indexed or ragged")

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

    raise ValueError("Dataset must have either locationIndex or row_size."
                     + "Cannot determine if ragged array is indexed or contiguous"
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
        raise ValueError("No row_size or locationIndex in dataset."
                          + "Cannot determine if indexed or ragged")

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
