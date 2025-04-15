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

import os
import re
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime as dt

import dask
import xarray as xr
import numpy as np

from tqdm import tqdm
from shapely.geometry import Point

from ascat.read_native.xarray_io import cell_io_catalog
from ascat.read_native.xarray_io import swath_io_catalog

from ascat.read_native.xarray_io import trim_dates
from ascat.read_native.xarray_io import append_to_netcdf

from ascat.utils import Spacecraft

process_warnings = True

class CellFileCollectionStack():
    """Collection of grid cell file collections."""

    def __init__(
            self,
            collections,
            ioclass,
            dupe_window=None,
            dask_scheduler="threads",
            **kwargs
    ):
        """Initialize.

        Parameters
        ----------
        collections: list of str or CellFileCollection
            A path to a cell file collection or a list of paths to cell file
            collections, or a list of CellFileCollection.
        ioclass: str, optional
            Name of the ioclass to use for reading the data. Either this or product_id
            must be specified.
        dupe_window : numpy.timedelta64
            Time difference between two observations at the same location_id below which
            the second observation will be considered a duplicate. Will be set to
            `np.timedelta64("10", "m")` if `None`. Default: `None`
        dask_scheduler : str, optional
            Dask scheduler to use for parallel processing. Default: "threads"
        **kwargs
            Keyword arguments to pass to the `ioclass` initialization.
        """
        self.ioclass = ioclass
        if isinstance(collections, (str, Path)):
            collections = [collections]

        if isinstance(collections[0], (str, Path)):
            all_subdirs = self._get_subdirs(collections)
            self.collections = [
                CellFileCollection(
                    subdir, ioclass=self.ioclass, ioclass_kws=kwargs
                )
                for subdir in all_subdirs
            ]
        else:
            self.collections = collections

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
            dask.config.set(scheduler=dask_scheduler, memory_limit="20GB")

    @staticmethod
    def _get_subdirs(collections):
        """Return a list of all TERMINAL subdirectories of the given collections.

        Parameters
        ----------
        collections : list of str or Path
            List of paths to collections.
        """
        return [Path(r) for c in collections for (r, d, f) in os.walk(c) if not d]

    @classmethod
    def from_product_id(
            cls,
            collections,
            product_id,
            dupe_window=None,
            dask_scheduler=None
    ):
        """Create a CellFileCollectionStack based on a product_id.

        Returns a CellFileCollectionStack object initialized with an io_class specified
        by `product_id` (case-insensitive).

        Parameters
        ----------
        collections : list of str or CellFileCollection
            A path to a cell file collection or a list of paths to cell file collections,
            or a list of CellFileCollection.
        product_id : str
            ASCAT ID of the cell file collections. Either this or ioclass must be
            specified.
        dupe_window : numpy.timedelta64
            Time difference between two observations at the same location_id below which
            the second observation will be considered a duplicate. Will be set to
            `np.timedelta64("10", "m")` if `None`. Default: `None`
        dask_scheduler : str, optional
            Dask scheduler to use for parallel processing. Will be set to "threads" when
            class is initialized if None. Default: None
        """
        product_id = product_id.upper()
        if product_id in cell_io_catalog:
            io_class = cell_io_catalog[product_id]
        else:
            error_str = f"Product {product_id} not recognized. Valid products are"
            error_str += f" {', '.join(cell_io_catalog.keys())}."
            raise ValueError(error_str)

        return cls(
            collections,
            io_class,
            dupe_window=dupe_window,
            dask_scheduler=dask_scheduler
        )

    def add_collection(self, collections, product_id=None):
        """Add a cell file collection to the stack, based on file path.

        Parameters
        ----------
        collections : str or list of str or CellFileCollection
            Path to the cell file collection to add, or a list of paths.
        product_id : str, optional
            ASCAT ID of the collections to add. Needed if collections is a string or
            list of strings.

        Raises
        ------
        ValueError
            If collections is a string or list of strings and product_id is not given.
        """
        new_idx = len(self.collections)
        if isinstance(collections, (str, Path)):
            collections = [collections]
        if isinstance(collections[0], (str, Path)):
            self.collections.extend(
                CellFileCollection.from_product_id(c, product_id)
                for c in self._get_subdirs(collections)
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

    def read(
            self,
            cell=None,
            location_id=None,
            bbox=None,
            geom=None,
            mask_and_scale=True,
            date_range=None,
            **kwargs
    ):
        """Read data for a cell or location_id.

        Parameters
        ----------
        cell : int
            Cell number to read data for.
        location_id : int
            Location ID to read data for.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates to read data within.
        mask_and_scale : bool, optional
            If True, mask and scale the data according to its `scale_factor` and
            `_FillValue`/`missing_value` before returning. Default: True.
        date_range : tuple of numpy.datetime64, optional
            Start and end dates to read data for.
        **kwargs : dict
            Keyword arguments to pass to the read function of the collection

        Returns
        -------
        xarray.Dataset
            Dataset containing the combined data for the given cell or location_id from
            all the collections in the stack.

        Raises
        ------
        ValueError
            If neither cell nor location_id is given.
        """
        if cell is not None:
            data = self._read_cells(cell, date_range=date_range, **kwargs)
        elif location_id is not None:
            data = self._read_locations(location_id, date_range=date_range, **kwargs)
        elif bbox is not None:
            data = self._read_bbox(bbox, date_range=date_range, **kwargs)
        elif geom is not None:
            data = self._read_geometry(geom, date_range=date_range, **kwargs)
        else:
            raise ValueError("Need to specify either cell, location_id or bbox")

        if data is None:
            return data

        data = data.sortby(["sat_id", "locationIndex", "time"])

        # Deduplicate data
        dupl = np.insert(
            (abs(data["time"].values[1:] -
                 data["time"].values[:-1]) < self.dupe_window),
            0,
            False,
        )
        data = data.sel(obs=~dupl)

        if mask_and_scale:
            return xr.decode_cf(data, mask_and_scale=True)
        return data

    def merge_and_write(
            self,
            out_dir,
            cells=None,
            date_range = None,
            out_cell_size=None,
            processes=8
    ):
        """Merge the data in all the collections by cell, and write each cell to disk.

        Parameters
        ----------
        out_dir : str or Path
            Path to output directory.
        cells : list of int, optional
            Cells to write. If None, write all cells.
        date_range : tuple of numpy.datetime64, optional
            Start and end dates to read data for before writing.
        out_cell_size : tuple, optional
            Size of the output cells in degrees (assumes they are square).
            If None, and the component collections all have the same cell size,
            use that.
        processes : int, optional
            Number of processes to use for parallel processing. Default: 8

        Raises
        ------
        ValueError
            If out_cell_size is None and the component collections do not all have the
            same cell size.
        """
        if out_cell_size is None and self._different_cell_sizes is True:
            raise ValueError("Different cell sizes found, need to specify out_cell_size"
                             + " as argument to write_cells function")
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        cells = self.subcollection_cells(cells, out_cell_size, date_range=date_range)

        if processes < 2:
            for cell in tqdm(cells):
                self._write_single_cell(out_dir, self.ioclass, cell, date_range, out_cell_size)
            return

        with mp.Pool(processes=processes) as pool:
            chunksize_heuristic = (len(cells)//processes)+1
            args_list = [
                (out_dir, self.ioclass, cell, date_range, out_cell_size) for cell in cells
            ]

            for _ in tqdm(
                pool.imap_unordered(
                    self._write_single_cell_wrapper,
                    args_list,
                    chunksize=chunksize_heuristic
                ),
                total=len(cells)
            ):
                pass

    # def append_to_collection_on_disk(
    #         self,
    #         disk_collection,
    #         cells=None,
    #         date_range = None,
    #         out_cell_size=None,
    #         processes=8
    # ):
    #     """Append data from the stack to a collection that already exists on disk.

    #     If there is an existing collection on disk with a locations dimension that includes
    #     all the locations in the stack (or the specified selection from the stack), then
    #     data can be appended to that collection without actually reading it into memory.

    #     This is especially useful if a very large collection is used as an archive and
    #     needs to be expanded with new data periodically.

    #     Currently the only supported file format is netCDF4.

    #     You would probably be better served by appending directly from swath files...

    #     Notes
    #     -----
    #     Should take a disk_collection as argument, as well as all the possible arguments
    #     to self.read(), except for (?)

    #     Can't check for duplicates with already-existing data, need to get a max date from
    #     the disk_collection and ensure that only new data is appended.

    #     """
    #     disk_path = disk_collection.path
    #     # should parallelize this
    #     for cell in cells:
    #         # IF cell is not in the disk collection, then just write a new file (but warn)
    #         disk_data = disk_collection.read(cell=cell)
    #         disk_location_ids = disk_data["location_id"].values
    #         stack_data = self.read(
    #             cell=cell,
    #             mask_and_scale=False,
    #             date_range=date_range,
    #             # out_cell_size=out_cell_size
    #         )
    #         stack_location_ids = stack_data["location_id"].values
    #         output_location_ids = disk_location_ids
    #         cell_fname = disk_collection.ioclass.fn_format.format(cell)


    #         # if there are any new locations in the stack that aren't on disk...
    #         if ~np.all(stack_location_ids == disk_location_ids):
    #             if ~np.all(np.isin(stack_location_ids, disk_location_ids)):
    #                 # output_location_ids will now be all unique location_ids in the
    #                 # stack and disk collections combined

    #                 stack_locationIndex = stack_data["locationIndex"].values
    #                 output_location_ids = np.unique(np.append(stack_location_ids, disk_location_ids))
    #                 all_stack_location_ids = stack_location_ids[stack_locationIndex]

    #                 # recalculate disk locationIndex
    #                 disk_locationIndex = disk_data["locationIndex"].values
    #                 all_disk_location_ids = disk_location_ids[disk_locationIndex]
    #                 new_disk_locationIndex = np.searchsorted(
    #                     output_location_ids,
    #                     all_disk_location_ids
    #                 )

    #             # recalculate stack locationIndex
    #             new_stack_locationIndex = np.searchsorted(
    #                 output_location_ids,
    #                 all_stack_location_ids
    #             )

    #             # these new locationIndex arrays have been calculated with respect to a
    #             # sorted location_id array, but there's no guarantee that the location_id
    #             # array on disk or in the stack is actually sorted.

    #             # get the locations from the stack that aren't already in the disk collection
    #             new_locations = stack_data.isel(
    #                 locations=(~np.isin(stack_location_ids, disk_location_ids))
    #             )
    #             # append the new locations dim data
    #             append_to_netcdf(
    #                 disk_path / cell_fname,
    #                 new_locations,
    #                 "locations"
    #             )

    #         # next need to sort the locations dimension, rewrite the locationIndex,
    #         # and then append the new obs dim data

    #         else:
    #             disk_collection.close()
    #             append_to_netcdf(
    #                 disk_path / cell_fname,
    #                 stack_data,
    #                 "obs"
    #             )

    def _collections_in_date_range(self, date_range):
        if date_range is None:
            return self.collections

        start_date = np.datetime64(date_range[0], "ns")
        end_date = np.datetime64(date_range[1], "ns")
        return [
            coll
            for coll in self.collections
            if (
                    (coll.min_datetime is None and coll.max_datetime is None)
                    or ((coll.min_datetime < end_date) and (coll.max_datetime > start_date))
            )
        ]


    def subcollection_cells(self, cells=None, out_cell_size=None, date_range=None):
        """Get the cells that are covered by all the subcollections. If out_cell_size is
        passed, then it returns the cells in the new cell-scheme that are covered by the
        subcollections.

        Parameters
        ----------
        cells : list of int, optional
            Cells to check. If None, check all cells.
        out_cell_size : int, optional
            The size of the cells in the new cell-scheme.

        Returns
        -------
        set
            Cells covered by all subcollections.
        """
        if out_cell_size is None:
            covered_cells = {
                c
                for coll in self._collections_in_date_range(date_range)
                for c in coll.cells_in_collection
                if (cells is None or c in cells)
            }
            return covered_cells

        # if we assume the collections all have the same grid etc then we only
        # really need to check the first collection
        # But would be better to redesign so we just pull from a grid or something
        # Or return a list for each collection... but then the reading logic should
        # be different.
        new_cells = set()
        for coll in self._collections_in_date_range(date_range):
            coll.create_cell_lookup(out_cell_size)
            # add each new cell to the set if any of the old cells that overlap it are
            # in the cells list
            new_cells.update(new_cell for new_cell, old_cells in coll.cell_lut.items()
                             if np.any(np.isin(old_cells,
                                               (cells or coll.cells_in_collection))))

        return new_cells

    def _read_cells(
        self,
        cells,
        search_cell_size=None,
        valid_gpis=None,
        date_range=None,
        **kwargs
    ):
        """Read data for a list of cells.
        If search_cell_size is passed, then it interprets "cells" with respect to the
        new cell-scheme, determines the cells from the original cell-scheme that are
        covered by the new cells, reads those, and trims the extraneous grid points from
        each new cell.

        Parameters
        ----------
        cells : int or list of int
            The cell or list of cells to read.
        search_cell_size : numeric
            The side length, in degrees, of the cell-scheme that the input passed to
            `cells` refers to. That is, if the data is in a 10 degree grid, but
            `search_cell_size` is 5, then passing `cells=13` will result in the function
            returning data from cell 13 of the 5-degree grid, NOT the 10-degree grid.
        valid_gpis : list of int, optional
            List of gpis to keep. If None, keep all gpis.
        date_range : tuple of numpy.datetime64, optional
            Start and end dates to read data for.
        **kwargs
            Keyword arguments to pass to a `CellFileCollection`'s `read` method.

        Returns
        -------
        xarray.Dataset
            Dataset containing the data for the given cells in indexed ragged array
            format.
        """
        search_cells = cells if isinstance(cells, list) else [cells]
        search_cells = np.unique(search_cells)

        if search_cell_size is not None:
            data = []
            for coll in self._collections_in_date_range(date_range):
                coll.create_cell_lookup(search_cell_size)
                old_cells = [
                    c
                    for cell in search_cells
                    for c in coll.cell_lut[cell]
                    if c in coll.cells_in_collection
                ]

                data.extend(
                    [
                        self._trim_to_gpis(
                            coll.read(
                                cell=cell,
                                mask_and_scale=False,
                                **kwargs
                            ),
                            coll.grid.grid_points_for_cell(cell)[0].compressed(),
                        )
                        for cell in old_cells
                    ]
                )

        else:
            data = [
                coll.read(
                    cell=cell,
                    mask_and_scale=False,
                    **kwargs
                )
                for coll in self._collections_in_date_range(date_range)
                for cell in search_cells
            ]

        data = [ds for ds in data if ds is not None]

        if data == []:
            return None

        locs_merged = xr.combine_nested(
            [self._only_locations(ds) for ds in data], concat_dim="locations"
        )

        _, idxs = np.unique(locs_merged["location_id"].values, return_index=True)

        location_vars = {
            var: locs_merged[var][idxs]
            for var in locs_merged.variables
        }

        location_sorter = np.argsort(location_vars["location_id"].values)

        locs_merged.close()

        merged_ds = xr.combine_nested(
            [self._preprocess(ds, location_vars, location_sorter)
             for ds in data],
            concat_dim="obs",
            data_vars="minimal",
            coords="minimal",
            combine_attrs="drop_conflicts",
        )

        merged_ds = trim_dates(merged_ds, date_range)
        merged_ds = self._trim_to_gpis(merged_ds, valid_gpis)

        return merged_ds

    def _read_bbox(
            self,
            bbox,
            date_range=None,
            **kwargs
    ):
        location_ids = np.unique(
            [coll.grid.get_bbox_grid_points(*bbox)
             for coll in self._collections_in_date_range(date_range)]
        )
        return self._read_locations(location_ids, date_range=date_range, **kwargs)
        # cells = self.ioclass.grid.gpi2cell(location_ids)
        # return self._read_cells(cells, valid_gpis=location_ids, date_range=date_range, **kwargs)

    def _read_geometry(
            self,
            geom,
            date_range=None,
            **kwargs
    ):
        bbox = geom.bounds
        latmin, latmax, lonmin, lonmax = bbox[1], bbox[3], bbox[0], bbox[2]
        bbox_gpis, bbox_lats, bbox_lons = self.ioclass.grid.get_bbox_grid_points(
            latmin,
            latmax,
            lonmin,
            lonmax,
            both=True
        )

        # now that we have the grid points that are within the bounding box, we can
        # check which ones are actually within the geometry
        if len(bbox_gpis) > 0:
            geom_location_ids = [
                gpi
                for gpi, lat, lon in zip(bbox_gpis, bbox_lats, bbox_lons)
                if geom.contains(Point(lon, lat))
            ]
        else:
            geom_location_ids = []

        return self._read_locations(geom_location_ids, date_range=date_range, **kwargs)

    def _read_locations(
        self,
        location_ids,
        date_range=None,
        **kwargs
    ):
        cells = self.ioclass.grid.gpi2cell(location_ids)
        return self._read_cells(cells, valid_gpis=location_ids, date_range=date_range, **kwargs)

    # def _read_locations(
    #     self,
    #     location_ids,
    #     date_range=None,
    #     **kwargs
    # ):
    #     location_ids = (
    #         location_ids
    #         if isinstance(location_ids, (list, np.ndarray))
    #         else [location_ids]
    #     )

    #     # all data here is converted to the SAME GRID CELL SIZE within coll.read()
    #     # before being merged later
    #     data = [d for d in
    #             (coll.read(
    #                 location_id=location_id,
    #                 mask_and_scale=False,
    #                 date_range=date_range,
    #                 **kwargs)
    #              for coll in self._collections_in_date_range(date_range)
    #              for location_id in location_ids)
    #             if d is not None]

    #     # merge all the locations-dimensional data vars into one dataset,
    #     # in order to determine the grid points and indexing for the locations
    #     # dimension of the merged dataset.

    #     # coords="all" is necessary in case one of the coords has nan values
    #     # (e.g. altitude, in the case of ASCAT H129)
    #     locs_merged = xr.combine_nested(
    #         [self._only_locations(ds) for ds in data], concat_dim="locations",
    #         coords="all",
    #     )

    #     _, idxs = np.unique(locs_merged["location_id"].values, return_index=True)

    #     location_vars = {
    #         var: locs_merged[var][idxs]
    #         for var in locs_merged.variables
    #     }

    #     location_sorter = np.argsort(location_vars["location_id"].values)

    #     locs_merged.close()

    #     # merge the data variables
    #     merged_ds = xr.combine_nested(
    #         [self._preprocess(ds, location_vars, location_sorter) for ds in data],
    #         concat_dim="obs",
    #         data_vars="minimal",
    #         coords="minimal",
    #     )

    #     return merged_ds

    def _write_single_cell(self, out_dir, ioclass, cell, date_range, out_cell_size, **kwargs):
        """Write data for a single cell from the stack to disk.

        Parameters
        ----------
        out_dir : str or Path
            Path to output directory.
        ioclass : class
            IO class to use for the writer.
        cell : tuple
            Cell to write.
        date_range : tuple of numpy.datetime64
            Start and end dates to read data for before writing.
        out_cell_size : tuple
            Size of the output cell.
        **kwargs
            Keyword arguments to pass to the ioclass write function.
        """
        data = self.read(
            cell=cell,
            date_range=date_range,
            search_cell_size=out_cell_size,
            mask_and_scale=False
        )
        fname = ioclass.fn_format.format(cell)
        data.attrs["id"] = fname
        writer = ioclass(data)
        writer.write(out_dir / fname, **kwargs)
        data.close()
        writer.close()

    def _write_single_cell_wrapper(self, args):
        """Wrap arguments for `self._write_single_cell` and pass to that method.

        Parameters
        ----------
        args : tuple
            Tuple of arguments to pass to `self._write_single_cell`.
        """
        self._write_single_cell(*args)

    @staticmethod
    def _preprocess(ds, location_vars, location_sorter):
        """Pre-processing to be done on a component dataset so it can be merged with others.

        Assumes `ds` is an indexed ragged array. (Re)-calculates the `locationIndex`
        values for `ds` with respect to the `location_id` variable for the merged
        dataset, which may include locations not present in `ds`.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.
        location_vars : dict
            Dictionary of ordered location variable DataArrays for the merged data.
        location_sorter : numpy.ndarray
            Result of `np.argsort(location_vars["location_id"])`, used to calculate
            the `locationIndex` variable. Calculated outside this function to avoid
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
            locationIndex = location_sorter[np.searchsorted(
                location_vars["location_id"].values,
                ds["location_id"].values[ds["locationIndex"]],
                sorter=location_sorter,
            )]
            ds = ds.drop_dims("locations")

        # if not, we just have a single location, and logic is different
        else:
            locationIndex = location_sorter[np.searchsorted(
                location_vars["location_id"].values,
                np.repeat(ds["location_id"].values, ds["locationIndex"].size),
                sorter=location_sorter,
            )]

        ds["locationIndex"] = ("obs", locationIndex)

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
    def _trim_to_gpis(ds, gpis):
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

        Notes
        -----
        Does NOT trim the locations dimension, only the obs dimension. There may then
        be empty locations in the dataset.
        """
        if ds is None:
            return None
        if gpis is None:
            return ds

        # first trim out any gpis not in the dataset from the gpi list
        gpis = np.intersect1d(gpis, ds["location_id"].values, assume_unique=True)

        # this is a list of the locationIndex values that correspond to the gpis we're keeping
        locations_idx = np.searchsorted(ds["location_id"].values, gpis)
        # this is the indices of the observations that have any of those locationIndex values
        obs_idx = np.in1d(ds["locationIndex"], locations_idx)

        # now we need to figure out what the new locationIndex vector will be once we drop all the other location_ids
        old_locationIndex = ds["locationIndex"].values
        new_locationIndex = np.searchsorted(
            locations_idx,
            old_locationIndex[np.isin(old_locationIndex, locations_idx)]
        )


        # then trim out any gpis in the dataset not in gpis
        ds = ds.isel({"obs": obs_idx, "locations": locations_idx})
        # and add the new locationIndex
        ds["locationIndex"] = ("obs", new_locationIndex)

        return ds

    def close(self):
        """Close all the collections."""
        for collection in self.collections:
            collection.close()

    def __enter__(self):
        """Context manager initialization."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

class CellFileCollection:
    """Collection of grid cell files.

    Represents a collection of grid cell files that live in the same directory,
    and contains methods to read data from them.
    """

    def __init__(self,
                 path,
                 ioclass,
                 ioclass_kws=None,
                 dir_name_format="{date1}_{date2}",
                 dir_date_format="%Y%m%d%H%M%S",
                 ):
        """Initialize."""
        self.path = Path(path)
        self.ioclass = ioclass
        self.grid = self.ioclass.grid
        self.grid_cell_size = self.ioclass.grid_cell_size

        self.max_cell = self.ioclass.max_cell
        self.min_cell = self.ioclass.min_cell
        self.dir_name_format = dir_name_format
        self.dir_date_format = dir_date_format
        self.min_datetime, self.max_datetime = self.date_range

        self.fn_format = self.ioclass.fn_format
        self.previous_cells = None
        self.fid = None
        self.out_cell_size = None
        self.cell_lut = None

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws

    @property
    def date_range(self):
        """Return the start and end date of the collection based on its dir name"""
        try:
            pattern = braces_to_re_groups(self.dir_name_format)
            match = re.match(pattern, self.path.stem)
            start_date = dt.strptime(match.group("date1"), self.dir_date_format)
            end_date = dt.strptime(match.group("date2"), self.dir_date_format)
            return np.datetime64(start_date, "ns"), np.datetime64(end_date, "ns")
        except Exception:
            warnings.warn(
                f"Could not determine date range for collection '{self.path.stem}'"
                " from directory name."
                " Using min/max datetime from files instead."
            )
            return None, None

    @classmethod
    def from_product_id(cls, collections, product_id, ioclass_kws=None):
        """Create a CellFileCollection based on a product_id.

        Returns a CellFileCollection object initialized with an io_class specified
        by `product_id` (case-insensitive).

        Parameters
        ----------
        collections : list of str or Path
            A path to a cell file collection or a list of paths to cell file collections,
            or a list of CellFileCollection.
        product_id : str
            ASCAT ID of the cell file collections.
        ioclass_kws : dict, optional
            Keyword arguments to pass to the ioclass initialization.

        Raises
        ------
        ValueError
            If product_id is not recognized.
        """
        product_id = product_id.upper()
        if product_id in cell_io_catalog:
            io_class = cell_io_catalog[product_id]
        else:
            error_str = f"Product {product_id} not recognized. Valid products are"
            error_str += f" {', '.join(cell_io_catalog.keys())}."
            raise ValueError(error_str)

        return cls(collections, io_class, ioclass_kws=ioclass_kws)

    @property
    def cells_in_collection(self):
        """Return a list of the cells in the collection.

        Returns
        -------
        list of int
            List of cells in the collection.
        """
        return [int(p.stem) for p in self.path.glob("*")]

    def _write_single_cell(self, out_dir, cell, date_range, out_cell_size, ra_type, **kwargs):
        """Write data for a single cell from the stack to disk.

        Parameters
        ----------
        out_dir : str or Path
            Path to output directory.
        cell : tuple
            Cell to write.
        date_range : tuple of numpy.datetime64
            Start and end dates to read data for before writing.
        out_cell_size : tuple
            Size of the output cell.
        **kwargs
            Keyword arguments to pass to the ioclass write function.
        """
        data = self.read(
            cell=cell,
            date_range=date_range,
            # search_cell_size=out_cell_size,
            mask_and_scale=False
        )
        fname = self.ioclass.fn_format.format(cell)
        data.attrs["id"] = fname
        writer = self.ioclass(data)
        writer.write(out_dir / fname, ra_type=ra_type, **kwargs)
        data.close()
        writer.close()

    def _write_single_cell_wrapper(self, args):
        """Wrap arguments for `self._write_single_cell` and pass to that method.

        Parameters
        ----------
        args : tuple
            Tuple of arguments to pass to `self._write_single_cell`.
        """
        self._write_single_cell(*args)

    def to_contiguous(self, out_dir, out_cell_size, processes=8):
        """
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        cells = self.cells_in_collection
        if processes < 2:
            for cell in tqdm(cells):
                self._write_single_cell(out_dir, cell, None, out_cell_size, "contiguous")
            return

        with mp.Pool(processes=processes) as pool:
            chunksize_heuristic = (len(cells)//processes)+1
            args_list = [
                (out_dir, cell, None, out_cell_size, "contiguous") for cell in cells
            ]

            for _ in tqdm(
                pool.imap_unordered(
                    self._write_single_cell_wrapper,
                    args_list,
                    chunksize=chunksize_heuristic
                ),
                total=len(cells)
            ):
                pass


    def read(
            self,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
            mask_and_scale=True,
            date_range=None,
            **kwargs
    ):
        """Read data from the collection for a cell, location_id, or set of coordinates.

        Parameters
        ----------
        cell : int
            Grid cell number to read.
        location_id : int
            Location id.
        coords : tuple
            Tuple of (lat, lon) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        mask_and_scale : bool, optional
            If True, mask and scale the data according to its `scale_factor` and
            `_FillValue`/`missing_value` before returning. Default: True.
        **kwargs : dict
            Keyword arguments passed to the ioclass.

        Returns
        -------
        xarray.Dataset
            Dataset containing the data for the given cell, location_id, or coordinates.

        Raises
        ------
        ValueError
            If neither cell, location_id, nor coords is given.
        """
        kwargs = {**self.ioclass_kws, **kwargs}
        if cell is not None:
            data = self._read_cell(cell, date_range=date_range, **kwargs)
        elif location_id is not None:
            data = self._read_location_id(location_id, date_range=date_range, **kwargs)
        elif coords is not None:
            data = self._read_latlon(coords[0], coords[1], date_range=date_range, **kwargs)
        elif bbox is not None:
            data = self._read_bbox(bbox, date_range=date_range, **kwargs)
        elif geom is not None:
            data = self._read_geometry(geom, date_range=date_range, **kwargs)
        else:
            raise ValueError("Either cell, location_id or coords (lon, lat)"
                             " must be given")

        if mask_and_scale:
            return xr.decode_cf(data, mask_and_scale=True)

        return data

    def create_cell_lookup(self, out_cell_size):
        """Create a lookup table self.cell_lut mapping a new cell-size grid to the existing one.

        Format of the table is a dictionary, where the keys are the cell numbers
        in the new cell-size grid, and the values are the cell numbers in the
        old cell-size grid which the new cell overlaps.

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
            # old_cells and new_cells are arrays of the same length, with each element
            # corresponding to a grid point in self.grid, its value representing the
            # cell that grid point is in.
            self.cell_lut = {new_cell:
                             np.unique(old_cells[np.where(new_cells == new_cell)[0]])
                             for new_cell in np.unique(new_cells)}

    def _open(self, location_id=None, cells=None, grid=None):
        """Open cell file using the given location_id (will open the cell file containing
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
            cells = grid.gpi2cell(location_id)

        if not isinstance(cells, (list, np.ndarray)):
            cells = [cells]

        filenames = [self.get_cell_path(c) for c in np.unique(cells)]

        if len(filenames) == 1:
            filenames = filenames[0]

        if (self.previous_cells is None) or set(self.previous_cells) != set(cells):
            self.close()

            try:
                self.fid = self.ioclass(filenames, **self.ioclass_kws)
            except IOError as e:
                success = False
                self.fid = None
                msg = f"I/O error({e.errno}): {e.strerror}, {filenames}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self.previous_cells = None
            else:
                self.previous_cells = cells

        return success

    def _read_cell(self, cell, date_range=None, **kwargs):
        """Read data from one entire cell.

        Parameters
        ----------
        cell : int
            Cell number.
        **kwargs : dict
            Keyword arguments passed to the ioclass.

        Returns
        -------
        data : xarray.Dataset
            Data for the cell.
        """
        data = None
        if self._open(cells=cell):
            data = self.fid.read(mask_and_scale=False,
                                 date_range=date_range,
                                 **kwargs)

        return data

    def _read_cells(self, cell, date_range=None, **kwargs):
        """Read data from several entire cells.

        Parameters
        ----------
        cell : int
            Cell number.
        **kwargs : dict
            Keyword arguments passed to the ioclass.

        Returns
        -------
        data : xarray.Dataset
            Data for the cell.
        """
        data = None
        if self._open(cells=cell):
            data = self.fid.read(mask_and_scale=False,
                                 date_range=date_range,
                                 **kwargs)

        return data

    def _read_latlon(self, lat, lon, date_range=None, **kwargs):
        """Read data for the nearest grid point to the given coordinate.

        Converts lon/lat pair to a location_id, then reads data for that location_id.

        Parameters
        ----------
        lat : float
            Latitude coordinate.
        lon : float
            Longitude coordinate.
        **kwargs : dict
            Keyword arguments passed to the ioclass `read` method.

        Returns
        -------
        data : xarray.Dataset
            Data at the given coordinates.
        """
        location_id, _ = self.grid.find_nearest_gpi(lon, lat)

        return self._read_location_id(location_id,
                                      date_range=date_range,
                                      **kwargs)

    def _read_bbox(self, bbox, date_range=None, **kwargs):
        location_ids = self.grid.get_bbox_grid_points(*bbox)
        return self._read_location_id(location_ids, date_range=date_range, **kwargs)

    def _read_geometry(self, geom, date_range=None, **kwargs):
        bbox = geom.bounds
        latmin, latmax, lonmin, lonmax = bbox[1], bbox[3], bbox[0], bbox[2]
        bbox_gpis, bbox_lats, bbox_lons = self.grid.get_bbox_grid_points(
            latmin,
            latmax,
            lonmin,
            lonmax,
            both=True
        )

        if len(bbox_gpis) > 0:
            geom_location_ids = [
                gpi
                for gpi, lat, lon in zip(bbox_gpis, bbox_lats, bbox_lons)
                if geom.contains(Point(lon, lat))
            ]
        else:
            geom_location_ids = []

        return self._read_location_id(geom_location_ids, date_range=date_range, **kwargs)

    def _read_location_id(self, location_id, date_range=None, **kwargs):
        """Read data for given grid point.

        Parameters
        ----------
        location_id : int
            Location identifier.
        **kwargs : dict
            Keyword arguments passed to the ioclass `read` method.

        Returns
        -------
        data : xarray.Dataset
            Data for the given grid point.
        """
        data = None

        if self._open(location_id=location_id):
            data = self.fid.read(location_id=location_id,
                                 mask_and_scale=False,
                                 date_range=date_range,
                                 **kwargs)

        return data

    def get_cell_path(self, cell=None, location_id=None):
        """Get path to cell file given cell number or location id.

        Returns a path to a cell file in the collection's directory, whether the file
        exists or not, as long as the cell number or location id is within the grid.

        Parameters
        ----------
        cell : int, optional
            Cell number.
        location_id : int, optional
            Location identifier.

        Returns
        -------
        path : pathlib.Path
            Path to cell file.

        Raises
        ------
        ValueError
            If neither cell nor location_id is given.
        ValueError
            If the given cell number or location_id is not within the grid.
        """
        if location_id is not None:
            cell = self.grid.gpi2cell(location_id)
        elif cell is None:
            raise ValueError("Either location_id or cell must be given")

        if (cell > self.max_cell) or (cell < self.min_cell):
            raise ValueError(f"Cell {cell} is not in grid")

        return self.path / self.fn_format.format(cell)

    def close(self):
        """Close file."""
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    def __enter__(self):
        """Context manager initialization."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()


class SwathFileCollection:
    """ Collection of time-series swath files.

    Parameters
    ----------
    path : str or Path
        Path to the swath file collection.
    ioclass : ascat.read_native.xarray_io.SwathIOBase
        IO class to use for reading the data.
    ioclass_kws : dict, optional
        Keyword arguments to pass to the ioclass initialization. Default: None
    dask_scheduler : str, optional
        Dask scheduler to use for parallel processing in xarray. In testing this just
        made most things slower, but it may be useful in some cases. Default: None


    Attributes
    ----------
    path : Path
        Path to the swath file collection.
    ioclass : class
        IO class to use for reading the data.
    ioclass_kws : dict
        Keyword arguments to pass to the ioclass initialization. May include ioclass
        attributes that will override any that are set in the current ioclass.
    grid : pygeogrids.CellGrid object
        Grid object defining the grid the data is on.
    ts_dtype : numpy.dtype
        Data types to encode the time series data as when writing.
    beams_vars : list of str
        List of names of the variables that have a beams dimension.
    date_format : str
        Format of the date in the filename.
    cell_fn_format : str
        Format for the names of the cell files that will be written out.
    chron_files : function
        Function to search for files in the collection based on their date.
    previous_cell : int or list of int
    fid : ascat.read_native.xarray_io.SwathIOBase object
        The currently open instance of self.ioclass.
    max_buffer_memory_mb : int
        Maximum amount of memory to use for buffering data when stacking to disk.

    Methods
    -------
    """

    def __init__(self,
                 path,
                 ioclass,
                 ioclass_kws=None,
                 dask_scheduler=None,
                 ):
        self.path = Path(path)
        self.ioclass = ioclass

        if ioclass_kws is None:
            self.ioclass_kws = {}
        else:
            self.ioclass_kws = ioclass_kws
            for k in list(ioclass_kws.keys()):
                # allow user to override individual ioclass attributes if they want
                if k in [
                        "fn_pattern",
                        "sf_pattern",
                        "date_format",
                        "grid",
                        "grid_cell_size",
                        "cell_fn_format",
                        "beams_vars",
                        "ts_dtype",
                        "fn_read_fmt",
                        "sf_read_fmt",
                ]:
                    v = ioclass_kws.pop(k)
                    setattr(self.ioclass, k, v)

        self.grid = self.ioclass.grid
        self.ts_dtype = self.ioclass.ts_dtype
        self.beams_vars = self.ioclass.beams_vars

        self.date_format = self.ioclass.date_format
        self.cell_fn_format = self.ioclass.cell_fn_format
        # this should perhaps be optional, the user could provide their own filesearch function
        self.chron_files = self.ioclass.chron_files(self.path)

        self.previous_cell = None
        self._open_fnames = None
        self.fid = None
        self.max_buffer_memory_mb = 6*1024

        self._possible_gpis = None


        if dask_scheduler is not None:
            dask.config.set(scheduler=dask_scheduler)

    @classmethod
    def from_product_id(cls, path, product_id, ioclass_kws=None, dask_scheduler=None):
        """Create a SwathFileCollection based on a product_id.

        Returns a SwathFileCollection object initialized with an io_class specified
        by `product_id` (case-insensitive).

        Parameters
        ----------
        path : str or Path
            Path to the swath file collection.
        product_id : str
            Identifier for the specific ASCAT product the swath files are part of.
        ioclass_kws : dict, optional
            Keyword arguments to pass to the ioclass initialization. Default: None
        dask_scheduler : str, optional
            Dask scheduler to use for parallel processing. Will be set to "threads" when
            class is initialized if None. Default: None

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
            io_class,
            ioclass_kws=ioclass_kws,
            dask_scheduler=dask_scheduler
        )

    def stack(
            self,
            out_dir,
            fnames=None,
            date_range=None,
            mode="w",
            processes=1,
            buffer_memory_mb=None,
            dupe_window=None
    ):
        """Stack swath files and split them into cell timeseries files.

        Reads swath files into memory, stacking their datasets in a buffer until the sum
        of their sizes exceeds self.max_buffer_memory_mb. Then, splits the buffer into
        cell timeseries datasets, writes them to disk in parallel, and clears the
        buffer. This process repeats until all files have been processed, with
        subsequent writes appending new data to existing cell files when appropriate.

        Parameters
        ----------
        out_dir : pathlib.Path
            Output directory to write the stacked files to.
        fnames : list of pathlib.Path, optional
            List of swath filenames to stack.
        date_range : tuple of datetime.datetime
            Start and end dates to read data for before writing.
        mode : str, optional
            Write mode. Default is "w", which will clear all files from out_dir before
            processing. Use "a" to append data to existing files (only if those have
            also been produced by this function).
        processes : int, optional
            Number of processes to use for parallel writing. Default is 1.
        buffer_memory_mb : numeric, optional
            Maximum amount of memory to use for the buffer, in megabytes. Will be set to
            `self.max_buffer_memory_mb` if None. Default is None.
        dupe_window : numpy.timedelta64, optional
            Time window within which duplicate observations will be removed. Default is
            `None`.

        Raises
        ------
        ValueError
            If mode is not "w" or "a".
        """
        if mode == "w":
            if process_warnings == True:
                input(f"Calling ragged_array_ts.stack with mode='w' will clear all files"
                    f" from {out_dir}.\nPress enter to continue, or ctrl+c to cancel. ")
            for f in out_dir.glob("*.nc"):
                f.unlink()
        elif mode == "a":
            if process_warnings == True:
                input(f"Calling ragged_array_ts.stack with mode='a' will append data to"
                    f" existing files in `out_dir`:\n{out_dir}\nIf it is important to"
                    " preserve the data in this directory it its current state, please"
                    " save a backup elsewhere before continuing, or choose a different"
                    " 'out_dir' for this function and combine the results afterwards"
                    " using a CellFileCollectionStack.\nPress enter to continue, or"
                    " ctrl+c to cancel. ")
        else:
            raise ValueError(f"Invalid mode {mode} for 'SwathFileCollection.stack'."
                             " Valid modes are 'w' and 'a'.")

        if date_range is not None:
            start_dt, end_dt = date_range
            range_fnames = self.get_filenames(start_dt=start_dt, end_dt=end_dt)
            if fnames is not None:
                fnames = list(set(fnames).intersection(set(range_fnames)))
                warnings.warn("Both `fnames` and `date_range` were passed. `fnames`"
                              " will be filtered to only include files within the"
                              " passed `date_range`.\n")
            else:
                fnames = range_fnames
        elif fnames is None:
            fnames = self.get_filenames()

        self.max_buffer_memory_mb = buffer_memory_mb or self.max_buffer_memory_mb
        buffer = []
        buffer_size = 0
        # process = psutil.Process()
        total_swaths = len(fnames)
        for iteration, f in enumerate(fnames):
            self._open(f)
            ds = self.fid.read(mask_and_scale=False).load()
            # buffer_size  = process.memory_info().rss / 1e6
            print("Filling swaths buffer..."
                  f" {buffer_size:.2f}MB/{self.max_buffer_memory_mb:.2f}MB",
                  end="\r")
            buffer_size += ds.nbytes / 1e6
            if buffer_size > self.max_buffer_memory_mb:
                out_dir.mkdir(parents=True, exist_ok=True)
                print("\nBuffer full. Processing swath files..."
                      "                     ",
                      end="\r")
                combined_ds = self.process(
                    xr.combine_nested(
                        buffer,
                        concat_dim="obs",
                        combine_attrs=self.ioclass.combine_attributes
                    )
                )
                print(f"Processed {iteration}/{total_swaths} swath files."
                      " Dumping to cell files...")
                self._parallel_write_cells(
                    combined_ds,
                    out_dir,
                    processes=processes,
                    dupe_window=dupe_window
                )
                print("Finished dumping buffer to cell files.")
                combined_ds.close()
                buffer = []
                buffer_size = ds.nbytes / 1e6
            buffer.append(ds)

        if len(buffer) > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processed {total_swaths}/{total_swaths} swath files."
                  " Processing remaining buffer data...",
                  end="\r")
            combined_ds = self.process(
                xr.combine_nested(
                    buffer,
                    concat_dim="obs",
                    combine_attrs=self.ioclass.combine_attributes,
                )
            )
            print(f"Processed {total_swaths}/{total_swaths} swath files."
                  " Dumping to cell files...")
            self._parallel_write_cells(
                combined_ds,
                out_dir,
                processes=processes,
                dupe_window=dupe_window
            )
            total_cell_files = len(list(out_dir.glob("*.nc")))
            print(f"Finished stacking {total_swaths} swath files to {total_cell_files}"
                  " cell files.")
            combined_ds.close()

    def process(self, data):
        """Process a stacked dataset of swath data into a format that is ready to be
        split into cell timeseries datasets, and return the processed dataset.

        Parameters
        ----------
        data : xarray.Dataset
            Stacked dataset to process.
        """
        # if there are kwargs, use them instead of self.ioclass_kws
        beam_idx = {"for": 0, "mid": 1, "aft": 2}

        # if any beam has backscatter data for a record, the record is valid. Drop
        # observations that don't have any backscatter data.
        if "backscatter" in data.variables:
            if data["obs"].size > 0:
                data = data.sel(
                    obs=~np.all(
                        data["backscatter"] == data["backscatter"].attrs["missing_value"],
                        # np.isnan(data["backscatter"]),
                        axis=1
                    )
                )

        # break the beams dimension variables into separate variables for
        # the fore, mid, and aft beams
        if data["obs"].size > 0:
            if "beams" in data.dims:
                for var in self.ts_dtype.names:
                    if var[:-4] in self.beams_vars:
                        ending = var[-3:]
                        data[var] = data.sel(beams=beam_idx[ending])[var[:-4]]
                    # if var in data.variables:
                    #     if data[var].dtype != self.ts_dtype[var]:
                    #         data[var] = data[var].astype(self.ts_dtype[var])
                        # data[var].attrs["dtype"] = self.ts_dtype[var]
                        # data[var].attrs["_FillValue"] = dtype_to_nan[self.ts_dtype[var]]
                        # data[var].attrs["missing_value"] = data[var].attrs["_FillValue"]

                # drop the variables on the beams dimension
                data = data.drop_dims("beams")

        # Find which cell each observation belongs to, and assign it as a coordinate.
        data = data.assign_coords(
            {"cell": ("obs", self.grid.gpi2cell(data["location_id"].values))}
        )
        # Must set an index for the cell coordinate so that we can select by it later.
        data = data.set_xindex("cell")

        return data

    def read(
            self,
            date_range,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
            **kwargs
    ):
        """Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it.

        If the time range is large, this can be slow. It may make more sense to
        convert to cell files first and access that data from disk using
        a `CellFileCollection` or `CellFileCollectionStack`.

        Parameters
        ----------
        date_range : tuple of datetime.datetime
            Start and end dates.
        cell : int or list of int, optional
            Grid cell number to read.
        location_id : int, optional
            Location id.
        coords : tuple, optional
            Tuple of (lat, lon) coordinates.
        bbox : tuple, optional
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        geometry : shapely.geometry, optional
            Geometry object; use to read data that intersects the geometry.
        """
        start_dt, end_dt = date_range
        fnames = self.get_filenames(
            start_dt,
            end_dt,
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            geom=geom,
        )
        if cell is not None:
            data = self._read_cell(fnames, cell, **kwargs)
        elif location_id is not None:
            data = self._read_location_id(fnames, location_id, **kwargs)
        elif coords is not None:
            data = self._read_latlon(fnames, coords[0], coords[1], **kwargs)
        elif bbox is not None:
            data = self._read_bbox(fnames, bbox, **kwargs)
        elif geom is not None:
            data = self._read_geometry(fnames, geom, **kwargs)
        elif self._open(fnames):
            data = self.fid.read(**kwargs)
        else:
            raise ValueError(f"No swath files found in directory {self.path} for the"
                             f" passed date range: {start_dt} - {end_dt}")

        data = data.set_xindex("time")
        start_dt = np.datetime64(start_dt)
        end_dt = np.datetime64(end_dt)
        data = data.isel(obs=(data.time >= start_dt) & (data.time < end_dt))
        data = data.reset_index("time", drop=False)
        return data

    def close(self):
        """Close collection and constituent xarray datasets."""
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    def get_filenames(
            self,
            start_dt=None,
            end_dt=None,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
    ):
        """Get filenames for the given time range.

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

        Raises
        ------
        NotImplementedError
            If the ioclass does not have a file search method named `chron_files`.
        """
        if start_dt is None and end_dt is None:
            fnames = list(self.path.glob("**/*.nc"))
        elif self.chron_files:
            fnames = self.chron_files.search_period(
                start_dt,
                end_dt,
                date_field_fmt=self.date_format,
                end_inclusive=False
            )
        else:
            raise NotImplementedError("File search not implemented for this product."
                                      " Check if fn_pattern and sf_pattern are defined"
                                      f" in ioclass {self.ioclass.__name__}")


        # now check for location_ids
        location_ids = None
        if cell is not None:
            location_ids = self.grid.grid_points_for_cell(cell)
        elif location_id is not None:
            location_ids = [location_id]
        elif coords is not None:
            location_ids = self._location_id_from_coords(*coords)
        elif bbox is not None:
            location_ids = self._location_id_from_bbox(bbox)
        elif geom is not None:
            location_ids = self._location_id_from_geometry(geom)

        if location_ids is not None:
            gpi_lookup = np.zeros(self.grid.gpis.max()+1, dtype=bool)
            gpi_lookup[location_ids] = 1
            fnames = [f
                    for f in fnames
                    if self._open(f) and self.fid.contains_location_ids(lookup_vector=gpi_lookup)]

        return fnames

    def swath_data_generator(
            self,
            start_dt=None,
            end_dt=None,
            cell=None,
            location_id=None,
            coords=None,
            bbox=None,
            geom=None,
    ):
        """Return a generator producing the data for each requested swath file.

        Parameters
        ----------
        start_dt : datetime.datetime
            Start time.
        end_dt : datetime.datetime
            End time.
        cell : int
            Grid cell number to select.
        location_id : int
            Location id.
        coords : tuple
            Tuple of (lat, lon) coordinates.
        bbox : tuple
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        geom : shapely.geometry
            Geometry object; use to select data that intersects the geometry.

        Yields
        ------
        start_timestamp : numpy.datetime64
            Sensing start time of the swath file.
        end_timestamp : numpy.datetime64
            Sensing end time of the swath file.
        sat : str
            Satellite name.
        data : xarray.Dataset
            Dataset for each swath file intersecting the requested extent.
        """
        fnames = self.get_filenames(
            start_dt=start_dt,
            end_dt=end_dt,
            cell=cell,
            location_id=location_id,
            coords=coords,
            bbox=bbox,
            geom=geom,
        )

        for f in fnames:
            self._open(f)
            data = self.fid.read()
            start_timestamp = np.datetime64(data.attrs["sensing_start_time_utc"], "ns")
            end_timestamp = np.datetime64(data.attrs["sensing_end_time_utc"], "ns")
            spacecraft = Spacecraft(data.attrs["spacecraft"])
            sat = spacecraft.satellite

            yield start_timestamp, end_timestamp, sat, data

    def _open(self, fnames):
        """Open swath files

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
                self.fid = self.ioclass(fnames, **self.ioclass_kws)
            except IOError as e:
                success = False
                self.fid = None
                msg = f"I/O error({e.errno}): {e.strerror}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        # else:
        #     # handling for adding new extra data to the open dataset
        #     pass

        return success

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

    def _write_cell_ds(self, ds, out_dir, cell, dupe_window=None):
        """Write a cell dataset to a file."""
        fname = self.ioclass.cell_fn_format.format(cell)

        out_ds = self._cell_data_as_indexed_ra(ds, cell, dupe_window)
        # out_ds = xr.decode_cf(out_ds, mask_and_scale=False)
        # for var in self.ts_dtype.names:
        #     if var in out_ds.variables:
        #         if out_ds[var].dtype != self.ts_dtype[var]:
        #             out_ds[var] = out_ds[var].astype(self.ts_dtype[var])

        out_ds = xr.decode_cf(out_ds, mask_and_scale=True)
        out_ds.attrs["id"] = fname

        writer = self.ioclass(out_ds)
        writer.write(out_dir / fname, mode="a")
        writer.close()

    def _write_cell_ds_wrapper(self, args):
        """Wrapper for write_cell_ds to allow parallel processing with imap."""
        self._write_cell_ds(*args)

    def _parallel_write_cells(self, ds, out_dir, processes=8, dupe_window=None):
        """Write a stacked dataset to a set of cell files in parallel."""
        cells = np.unique(ds.cell.values)
        args = [
            (
                ds.isel(obs=np.where(ds.cell.values == cell)[0]),
                out_dir,
                cell,
                dupe_window
            )
            for cell in cells
        ]

        if processes > 1:
            chunksize_heuristic = (len(cells)//processes)+1
            with mp.Pool(processes=processes) as pool:
                for _ in tqdm(pool.imap_unordered(self._write_cell_ds_wrapper,
                                                  args,
                                                  chunksize=chunksize_heuristic),
                              total=len(cells)):
                    pass
        else:
            for arguments in tqdm(args):
                self._write_cell_ds(*arguments)

    def _location_id_from_coords(self, lat, lon):
        """Get the location_id for a given lat/lon pair."""
        return self.grid.find_nearest_gpi(lon, lat)[0]

    def _location_id_from_bbox(self, bbox):
        """Get the location_ids for a given bounding box.
        (latmin, latmax, lonmin, lonmax)"""
        return self.grid.get_bbox_grid_points(*bbox)


    def _location_id_from_geometry(self, geom):
        """Get the location_ids for a given geometry."""
        bbox = geom.bounds
        latmin, latmax, lonmin, lonmax = bbox[1], bbox[3], bbox[0], bbox[2]
        bbox_gpis, bbox_lats, bbox_lons = self.grid.get_bbox_grid_points(
            latmin,
            latmax,
            lonmin,
            lonmax,
            both=True
        )
        # now that we have the grid points that are within the bounding box, we can
        # check which ones are actually within the geometry
        if len(bbox_gpis) > 0:
            geom_location_ids = [
                gpi
                for gpi, lat, lon in zip(bbox_gpis, bbox_lats, bbox_lons)
                if geom.contains(Point(lon, lat))
            ]
        else:
            geom_location_ids = []

        return geom_location_ids

    def _read_cell(self, fnames, cell, **kwargs):
        """Read data from the entire cell."""
        # if there are kwargs, use them instead of self.ioclass_kws

        data = None
        if self._open(fnames):
            data = self.fid.read(cell=cell, **kwargs)

        return data

    def _read_latlon(self, fnames, lat, lon, **kwargs):
        """Reading data for given longitude and latitude coordinate.

        Parameters
        ----------
        lat : float
            Latitude coordinate.
        lon : float
            Longitude coordinate.

        Returns
        -------
        data : dict of values
            data record.
        """
        location_id = self._location_id_from_coords(lat, lon)

        return self._read_location_id(fnames, location_id, **kwargs)

    def _read_bbox(self, fnames, bbox, **kwargs):
        """Reading data for given bounding box.

        Parameters
        ----------
        bbox : tuple or list
            Bounding box coordinates (latmin, latmax, lonmin, lonmax).
        """
        location_ids = self._location_id_from_bbox(bbox)
        return self._read_location_id(fnames, location_ids, **kwargs)

    def _read_geometry(self, fnames, geom, **kwargs):
        """Reading data for given geometry.

        Assumes geometry and grid are in the same coordinate system.

        Parameters
        ----------
        geometry : shapely.geometry
            Geometry object.
        """
        location_ids = self._location_id_from_geometry(geom)
        return self._read_location_id(fnames, location_ids, **kwargs)

    def _read_location_id(self, fnames, location_id, **kwargs):
        """Read data for given grid point.

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

        if self._open(fnames):
            gpi_lookup = np.zeros(self.grid.gpis.max()+1, dtype=bool)
            gpi_lookup[location_id] = 1
            data = self.fid.read(location_id=location_id, lookup_vector=gpi_lookup, **kwargs)

        return data


    def __enter__(self):
        """Context manager initialization."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

def braces_to_re_groups(string):
    """
    Convert braces to character patterns defining regular expression groups.
    If any group name is repeated in the template string, a backreference
    is used for subsequent appearances.

    Parameters
    ----------
    string : str
        String with braces.

    Returns
    -------
    string : str
        String with regular expression groups.

    Examples
    --------
    >>> braces_to_re_groups("{year}-{month}-{day}")
    "(?P<year>.+)-(?P<month>.+)-(?P<day>.+)"
    >>> braces_to_re_groups("{year}-{month}-{day}_{year}-{month}-{day2}")
    "(?P<year>.+)-(?P<month>.+)-(?P<day>.+)_(?P=year)-(?P=month)-(?P<day2>.+)"
    """

    pattern = re.compile(r"{(.+?)}")
    seen = set()
    parts = pattern.split(string)

    for i in range(1, len(parts), 2):
        content = parts[i]

        if content in seen:
            parts[i] = f"(?P={content})"
        else:
            parts[i] = f"(?P<{content}>.+)"
            seen.add(content)

    return "".join(parts)

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
        return self.locations["location_id"]

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
        pos = self.locations["location_id"] == location_id

        if not pos.any():
            print(f"location_id not found: {location_id}")
            data = None
        else:
            sel = self.locations[pos]
            i = sel.index.values[0]

            if self.cache:
                j = self.dataset["locationIndex"].values == i
                data = self.dataset.sel(locations=i, time=j)
            else:
                with xr.open_dataset(
                        self.filename,
                        mask_and_scale=self.mask_and_scale) as dataset:
                    j = dataset["locationIndex"].values == i
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
        return self.locations["location_id"]

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
        pos = self.locations["location_id"] == location_id

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
