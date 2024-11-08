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

from typing import Iterable

import numpy as np
import xarray as xr

from ascat.read_native.grid_registry import PyGeoGridRegistry
from ascat.utils import get_grid_gpis
from pygeogrids import BasicGrid, CellGrid

registry = PyGeoGridRegistry()


@xr.register_dataset_accessor("pgg")
class PyGeoGriddedArrayAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._grid = None

    @property
    def grid(self):
        if self._grid is None:
            if grid_name := self._obj.attrs.get("grid_name"):
                self._grid = registry.get(grid_name)
        return self._grid

    def set_grid_name(
        self,
        grid_name: str,
        grid: BasicGrid = None,
    ):
        try:
            self._grid = registry.get(grid_name)
        except KeyError:
            if grid is not None:
                registry.register(grid_name, lambda: grid)
            else:
                raise ValueError(
                    f"Grid {grid_name} is not registered."
                    " Please pass a grid object to the `grid` argument."
                )
        self._obj.attrs["grid_name"] = grid_name

    def sel_bbox(self, bbox: Iterable[float]):
        gpis, lookup_vector = get_grid_gpis(grid=self.grid, bbox=bbox)
        return self._obj.pgg.sel_gpi(gpis)

    def sel_coords(
        self, coords: Iterable[Iterable[float]], max_coord_dist: float = np.inf
    ):
        gpis, lookup_vector = get_grid_gpis(
            grid=self.grid, coords=coords, max_coord_dist=max_coord_dist
        )
        return self._obj.pgg.sel_gpi(gpis)

    def sel_cells(self, cells: Iterable[float]):
        assert isinstance(self._grid, CellGrid)
        gpis, lookup_vector = get_grid_gpis(grid=self._grid, cell=cells)
        return self._obj.pgg.sel_gpi(gpis)

    def sel_gpis(
        self,
        gpis: Iterable[int] = None,
        lookup_vector: np.ndarray = None,
        gpi_var: str = "location_id",
    ) -> xr.Dataset:
        return self._obj.cf_geom.sel_instances(
            instance_vals=gpis,
            instance_lookup_vector=lookup_vector,
            instance_uid=gpi_var,
        )


@xr.register_dataset_accessor("cf_geom")
class CFDiscreteGeometryAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._arr_type = None
        self._sample_dimension = None
        self._instance_dimension = None
        self._count_var = None
        self._index_var = None
        self.array_type

    @property
    def array_type(self):
        try:
            self._arr_type = self._obj.raga.array_type
            return self._arr_type
        except ValueError:
            pass
        try:
            self._arr_type = self._obj.parr.array_type
            return self._arr_type
        except ValueError:
            pass
        try:
            self._arr_type = self._obj.ormu.array_type
            return self._arr_type
        except ValueError:
            raise ValueError("CF discrete geometry type could not be determined.")

    def sel_instances(
        self,
        instance_vals: Iterable[int] = None,
        instance_lookup_vector: np.ndarray = None,
        instance_uid: str = "location_id",
    ):
        if self.array_type == "indexed":
            return self._obj.raga.sel_instances(
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
            )
        if self.array_type == "contiguous":
            return self._obj.raga.sel_instances(
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
            )
        # if self.array_type == "orthomulti":
        #     return self._obj.ormu.sel_instances(instance_vals=instance_vals,
        #                                         instance_lookup_vector=instance_lookup_vector,
        #                                         instance_uid=instance_uid)
        if self.array_type == "point":
            return self._obj.parr.sel_instances(
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
                instance_uid=instance_uid,
            )

    def to_point_array(self):
        if self.array_type == "indexed":
            return self._obj.raga.to_point_array()
        if self.array_type == "contiguous":
            return self._obj.raga.to_point_array()
        if self.array_type == "orthomulti":
            return self._obj.ormu.to_point_array()
        if self.array_type == "point":
            return self._obj

    def to_indexed_ragged(self):
        if self.array_type == "indexed":
            return self._obj
        if self.array_type == "contiguous":
            return self._obj.raga.to_indexed_ragged()
        if self.array_type == "orthomulti":
            return self._obj.ormu.to_indexed_ragged()
        if self.array_type == "point":
            return self._obj.parr.to_indexed_ragged()

    def to_contiguous_ragged(self):
        if self.array_type == "contiguous":
            return self._obj
        if self.array_type == "indexed":
            return self._obj.raga.to_contiguous_ragged()
        if self.array_type == "orthomulti":
            return self._obj.ormu.to_contiguous_ragged()
        if self.array_type == "point":
            return self._obj.parr.to_contiguous_ragged()

    def to_orthomulti(self):
        if self.array_type == "orthomulti":
            return self._obj
        if self.array_type == "indexed":
            return self._obj.raga.to_orthomulti()
        if self.array_type == "contiguous":
            return self._obj.raga.to_orthomulti()
        if self.array_type == "point":
            return self._obj.parr.to_orthomulti()


@xr.register_dataset_accessor("parr")
class PointArrayAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._ra_type = None
        self._sample_dimension = None
        self.array_type

    @property
    def array_type(self) -> str:
        print("hello typeraer")
        if self._ra_type is None:
            if self._obj.attrs["featureType"] == "point":
                self._ra_type = "point"
                self._sample_dimension = list(self._obj.dims)[0]
            else:
                raise ValueError(
                    "Dataset is not a point array"
                    "(should have featureType='point' in attributes)."
                )
        return self._ra_type

    def sel_instances(
        self,
        instance_vals: Iterable[int] = None,
        instance_lookup_vector: np.ndarray = None,
        instance_uid: str = "location_id",
    ):
        ds = self._obj
        print("hiiii")
        print(ds)
        return self._select_instances(
            ds,
            self._sample_dimension,
            instance_vals,
            instance_lookup_vector,
            instance_uid,
        )

    def to_indexed_ragged(
        self,
        instance_dim: str = "locations",
        instance_uid: str = "location_id",
        index_var: str = "locationIndex",
        instance_vars: Iterable[str] = None,
        set_coords: Iterable[str] = None,
    ) -> xr.Dataset:
        return self._point_to_indexed(
            self._obj,
            self._sample_dimension,
            instance_dim,
            instance_uid,
            index_var,
            instance_vars,
            set_coords,
        )

    def to_contiguous_ragged(
        self,
        instance_dim: str = "locations",
        instance_uid: str = "location_id",
        count_var: str = "row_size",
        instance_vars: Iterable[str] = None,
        set_coords: Iterable[str] = None,
    ) -> xr.Dataset:
        return self._point_to_contiguous(
            self._obj, instance_dim, instance_uid, count_var, instance_vars, set_coords
        )

    def to_orthomulti(self):
        pass

    @staticmethod
    def _select_instances(
        ds: xr.Dataset,
        sample_dim: str,
        instance_vals: Iterable[int] = None,
        instance_lookup_vector: np.ndarray = None,
        instance_uid: str = "location_id",
    ) -> xr.Dataset:
        if instance_lookup_vector is not None:
            sample_idx = instance_lookup_vector[ds[instance_uid]]
            return ds.sel({sample_dim: sample_idx})
        elif instance_vals is not None and len(instance_vals) > 0:
            sample_idx = np.isin(ds[instance_uid], instance_vals)
            return ds.sel({sample_dim: sample_idx})
        return None

    @staticmethod
    def _point_to_indexed(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        instance_uid: str,
        index_var: str = "locationIndex",
        instance_vars: Iterable[str] = None,
        set_coords: Iterable[str] = None,
    ) -> xr.Dataset:
        instance_id, unique_index_1d, instanceIndex = np.unique(
            ds[instance_uid], return_index=True, return_inverse=True
        )
        ds[index_var] = (sample_dim, instanceIndex)

        potential_instance_vars = [
            "lon",
            "lat",
            "alt",
            "longitude",
            "latitude",
            "altitude",
            "location_description",
        ]
        instance_vars = instance_vars or potential_instance_vars
        instance_vars = [instance_uid] + instance_vars

        potential_set_coords = [
            instance_uid,
            "lon",
            "lat",
            "alt",
            "longitude",
            "latitude",
            "altitude",
        ]
        set_coords = set_coords or potential_set_coords

        for var in instance_vars:
            if var in ds:
                ds = ds.assign(
                    {var: (instance_dim, ds[var][unique_index_1d].data, ds[var].attrs)}
                )
                if var in set_coords:
                    ds = ds.set_coords(var)
        ds = ds.assign_attrs({"featureType": "timeSeries"})
        return ds

    @staticmethod
    def _point_to_contiguous(
        ds: xr.Dataset,
        instance_dim: str,
        instance_uid: str,
        count_var: str = "row_size",
        instance_vars: Iterable[str] = None,
        set_coords: Iterable[str] = None,
    ) -> xr.Dataset:
        ds = ds.sortby(["location_id", "time"])
        instance_id, unique_index_1d, row_size = np.unique(
            ds[instance_uid], return_index=True, return_counts=True
        )

        ds[count_var] = ("locations", row_size)

        potential_instance_vars = [
            "lon",
            "lat",
            "alt",
            "longitude",
            "latitude",
            "altitude",
            "location_description",
        ]
        instance_vars = instance_vars or potential_instance_vars
        instance_vars = [instance_uid] + instance_vars

        potential_set_coords = [
            instance_uid,
            "lon",
            "lat",
            "alt",
            "longitude",
            "latitude",
            "altitude",
        ]
        set_coords = set_coords or potential_set_coords

        for var in instance_vars:
            if var in ds:
                ds[var] = (instance_dim, ds[var][unique_index_1d].data, ds[var].attrs)
                if var in set_coords:
                    ds = ds.set_coords(var)
        ds = ds.assign_attrs({"featureType": "timeSeries"})
        return ds


@xr.register_dataset_accessor("raga")
class RaggedArrayAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._ra_type = None
        self._sample_dimension = None
        self._instance_dimension = None
        self._count_var = None
        self._index_var = None
        self._timeseries_id = None
        self.array_type

    @property
    def array_type(self):
        if self._ra_type is not None:
            return self._ra_type

        ds = self._obj
        for v in ds.variables:
            if "instance_dimension" in ds[v].attrs:
                self._ra_type = "indexed"
                self._index_var = v
                self._instance_dimension = ds[v].attrs["instance_dimension"]
                self._sample_dimension = ds[v].dims[0]
                return self._ra_type

            if "sample_dimension" in ds[v].attrs:
                self._ra_type = "contiguous"
                self._count_var = v
                self._sample_dimension = ds[v].attrs["sample_dimension"]
                self._instance_dimension = ds[v].dims[0]
                return self._ra_type

        raise ValueError("Ragged array type could not be determined.")

    @property
    def timeseries_id(self):
        if self._timeseries_id is not None:
            return self._timeseries_id
        for v in self._obj.variables:
            if cf_role := self._obj[v].attrs.get("cf_role"):
                if cf_role == "timeseries_id":
                    self._timeseries_id = v
                    return self.timeseries_id
        raise ValueError(
            "Timeseries ID could not be determined from dataset attributes."
        )

    def to_indexed_ragged(self, index_var: str = "locationIndex"):
        if self.array_type == "indexed":
            return self._obj
        if self.array_type == "contiguous":
            if self._index_var is None:
                self._index_var = index_var
            return self._contiguous_to_indexed(
                self._obj,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
                self._index_var,
            )

    def to_contiguous_ragged(
        self, count_var: str = "row_size", sort_vars: Iterable[str] = None
    ):
        if self.array_type == "contiguous":
            return self._obj
        if self.array_type == "indexed":
            if self._count_var is None:
                self._count_var = count_var
            return self._indexed_to_contiguous(
                self._obj,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
                self._index_var,
                sort_vars=sort_vars,
            )

    def to_point_array(self):
        if self.array_type == "indexed":
            return self._indexed_to_point(
                self._obj,
                self._sample_dimension,
                self._instance_dimension,
                self._index_var,
            )
        if self.array_type == "contiguous":
            return self._contiguous_to_point(
                self._obj,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
            )

    def sel_instances(
        self,
        instance_vals: Iterable[int] = None,
        instance_lookup_vector: np.ndarray = None,
    ):
        if self.array_type == "indexed":
            # convert to point array, select there, convert back\
            ds = self.to_point_array()
            print(ds)
            instances = ds.parr.sel_instances(
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
                instance_uid=self.timeseries_id,
            )
            return instances.parr.to_indexed_ragged()

        if self.array_type == "contiguous":
            return self._select_instances_contiguous(
                self._obj,
                self._sample_dimension,
                self._instance_dimension,
                self.timeseries_id,
                self._count_var,
                self._index_var,
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
            )

    @staticmethod
    def _select_instances_contiguous(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        instance_uid: str,
        count_var: str,
        index_var: str,
        instance_vals: Iterable[int] = None,
        instance_lookup_vector: np.ndarray = None,
    ) -> xr.Dataset:
        # In this case we /can/ use the lookup vector but it will be slower
        if instance_vals is None or len(instance_vals) == 0:
            if instance_lookup_vector is not None:
                sample_instances = np.repeat(ds[instance_dim], ds[count_var])
                sample_instance_ids = np.repeat(ds[instance_uid], ds[count_var])
                sample_idxs = instance_lookup_vector[sample_instance_ids]
                instances_idxs = np.aggregate(sample_instances, sample_idxs, func="any")
                return ds.sel({sample_dim: sample_idxs, instance_dim: instances_idxs})
            else:
                sample_idxs = []
                instances_idxs = []

        if len(instance_vals) == 1:
            instances_idx = np.where(ds[instance_uid] == instance_vals[0])[0][0]
            sample_start = int(
                ds[count_var].isel({instance_dim: slice(0, instances_idx)}).sum().values
            )
            sample_end = int(
                sample_start + ds[count_var].isel({instance_dim: instances_idx}).values
            )

            return ds.isel(
                {
                    sample_dim: slice(sample_start, sample_end),
                    instance_dim: instances_idx,
                }
            )
        else:
            instances_idxs = np.where(np.isin(ds[instance_uid], instance_vals))[0]

        if instances_idxs.size > 0:
            sample_starts = [
                int(ds[count_var].isel({instance_dim: slice(0, i)}).sum().values)
                for i in instances_idxs
            ]
            sample_ends = [
                int(start + ds[count_var].isel({instance_dim: i}).values)
                for start, i in zip(sample_starts, instances_idxs)
            ]
            sample_idxs = np.concatenate(
                [range(start, end) for start, end in zip(sample_starts, sample_ends)]
            )
            # locations_idxs = [i for i in locations_idxs]
        else:
            sample_idxs = []
            instances_idxs = []

        ds = ds.isel({sample_dim: sample_idxs, instance_dim: instances_idxs})
        return ds

    @staticmethod
    def _contiguous_to_indexed(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        count_var: str,
        index_var: str,
    ) -> xr.Dataset:
        """
        Convert a contiguous ragged array dataset to an indexed ragged array dataset.
        """
        row_size = np.where(ds[count_var].data > 0, ds[count_var].data, 0)

        locationIndex = np.repeat(np.arange(row_size.size), row_size)

        ds = ds.assign(
            {
                index_var: (
                    sample_dim,
                    locationIndex,
                    {"instance_dimension": instance_dim},
                )
            }
        ).drop_vars([count_var])

        # put locationIndex as first var
        ds = ds[[index_var] + [var for var in ds.variables if var != index_var]]

        return ds

    @staticmethod
    def _indexed_to_contiguous(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        count_var: str,
        index_var: str,
        sort_vars: Iterable[str] = None,
    ) -> xr.Dataset:
        """
        Convert an indexed ragged array dataset to a contiguous ragged array dataset
        """
        # if not ds.chunks:
        #     ds = ds.chunk({"obs": 1_000_000})
        sort_vars = sort_vars or [sample_dim]

        ds = ds.sortby([index_var, *sort_vars])
        idxs, sizes = np.unique(ds[index_var], return_counts=True)

        row_size = np.zeros_like(ds[instance_dim].data)
        row_size[idxs] = sizes
        ds = ds.assign(
            {count_var: (instance_dim, row_size, {"sample_dimension": sample_dim})}
        ).drop_vars([index_var])

        return ds

    @staticmethod
    def _indexed_to_point(
        ds: xr.Dataset, sample_dim: str, instance_dim: str, index_var: str
    ):
        instance_vars = [var for var in ds.variables if instance_dim in ds[var].dims]
        for instance_var in instance_vars:
            ds = ds.assign(
                {
                    instance_var: (
                        sample_dim,
                        ds[instance_var][ds[index_var]].data,
                        ds[instance_var].attrs,
                    )
                }
            )
        ds = ds.drop_vars([index_var]).assign_attrs({"featureType": "point"})
        return ds

    @staticmethod
    def _contiguous_to_point(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        count_var: str,
    ):
        """Convert a ragged array dataset to a Point Array.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset with only the time series variables.
        """
        row_size = ds[count_var].data
        ds = ds.drop_vars(count_var)
        instance_vars = [var for var in ds.variables if instance_dim in ds[var].dims]
        for instance_var in instance_vars:
            ds = ds.assign(
                {
                    instance_var: (
                        sample_dim,
                        np.repeat(ds[instance_var].data, row_size),
                        ds[instance_var].attrs,
                    )
                }
            )
        ds = ds.assign_attrs({"featureType": "point"})
        return ds


@xr.register_dataset_accessor("ormu")
class OrthoMultiArrayAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        ...
        # self._obj = xarray_obj
        # self._ra_type = None
        # self._sample_dimension = None
        # self._instance_dimension = None
        # self._count_var = None
        # self._index_var = None
        # self._timeseries_id = None
        # self.array_type
        #

    @property
    def array_type(self):
        return None
