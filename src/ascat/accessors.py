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
from __future__ import annotations

from typing import Any, Union, Sequence

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry

from pygeogrids import BasicGrid, CellGrid

from ascat.cf_array import cf_array_class, cf_array_type
from ascat.grids import GridRegistry
from ascat.utils import get_grid_gpis

from matplotlib import pyplot as plt


registry = GridRegistry()


@xr.register_dataset_accessor("pgg")
class PyGeoGriddedArrayAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._grid = None

    @property
    def grid(self):
        if self._grid is None:
            if grid_name := self._obj.attrs.get("grid_mapping_name"):
                self._grid = registry.get(grid_name)
        assert isinstance(self._grid, BasicGrid)
        return self._grid

    def set_grid_name(
        self,
        grid_name: str,
        grid_class: Union[type, None] = None,
    ):
        try:
            self._grid = registry.get(grid_name)
        except KeyError:
            if grid_class is not None:
                registry.register(grid_name, grid_class)
            else:
                raise ValueError(
                    f"Grid {grid_name} is not registered."
                    " Please pass a class for creating a grid to the `grid` argument."
                )
        self._obj.attrs["grid_mapping_name"] = grid_name

    def sel_bbox(
            self,
            bbox: Sequence[float]
    ) -> xr.Dataset:
        """
        Select data for a bounding box.

        bbox : tuple, optional
            Tuple of (latmin, latmax, lonmin, lonmax) coordinates.
        """
        gpis, lookup_vector = get_grid_gpis(
            grid=self.grid,
            bbox=bbox,
            return_lookup=True
        )
        return self._obj.pgg.sel_gpis(gpis, lookup_vector)

    def sel_coords(
        self,
        coords: Sequence[Sequence[float]],
        max_coord_dist: float = np.inf
    ) -> xr.Dataset:
        gpis, lookup_vector = get_grid_gpis(
            grid=self.grid,
            coords=coords,
            max_coord_dist=max_coord_dist,
            return_lookup=True,
        )
        return self._obj.pgg.sel_gpis(gpis, lookup_vector)

    def sel_cells(self, cells: Sequence[float]) -> xr.Dataset:
        assert isinstance(self.grid, CellGrid)
        gpis, lookup_vector = get_grid_gpis(
            grid=self.grid,
            cell=cells,
            return_lookup=True
        )
        return self._obj.pgg.sel_gpis(gpis, lookup_vector)

    def sel_geom(self, geom: BaseGeometry) -> xr.Dataset:
        gpis, lookup_vector = get_grid_gpis(
            grid=self.grid,
            geom=geom,
            return_lookup=True
        )
        return self._obj.pgg.sel_gpis(gpis, lookup_vector)

    def sel_gpis(
        self,
        gpis: Union[Sequence[int], None] = None,
        lookup_vector: Union[np.ndarray, None] = None,
    ) -> xr.Dataset:
        if lookup_vector is None:
            _, lookup_vector = get_grid_gpis(
                grid=self.grid,
                location_id=gpis,
                return_lookup=True
            )
        return self._obj.cf_geom.sel_instances(
            instance_vals=gpis,
            instance_lookup_vector=lookup_vector,
        )

    def lonlat_vars_from_gpi_var(
        self,
        gpi_var,
        lon_var="lon",
        lat_var="lat"
    ) -> tuple[xr.DataArray, xr.DataArray]:
        grid = self.grid
        lons, lats = grid.gpi2lonlat(self._obj[gpi_var])
        return (
            xr.DataArray(lons, dims=self._obj[gpi_var].dims, name=lon_var),
            xr.DataArray(lats, dims=self._obj[gpi_var].dims, name=lat_var)
        )



@xr.register_dataset_accessor("cf_geom")
class CFDiscreteGeometryAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._ds = xarray_obj
        self._obj = cf_array_class(self._ds, self.array_type)
        self._coord_vars = None
        self._instance_vars = None

    @property
    def array_type(self) -> str:
        return cf_array_type(self._ds)

    @property
    def timeseries_id(self) -> str:
        return self._obj.timeseries_id

    def set_coord_vars(self, coord_vars: Sequence[str]):
        self._coord_vars = coord_vars
        if self._instance_vars is None:
            self._instance_vars = []
        self._instance_vars.extend(coord_vars)

        self._obj = cf_array_class(
            self._ds,
            self.array_type,
            coord_vars=coord_vars,
            instance_vars=self._instance_vars,
        )

    def set_instance_vars(self, instance_vars: Sequence[str]):
        self._instance_vars = instance_vars
        self._obj = cf_array_class(
            self._ds,
            self.array_type,
            coord_vars=self._coord_vars,
            instance_vars=instance_vars,
        )

    def set_sample_dimension(self, sample_dim: str):
        self._ds = self._obj.set_sample_dimension(sample_dim)
        self._obj = cf_array_class(self._ds, self.array_type)
        return self._ds

    def sel_instances(
        self,
        instance_vals: Union[Sequence[int, str], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray[Any, np.dtype[np.bool]], None] = None,
        **kwargs,
    ):
        return self._obj.sel_instances(
            instance_vals=instance_vals,
            instance_lookup_vector=instance_lookup_vector,
            **kwargs,
        )

    def to_point_array(self):
        return self._obj.to_point_array()

    def to_indexed_ragged(self, **kwargs):
        return self._obj.to_indexed_ragged(**kwargs)

    def to_contiguous_ragged(self, **kwargs):
        return self._obj.to_contiguous_ragged(**kwargs)

    def to_orthomulti(self, **kwargs):
        return self._obj.to_orthomulti(**kwargs)

    def resample_to_orthomulti(self, **kwargs):
        return self._obj.resample_to_orthomulti(**kwargs)

    def to_raster(self, *args, **kwargs):
        return self._obj.to_raster(*args, **kwargs)
