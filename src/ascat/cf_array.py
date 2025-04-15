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

from typing import Union, Sequence

import numpy as np
import xarray as xr


def check_orthomulti_ts(ds):
    # Assumptions:
    # - two dimensions [DONE]
    # - single variable with only the sample dimension (e.g. time) [TODO]
    # - data variables have sample and instance dimension [TODO]
    # - data variables have ALL instance dimension coordinates listed as coordinates [TODO]
    if len(ds.dims) == 2:
        for v in ds.variables:
            if "cf_role" in ds[v].attrs and ds[v].attrs["cf_role"] == "timeseries_id":
                return True
    return False


def cf_array_type(ds):
    if ds.attrs.get("featureType") == "point":
        return "point"
    for v in ds.variables:
        if "instance_dimension" in ds[v].attrs:
            return "indexed"
        if "sample_dimension" in ds[v].attrs:
            return "contiguous"
    if check_orthomulti_ts(ds):
        return "orthomulti_ts"
    raise ValueError("Array type could not be determined.")


def cf_array_class(ds, array_type, **kwargs):
    if array_type == "point":
        return TimeseriesPointArray(ds, **kwargs)
    if array_type == "indexed":
        return RaggedArray(ds, **kwargs)
    if array_type == "contiguous":
        return RaggedArray(ds, **kwargs)
    if array_type == "orthomulti_ts":
        return OrthoMultiTimeseriesArray(ds, **kwargs)
    raise ValueError(f"Array type '{array_type}' not recognized."
                     "Should be one of 'point', 'indexed', 'contiguous', 'orthomulti_ts'.")


def point_to_indexed(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    timeseries_id: str,
    index_var: str = "locationIndex",
    instance_vars: Union[Sequence[str], None] = None,
    coord_vars: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    coord_vars = coord_vars or []
    instance_vars = instance_vars or []
    instance_vars = [timeseries_id] + instance_vars

    _, unique_index_1d, instanceIndex = np.unique(
        ds[timeseries_id], return_index=True, return_inverse=True
    )
    ds[index_var] = (sample_dim, instanceIndex, {"instance_dimension": instance_dim})

    for var in instance_vars:
        if var in ds:
            ds = ds.assign(
                {var: (instance_dim, ds[var][unique_index_1d].data, ds[var].attrs)}
            )
            if var in coord_vars:
                ds = ds.set_coords(var)
    ds = ds.assign_attrs({"featureType": "timeSeries"})
    return ds


def point_to_contiguous(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    timeseries_id: str,
    count_var: str = "row_size",
    instance_vars: Union[Sequence[str], None] = None,
    coord_vars: Union[Sequence[str], None] = None,
    sort_vars: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    coord_vars = coord_vars or []
    sort_vars = sort_vars or []
    instance_vars = instance_vars or []
    instance_vars = [timeseries_id] + instance_vars


    ds = ds.sortby([timeseries_id, *sort_vars])
    _, unique_index_1d, row_size = np.unique(
        ds[timeseries_id], return_index=True, return_counts=True
    )

    ds[count_var] = ("locations", row_size, {"sample_dimension": sample_dim})

    for var in instance_vars:
        if var in ds:
            encoding = ds[var].encoding
            ds[var] = (instance_dim, ds[var][unique_index_1d].data, ds[var].attrs)
            ds[var].encoding = encoding
            if var in coord_vars:
                ds = ds.set_coords(var)
    ds = ds.assign_attrs({"featureType": "timeSeries"})
    return ds


def contiguous_to_indexed(
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


def indexed_to_contiguous(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
    index_var: str,
    sort_vars: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    """
    Convert an indexed ragged array dataset to a contiguous ragged array dataset
    """
    sort_vars = sort_vars or []

    ds = ds.sortby([index_var, *sort_vars])
    idxs, sizes = np.unique(ds[index_var], return_counts=True)

    row_size = np.zeros_like(ds[instance_dim].data)
    row_size[idxs] = sizes
    ds = ds.assign(
        {count_var: (instance_dim, row_size, {"sample_dimension": sample_dim})}
    ).drop_vars([index_var])

    return ds


def indexed_to_point(
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


def contiguous_to_point(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
):
    """Convert a contiguous ragged array dataset to a Point Array.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    sample_dim : str
        Name of the sample dimension.
    instance_dim : str
        Name of the instance dimension.
    count_var : str
        Name of the count variable.

    Returns
    -------
    xarray.Dataset
        Dataset with only the time series variables.
    """
    row_size = ds[count_var].values
    ds = ds.drop_vars(count_var)
    instance_vars = [var for var in ds.variables if instance_dim in ds[var].dims]
    for instance_var in instance_vars:
        ds = ds.assign(
            {
                instance_var: (
                    sample_dim,
                    np.repeat(ds[instance_var].values, row_size),
                    ds[instance_var].attrs,
                )
            }
        )
    ds = ds.assign_attrs({"featureType": "point"})
    return ds


class CFDiscreteGeom:
    def __init__(
        self,
        xarray_obj: xr.Dataset,
        coord_vars: Union[Sequence[str], None] = None,
        instance_vars: Union[Sequence[str], None] = None,
        contiguous_sort_vars: Union[Sequence[str], None] = None,
    ):
        """
        Parameters
        ----------
        xarray_obj : xarray.Dataset
            Xarray dataset.
        coord_vars : Sequence[str], optional
            Coordinate variables, by default None.
        instance_vars : Sequence[str], optional
            Instance variables, by default None.
        contiguous_sort_vars : Sequence[str], optional
            Variables that each timeseries should be sorted by in contiguous ragged array format.
        """
        self._data = xarray_obj
        self._coord_vars = coord_vars or [
            "lon",
            "lat",
            "alt",
            "longitude",
            "latitude",
            "altitude",
        ]
        self._instance_vars = instance_vars or [
            "lon",
            "lat",
            "alt",
            "longitude",
            "latitude",
            "altitude",
            "location_description",
        ]

        self._contiguous_sort_vars = contiguous_sort_vars or [
            "time",
        ]

        self._ra_type = None
        self._sample_dimension = None
        self._instance_dimension = None
        self._count_var = None
        self._index_var = None
        self._timeseries_id = None
        self.array_type

    @property
    def array_type():
        raise NotImplementedError


class PointArray(CFDiscreteGeom):
    pass


class TimeseriesPointArray(PointArray):
    """
    Assumptions made beyond basic CF conventions:

    - cf_role="timeseries_id" is used to identify the timeseries ID variable for purposes
        of selecting instances and converting to ragged arrays. If you only have a single
        timeseries there's not much point in using this class.
    """
    @property
    def array_type(self) -> str:
        if self._ra_type is None:
            if self._data.attrs["featureType"] == "point":
                self._ra_type = "point"
                self._sample_dimension = str(list(self._data.dims)[0])
            else:
                raise ValueError(
                    "Dataset is not a point array"
                    "(should have featureType='point' in attributes)."
                )
        return self._ra_type

    @property
    def timeseries_id(self):
        if self._timeseries_id is not None:
            return self._timeseries_id
        for v in self._data.variables:
            if cf_role := self._data[v].attrs.get("cf_role"):
                if cf_role == "timeseries_id":
                    self._timeseries_id = v
                    return self.timeseries_id
        raise ValueError(
            "Timeseries ID could not be determined from dataset attributes."
        )

    def sel_instances(
        self,
        instance_vals: Union[Sequence[Union[int, str]], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
        timeseries_id: str = "location_id",
    ):
        ds = self._data
        return self._select_instances(
            ds,
            self._sample_dimension,
            instance_vals,
            instance_lookup_vector,
            timeseries_id,
        )

    def to_indexed_ragged(
        self,
        instance_dim: str = "locations",
        timeseries_id: str = "location_id",
        index_var: str = "locationIndex",
        instance_vars: Union[Sequence[str], None] = None,
        coord_vars: Union[Sequence[str], None] = None,
    ) -> xr.Dataset:
        return self._point_to_indexed(
            self._data,
            self._sample_dimension,
            instance_dim,
            timeseries_id,
            index_var,
            instance_vars or self._instance_vars,
            coord_vars or self._coord_vars,
        )

    def to_contiguous_ragged(
        self,
        instance_dim: str = "locations",
        timeseries_id: str = "location_id",
        count_var: str = "row_size",
        instance_vars: Union[Sequence[str], None] = None,
        coord_vars: Union[Sequence[str], None] = None,
        sort_vars: Union[Sequence[str], None] = None,
    ) -> xr.Dataset:
        return self._point_to_contiguous(
            self._data,
            self._sample_dimension,
            instance_dim,
            timeseries_id,
            count_var,
            instance_vars or self._instance_vars,
            coord_vars or self._coord_vars,
            sort_vars or self._contiguous_sort_vars,
        )

    def to_orthomulti(
            self,
            instance_dim: str = "locations",
            timeseries_id: str = "location_id",
            count_var: str = "row_size",
            instance_vars: Union[Sequence[str], None] = None,
            coord_vars: Union[Sequence[str], None] = None,
            sort_vars: Union[Sequence[str], None] = None,
    ):
        return self._point_to_orthomulti(
            self._data,
            self._sample_dimension,
            instance_dim,
            timeseries_id,
            count_var,
            instance_vars or self._instance_vars,
            coord_vars or self._coord_vars,
            sort_vars or self._contiguous_sort_vars,
        )

    def resample_to_orthomulti(
            self,
            instance_dim: str = "locations",
            timeseries_id: str = "location_id",
            count_var: str = "row_size",
            instance_vars: Union[Sequence[str], None] = None,
            coord_vars: Union[Sequence[str], None] = None,
            sort_vars: Union[Sequence[str], None] = None,
            vars_to_resample: Union[Sequence[str], None] = None,
            resample_method: callable = np.mean,
            resample_period: str = "1M",
    ):
        return self._resample_point_to_orthomulti(
            self._data,
            self._sample_dimension,
            instance_dim,
            timeseries_id,
            count_var,
            instance_vars or self._instance_vars,
            coord_vars or self._coord_vars,
            sort_vars or self._contiguous_sort_vars,
            vars_to_resample,
            resample_method,
            resample_period,
        )

    def to_point_array(self):
        return self._data

    def set_sample_dimension(self, sample_dim: str):
        if self._sample_dimension != sample_dim:
            self._data = self._data.rename_dims({self._sample_dimension: sample_dim})
            self._sample_dimension = sample_dim
        return self._data

    @staticmethod
    def _select_instances(
        ds: xr.Dataset,
        sample_dim: str,
        instance_vals: Union[Sequence[Union[int, str]], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
        timeseries_id: str = "location_id",
    ) -> xr.Dataset:
        if not ds.chunks:
            ds = ds.chunk({sample_dim: -1})
        if instance_vals is None:
            instance_vals = []
        if instance_lookup_vector is not None:
            sample_idx = instance_lookup_vector[ds[timeseries_id]]
            return ds.sel({sample_dim: sample_idx})
        sample_idx = np.isin(ds[timeseries_id], instance_vals)
        return ds.sel({sample_dim: sample_idx})

    @staticmethod
    def _point_to_indexed(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        timeseries_id: str,
        index_var: str = "locationIndex",
        instance_vars: Union[Sequence[str], None] = None,
        coord_vars: Union[Sequence[str], None] = None,
    ) -> xr.Dataset:
        return point_to_indexed(
            ds,
            sample_dim,
            instance_dim,
            timeseries_id,
            index_var,
            instance_vars,
            coord_vars,
        )

    @staticmethod
    def _point_to_contiguous(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        timeseries_id: str,
        count_var: str = "row_size",
        instance_vars: Union[Sequence[str], None] = None,
        coord_vars: Union[Sequence[str], None] = None,
        sort_vars: Union[Sequence[str], None] = None,
    ) -> xr.Dataset:
        return point_to_contiguous(
            ds,
            sample_dim,
            instance_dim,
            timeseries_id,
            count_var,
            instance_vars,
            coord_vars,
            sort_vars,
        )

    @staticmethod
    def _point_to_orthomulti(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        timeseries_id: str,
        count_var: str = "row_size",
        instance_vars: Union[Sequence[str], None] = None,
        coord_vars: Union[Sequence[str], None] = None,
        sort_vars: Union[Sequence[str], None] = None,
    ) -> xr.Dataset:
        """
        At the moment, minimum resolution is 1D
        """
        ds = ds.rename({sample_dim: "time"}).set_xindex("time")
        ds = ds.set_index(event=["time", timeseries_id]).unstack("event")
        for c in ds.coords:
            if "time" in ds[c].dims and c != "time":
                ds[c] = ds[c].max("time", keep_attrs=True)
        ds.attrs.pop("featureType")
        return ds


    @staticmethod
    def _resample_point_to_orthomulti(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        timeseries_id: str,
        count_var: str = "row_size",
        instance_vars: Union[Sequence[str], None] = None,
        coord_vars: Union[Sequence[str], None] = None,
        sort_vars: Union[Sequence[str], None] = None,
        vars_to_resample: Union[Sequence[str], None] = None,
        resample_method: callable = np.mean,
        resample_period: str = "1M",
    ) -> xr.Dataset:
        """
        At the moment, minimum resolution is 1D
        """
        ds = ds.rename({sample_dim: "time"}).set_xindex("time")
        ds = ds.set_index(event=["time", timeseries_id]).unstack("event")
        ds = ds.resample(time=resample_period).apply(resample_method)
        ds.attrs.pop("featureType")
        return ds



class RaggedArray(CFDiscreteGeom):
    @property
    def array_type(self):
        if self._ra_type is not None:
            return self._ra_type

        ds = self._data
        for v in ds.variables:
            if "instance_dimension" in ds[v].attrs:
                self._ra_type = "indexed"
                self._index_var = v
                self._instance_dimension = ds[v].attrs["instance_dimension"]
                self._sample_dimension = str(ds[v].dims[0])
                return self._ra_type

            if "sample_dimension" in ds[v].attrs:
                self._ra_type = "contiguous"
                self._count_var = v
                self._sample_dimension = ds[v].attrs["sample_dimension"]
                if len(ds[v].dims) > 0:
                    self._instance_dimension = ds[v].dims[0]
                return self._ra_type

        raise ValueError("Ragged array type could not be determined.")

    @property
    def timeseries_id(self):
        if self._timeseries_id is not None:
            return self._timeseries_id
        for v in self._data.variables:
            if cf_role := self._data[v].attrs.get("cf_role"):
                if cf_role == "timeseries_id":
                    self._timeseries_id = v
                    return self.timeseries_id
        raise ValueError(
            "Timeseries ID could not be determined from dataset attributes."
        )

    def to_indexed_ragged(
            self,
            index_var: str = "locationIndex"
    ) -> xr.Dataset:
        if self.array_type == "indexed":
            return self._data
        elif self.array_type == "contiguous":
            if self._index_var is None:
                self._index_var = index_var
            return self._contiguous_to_indexed(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
                self._index_var,
            )

    def to_contiguous_ragged(
        self,
        count_var: str = "row_size",
        sort_vars: Union[Sequence[str], None] = None
    ) -> xr.Dataset:
        if self.array_type == "contiguous":
            return self._data
        elif self.array_type == "indexed":
            if self._count_var is None:
                self._count_var = count_var
            return self._indexed_to_contiguous(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
                self._index_var,
                sort_vars=sort_vars or self._contiguous_sort_vars,
            )

    def to_point_array(self):
        if self.array_type == "indexed":
            return self._indexed_to_point(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._index_var,
            )
        if self.array_type == "contiguous":
            return self._contiguous_to_point(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
            )

    def sel_instances(
        self,
        instance_vals: Union[Sequence[Union[int, str]], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
    ) -> xr.Dataset:
        if self.array_type == "indexed":
            # convert to point array, select there, convert back\
            ds = self.to_point_array()
            instances = ds.cf_geom.sel_instances(
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
            )
            return instances.cf_geom.to_indexed_ragged(index_var=self._index_var)

        if self.array_type == "contiguous":
            return self._select_instances_contiguous(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self.timeseries_id,
                self._count_var,
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
            )


    def set_sample_dimension(self, sample_dim: str):
        if self._sample_dimension != sample_dim:
            self._data = self._data.rename_dims({self._sample_dimension: sample_dim})
            if self.array_type == "contiguous":
                self._data[self._count_var].attrs["sample_dimension"] = sample_dim
            self._sample_dimension = sample_dim
        return self._data


    @staticmethod
    def _select_instances_contiguous(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        timeseries_id: str,
        count_var: str,
        instance_vals: Union[Sequence[int], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
    ) -> xr.Dataset:
        if instance_vals is None:
            instance_vals = []

        # For contiguous using the lookup vector would be slower, so if we get only that,
        # we'll just turn it into an instance_vals array.
        if len(instance_vals) == 0:
            if instance_lookup_vector is not None and sum(instance_lookup_vector) > 0:
                instance_vals = np.where(instance_lookup_vector)[0]

        def get_single_instance_idxs(ds, instance_val):
            instances_idx = np.where(ds[timeseries_id] == instance_val)[0]
            if len(instances_idx) == 0:
                return None
            instances_idx = int(instances_idx[0])
            sample_start = int(
                ds[count_var].isel({instance_dim: slice(0, instances_idx)}).sum().values
            )
            sample_end = int(
                sample_start + ds[count_var].isel({instance_dim: instances_idx}).values
            )
            return sample_start, sample_end, instances_idx

        def select_single_instance(ds, sample_start, sample_end, instances_idx):
            return ds.isel(
                {
                    sample_dim: slice(sample_start, sample_end),
                    instance_dim: instances_idx,
                }
            )

        def select_several_instances(ds, sample_starts, sample_ends, instances_idxs):
            sample_idxs = np.concatenate(
                [range(start, end)
                 for start, end
                 in zip(sample_starts, sample_ends)
                 if end > start]
            )
            return ds.isel({sample_dim: sample_idxs,
                            instance_dim: np.array(instances_idxs)})


        if len(instance_vals) == 1:
            if get_single_instance_idxs(ds, instance_vals[0]) is None:
                return None
            return select_single_instance(ds, *get_single_instance_idxs(ds, instance_vals[0]))
        else:
            instance_vals = np.unique(instance_vals)
            ds[count_var].load()
            ds[timeseries_id].load()
            results = [get_single_instance_idxs(ds, instance_val)
                       for instance_val in instance_vals]
            results = [r for r in results if r is not None]
            if len(results) == 0:
                return None
            if not ds.chunks:
                ds = ds.chunk({sample_dim: -1})
            return select_several_instances(
                ds,
                *zip(*results)
            )


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
        return contiguous_to_indexed(ds, sample_dim, instance_dim, count_var, index_var)

    @staticmethod
    def _indexed_to_contiguous(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        count_var: str,
        index_var: str,
        sort_vars: Union[Sequence[str], None] = None,
    ) -> xr.Dataset:
        """
        Convert an indexed ragged array dataset to a contiguous ragged array dataset
        """
        return indexed_to_contiguous(
            ds, sample_dim, instance_dim, count_var, index_var, sort_vars
        )

    @staticmethod
    def _indexed_to_point(
        ds: xr.Dataset, sample_dim: str, instance_dim: str, index_var: str
    ):
        return indexed_to_point(ds, sample_dim, instance_dim, index_var)

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
        return contiguous_to_point(ds, sample_dim, instance_dim, count_var)


class OrthoMultiTimeseriesArray(CFDiscreteGeom):
    @property
    def array_type(self):
        if check_orthomulti_ts(self._data):
            for v in self._data.variables:
                if "cf_role" in self._data[v].attrs and self._data[v].attrs["cf_role"] == "timeseries_id":
                    self._timeseries_id = v
                    self._instance_dimension = self._data[v].dims[0]
                    break
            return "orthomulti_ts"
        else:
            raise ValueError("Dataset is not an orthomulti timeseries array.")

    def sel_instances(
        self,
        instance_vals: Union[Sequence[Union[int, str]], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
    ):
        """
        Select requested timeseries instances from an orthomulti timeseries array dataset.

        Parameters
        ----------
        instance_vals : Union[Sequence[Union[int, str]], np.ndarray], optional
            List of instance values to select, by default None
        instance_lookup_vector : Union[np.ndarray], optional
            Lookup vector for instance values, by default None
        """
        return self._select_instances(
            self._data,
            self._instance_dimension,
            self._timeseries_id,
            instance_vals,
            instance_lookup_vector,
        )

    def set_sample_dimension(self, sample_dim: str):
        if self._sample_dimension != sample_dim:
            self._data = self._data.rename_dims({self._sample_dimension: sample_dim})
            self._sample_dimension = sample_dim
        return self._data

    def to_raster(self,
                  x_var,
                  y_var):
        return self._data.reset_index(self._timeseries_id)\
                         .set_index({self._instance_dimension: [x_var, y_var]})\
                         .unstack(self._instance_dimension)

    @staticmethod
    def _select_instances(
        ds: xr.Dataset,
        instance_dim: str,
        timeseries_id: str,
        instance_vals: Union[Sequence[Union[int, str]], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
    ) -> xr.Dataset:
        """
        Selects requested instances from an orthomulti timeseries array dataset.

        Returns a dataset containing the requested instances. If instances are requested
        that are not in the dataset, no error will be thrown.
        """
        if instance_lookup_vector is not None:
            instance_bool = instance_lookup_vector[ds[timeseries_id]]
        else:
            instance_bool = np.isin(ds[timeseries_id], instance_vals)
        return ds.sel({instance_dim: instance_bool})
