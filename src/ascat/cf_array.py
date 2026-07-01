# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

from __future__ import annotations

from typing import Union, Sequence, Callable

import numpy as np
import xarray as xr

# The dataset-level conversions live in a class-free module; re-exported here
# for backward compatibility.
from ascat.cf_conversions import (  # noqa: F401
    point_to_indexed,
    point_to_contiguous,
    contiguous_to_indexed,
    indexed_to_contiguous,
    indexed_to_point,
    contiguous_to_point,
)

# Recognized CF discrete sampling geometry array types.
POINT = "point"
INDEXED = "indexed"
CONTIGUOUS = "contiguous"
ORTHOMULTI_TS = "orthomulti_ts"


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
    """Detect the CF discrete sampling geometry array type of ``ds``."""
    if ds.attrs.get("featureType") == "point":
        return POINT
    for v in ds.variables:
        if "instance_dimension" in ds[v].attrs:
            return INDEXED
        if "sample_dimension" in ds[v].attrs:
            return CONTIGUOUS
    if check_orthomulti_ts(ds):
        return ORTHOMULTI_TS
    raise ValueError("Array type could not be determined.")


def cf_array_class(ds, array_type, **kwargs):
    """Wrap ``ds`` in the array class matching ``array_type``."""
    classes = {
        POINT: TimeseriesPointArray,
        INDEXED: RaggedArray,
        CONTIGUOUS: RaggedArray,
        ORTHOMULTI_TS: OrthoMultiTimeseriesArray,
    }
    if array_type not in classes:
        raise ValueError(
            f"Array type '{array_type}' not recognized. Should be one of "
            f"{', '.join(classes)}.")
    return classes[array_type](ds, **kwargs)


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
        self._resolve()

    @classmethod
    def from_dataset(cls, ds, **kwargs):
        """Detect the array type of ``ds`` and wrap it in the right class."""
        return cf_array_class(ds, cf_array_type(ds), **kwargs)

    def _resolve(self):
        """
        Determine the array type and populate the dimension/variable metadata
        (``_sample_dimension``, ``_instance_dimension``, ``_count_var``,
        ``_index_var``, ``_ra_type``). Called once from ``__init__``.
        """
        raise NotImplementedError

    @property
    def array_type(self):
        return self._ra_type

    @property
    def timeseries_id(self):
        """Name of the variable carrying ``cf_role='timeseries_id'``."""
        if self._timeseries_id is not None:
            return self._timeseries_id
        for v in self._data.variables:
            if self._data[v].attrs.get("cf_role") == "timeseries_id":
                self._timeseries_id = v
                return self._timeseries_id
        raise ValueError(
            "Timeseries ID could not be determined from dataset attributes."
        )


class PointArray(CFDiscreteGeom):
    pass


class TimeseriesPointArray(PointArray):
    """
    Assumptions made beyond basic CF conventions:

    - cf_role="timeseries_id" is used to identify the timeseries ID variable for purposes
        of selecting instances and converting to ragged arrays. If you only have a single
        timeseries there's not much point in using this class.
    """
    def _resolve(self):
        if self._data.attrs.get("featureType") != "point":
            raise ValueError(
                "Dataset is not a point array"
                "(should have featureType='point' in attributes)."
            )
        self._ra_type = POINT
        self._sample_dimension = str(list(self._data.dims)[0])

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
        return point_to_indexed(
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
        return point_to_contiguous(
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
            resample_method: Callable = np.mean,
            resample_period: str = "1ME",
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
        resample_method: Callable = np.mean,
        resample_period: str = "1ME",
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
    def _resolve(self):
        ds = self._data
        for v in ds.variables:
            if "instance_dimension" in ds[v].attrs:
                self._ra_type = INDEXED
                self._index_var = v
                self._instance_dimension = ds[v].attrs["instance_dimension"]
                self._sample_dimension = str(ds[v].dims[0])
                return

            if "sample_dimension" in ds[v].attrs:
                self._ra_type = CONTIGUOUS
                self._count_var = v
                self._sample_dimension = ds[v].attrs["sample_dimension"]
                if len(ds[v].dims) > 0:
                    self._instance_dimension = ds[v].dims[0]
                return

        raise ValueError("Ragged array type could not be determined.")

    def to_indexed_ragged(
            self,
            index_var: str = "locationIndex"
    ) -> xr.Dataset:
        if self.array_type == INDEXED:
            return self._data
        elif self.array_type == CONTIGUOUS:
            if self._index_var is None:
                self._index_var = index_var
            return contiguous_to_indexed(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
                self._index_var,
            )
        raise ValueError(
            f"Cannot convert array type '{self.array_type}' to indexed ragged.")

    def to_contiguous_ragged(
        self,
        count_var: str = "row_size",
        sort_vars: Union[Sequence[str], None] = None
    ) -> xr.Dataset:
        if self.array_type == CONTIGUOUS:
            return self._data
        elif self.array_type == INDEXED:
            if self._count_var is None:
                self._count_var = count_var
            return indexed_to_contiguous(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
                self._index_var,
                sort_vars=sort_vars or self._contiguous_sort_vars,
            )
        raise ValueError(
            f"Cannot convert array type '{self.array_type}' to contiguous "
            "ragged.")

    def to_point_array(self):
        if self.array_type == INDEXED:
            return indexed_to_point(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._index_var,
            )
        if self.array_type == CONTIGUOUS:
            return contiguous_to_point(
                self._data,
                self._sample_dimension,
                self._instance_dimension,
                self._count_var,
            )
        raise ValueError(
            f"Cannot convert array type '{self.array_type}' to point array.")

    def sel_instances(
        self,
        instance_vals: Union[Sequence[Union[int, str]], np.ndarray, None] = None,
        instance_lookup_vector: Union[np.ndarray, None] = None,
    ) -> xr.Dataset:
        if self.array_type == INDEXED:
            # convert to point array, select there, convert back\
            ds = self.to_point_array()
            instances = ds.cf_geom.sel_instances(
                instance_vals=instance_vals,
                instance_lookup_vector=instance_lookup_vector,
            )
            return instances.cf_geom.to_indexed_ragged(index_var=self._index_var)

        if self.array_type == CONTIGUOUS:
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
            if self.array_type == CONTIGUOUS:
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


class OrthoMultiTimeseriesArray(CFDiscreteGeom):
    def _resolve(self):
        if not check_orthomulti_ts(self._data):
            raise ValueError(
                "Dataset is not an orthomulti timeseries array.")
        for v in self._data.variables:
            if self._data[v].attrs.get("cf_role") == "timeseries_id":
                self._timeseries_id = v
                self._instance_dimension = self._data[v].dims[0]
                break
        # the sample (e.g. time) dimension is the remaining dimension
        other_dims = [d for d in self._data.dims
                      if d != self._instance_dimension]
        if other_dims:
            self._sample_dimension = str(other_dims[0])
        self._ra_type = ORTHOMULTI_TS

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

    def to_point_array(self, sample_dim: str = "obs"):
        """
        Convert the orthomulti timeseries array to a point array.

        The instance and sample dimensions are stacked into a single sample
        dimension, so every instance/time combination becomes one observation
        (an orthomulti array is dense by construction).
        """
        inst = self._instance_dimension
        samp = self._sample_dimension
        ds = self._data.stack({sample_dim: (inst, samp)}).reset_index(sample_dim)
        # drop the positional instance level left over from stacking
        if inst in ds.coords or inst in ds.variables:
            ds = ds.drop_vars(inst)
        return ds.assign_attrs({"featureType": "point"})

    def to_indexed_ragged(self, sample_dim: str = "obs", **kwargs) -> xr.Dataset:
        """Convert to an indexed ragged array (via a point array)."""
        kwargs.setdefault("timeseries_id", self._timeseries_id)
        return TimeseriesPointArray(
            self.to_point_array(sample_dim=sample_dim)
        ).to_indexed_ragged(**kwargs)

    def to_contiguous_ragged(self, sample_dim: str = "obs",
                             **kwargs) -> xr.Dataset:
        """Convert to a contiguous ragged array (via a point array)."""
        kwargs.setdefault("timeseries_id", self._timeseries_id)
        return TimeseriesPointArray(
            self.to_point_array(sample_dim=sample_dim)
        ).to_contiguous_ragged(**kwargs)

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
