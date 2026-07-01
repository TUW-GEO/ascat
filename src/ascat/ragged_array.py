# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

from pathlib import Path

import numpy as np
import xarray as xr

from ascat.cf_conversions import indexed_to_contiguous
from ascat.cf_conversions import contiguous_to_indexed
from ascat.cf_conversions import contiguous_to_point
from ascat.cf_conversions import indexed_to_point
from ascat.cf_conversions import contiguous_to_incomplete
from ascat.cf_conversions import incomplete_to_contiguous
from ascat.cf_conversions import contiguous_to_orthogonal
from ascat.cf_conversions import orthogonal_to_contiguous


def verify_multidim(ds: xr.Dataset, instance_dim: str,
                    element_dim: str) -> None:
    """
    Verify a dataset follows the CF multidimensional array definition
    (orthogonal or incomplete).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    instance_dim : str
        Name of the instance dimension.
    element_dim : str
        Name of the element dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that instance dimension exists
    if instance_dim not in ds.dims:
        raise RuntimeError(f"Instance dimension is missing '{instance_dim}'")

    # check that element dimension exists
    if element_dim not in ds.dims:
        raise RuntimeError(f"Element dimension is missing '{element_dim}'")


def verify_contiguous_ragged(ds: xr.Dataset, count_var: str,
                             instance_dim: str) -> None:
    """
    Verify dataset follows contiguous ragged array CF definition.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    count_var : str
        Name of the count variable.
        Count variable contains the length of each time series feature. It is
        identified by having an attribute with name 'sample_dimension' whose
        value is name of the sample dimension. The count variable implicitly
        partitions into individual instances all variables that have the
        sample dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that count variable exists
    if count_var not in ds:
        raise RuntimeError(f"Count variable is missing: {count_var}")

    # check that count variable contains sample_dimension attribute
    if "sample_dimension" not in ds[count_var].attrs:
        raise RuntimeError(f"Count variable '{count_var}' has no "
                           "sample_dimension attribute")

    # check that count variable has instance_dimension as single dimension
    if ds[count_var].dims != (instance_dim,):
        raise RuntimeError(f"Count variable '{count_var}' must have the "
                           f"instance dimension '{instance_dim}' as its "
                           "single dimension")


def verify_indexed_ragged(ds: xr.Dataset, index_var: str,
                          sample_dim: str) -> None:
    """
    Verify dataset follows indexed ragged array CF definition.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    index_var : str
        The index variable can be identified by having an attribute with
        name of instance_dimension whose value is the instance dimension.
    sample_dim : str
        Name of the sample dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that index variable exists
    if index_var not in ds:
        raise RuntimeError(f"Index variable is missing: {index_var}")

    # check that index variable must have sample dimension as single dimension
    if ds[index_var].dims != (sample_dim,):
        raise RuntimeError(f"Index variable '{index_var}' must have the "
                           f"sample dimension '{sample_dim}' as its "
                           "single dimension")

    # check that index variable has instance_dimension attribute
    if "instance_dimension" not in ds[index_var].attrs:
        raise RuntimeError(f"Index variable '{index_var}' has no "
                           "instance_dimension attribute")


def verify_point_array(ds: xr.Dataset, sample_dim: str) -> None:
    """
    Verify dataset follows the CF point data array convention.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    sample_dim : str
        Name of the sample dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that the sample_dim exists
    if sample_dim not in ds.dims:
        raise RuntimeError(f"Sample dimension '{sample_dim}' is missing.")

    # check all data and coordinate variables have only the sample_dim
    for var in ds.variables:
        dims = ds[var].dims
        if ds[var].ndim > 0 and dims != (sample_dim,):
            raise RuntimeError(
                f"Variable '{var}' does not conform to point structure "
                f"(dims: {dims}). All variables must use only the sample_dim ('{sample_dim}')."
            )


class PointData:
    """
    Point data represent scattered locations and times with no implied
    relationship among of coordinate positions, both data and coordinates must
    share the same (sample) instance dimension.
    """

    def __init__(self, ds: xr.Dataset, sample_dim: str):
        """
        Initialize.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to be verified.
        sample_dim : str
            Name of the sample dimension. The sample dimension indicates the
            number of instances (e.g. stations, locations).
        """
        self.sample_dim = sample_dim
        self._data = ds
        self.validate()

    def validate(self):
        """Validate format."""
        verify_point_array(self.ds, self.sample_dim)

    @property
    def ds(self):
        return self._data

    def to_indexed(self, index_var: str = "obs", instance_dim: str = "loc"):
        """
        Convert point data to indexed ragged array.

        Parameters
        ----------
        index_var : str
            Name of the new index variable to be added.
        instance_dim : str
            Name of the instance dimension.

        Returns
        -------
        indexed : IndexedRaggedArray
            Indexed ragged array object.
        """
        if instance_dim not in self.ds:
            raise ValueError(
                f"'{instance_dim}' must be a coordinate in the dataset")

        instance_ids, inverse_index = np.unique(
            self.ds[instance_dim], return_inverse=True)

        # create index variable (mapping each sample to its instance)
        index_data = xr.DataArray(
            inverse_index,
            dims=(self.sample_dim,),
            attrs={"instance_dimension": instance_dim})

        new_ds = self.ds.copy()
        new_ds[index_var] = index_data
        new_ds = new_ds.assign_coords({instance_dim: instance_ids})

        return IndexedRaggedArray(
            new_ds, index_var=index_var, sample_dim=self.sample_dim)

    def to_contiguous(self,
                      count_var: str = "row_size",
                      instance_dim: str = "loc"):
        """
        Convert point data to contiguous ragged array.

        Parameters
        ----------
        count_var : str
            Name of the new count variable to be added (default: 'row_size').
        instance_dim : str
            Name of the instance dimension (default: 'loc').

        Returns
        -------
        contiguous : ContiguousRaggedArray
            Contiguous ragged array object.
        """
        if instance_dim not in self.ds:
            raise ValueError(
                f"'{instance_dim}' must be a coordinate in the dataset")

        group = self.ds.groupby(self.ds[instance_dim])
        row_sizes = group.groups
        instance_ids = list(row_sizes.keys())
        counts = np.array([len(row_sizes[i]) for i in instance_ids])

        sort_index = np.argsort(self.ds[instance_dim].values)
        sorted_ds = self.ds.isel({self.sample_dim: sort_index})

        count_data = xr.DataArray(
            counts,
            dims=(instance_dim,),
            attrs={"sample_dimension": self.sample_dim})

        new_ds = sorted_ds.copy()
        new_ds[count_var] = count_data

        if instance_dim in new_ds.coords:
            new_ds = new_ds.assign_coords(
                {instance_dim: np.array(instance_ids)})

        return ContiguousRaggedArray(
            new_ds, count_var=count_var, instance_dim=instance_dim)

    def to_incomplete(self,
                      count_var: str = "row_size",
                      instance_dim: str = "loc"):
        """
        Convert point data to an incomplete multidimensional array (CF 9.3.2).

        Parameters
        ----------
        count_var : str, optional
            Name of the intermediate count variable (default: "row_size").
        instance_dim : str, optional
            Name of the instance dimension (default: "loc").

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array time series.
        """
        return self.to_contiguous(
            count_var=count_var, instance_dim=instance_dim).to_incomplete()

    def to_orthogonal(self,
                      element_coord: str,
                      count_var: str = "row_size",
                      instance_dim: str = "loc",
                      element_dim: str = None,
                      strict: bool = True):
        """
        Convert point data to an orthogonal multidimensional array (CF 9.3.1).

        Parameters
        ----------
        element_coord : str
            Name of the per-sample coordinate defining the shared element axis.
        count_var : str, optional
            Name of the intermediate count variable (default: "row_size").
        instance_dim : str, optional
            Name of the instance dimension (default: "loc").
        element_dim : str, optional
            Name of the resulting element dimension (default: ``element_coord``).
        strict : bool, optional
            If True (default), raise if the instances do not form a complete
            grid on ``element_coord``.

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array time series.
        """
        return self.to_contiguous(
            count_var=count_var, instance_dim=instance_dim).to_orthogonal(
                element_coord, element_dim=element_dim, strict=strict)


class MultidimArray:
    """
    Base class for CF multidimensional array representations.

    Holds a dense ``(instance, element)`` dataset and the behaviour shared by
    the orthogonal (CF 9.3.1) and incomplete (CF 9.3.2) representations. The
    conversions back to ragged/point form are defined by the subclasses, which
    differ in whether the element dimension is a shared coordinate axis
    (orthogonal) or a padded positional index (incomplete).

    Attributes
    ----------
    instance_dim : str
        Name of the instance dimension.
    element_dim : str
        Name of the element dimension.
    ds : xarray.Dataset
        Multidimensional array dataset.
    """

    def __init__(self,
                 ds: xr.Dataset,
                 instance_dim: str = "loc",
                 element_dim: str = "time"):
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data stored in multidimensional array format.
        instance_dim : str
            Instance dimension name.
        element_dim : str
            Element dimension name.
        """
        self.instance_dim = instance_dim
        self.element_dim = element_dim
        self._data = ds
        self.validate()

        # if instance_dim exists as a variable, its values are the instance
        # identifiers used as keys for reading
        if self.instance_dim in ds:
            self._instances = ds[self.instance_dim].to_numpy()
            self._instance_lookup = np.zeros(
                self._instances.max() + 1,
                dtype=np.int64) + self._instances.size
            self._instance_lookup[self._instances] = np.arange(
                self._instances.size)
        else:
            self._instances = np.arange(ds.sizes[instance_dim])
            self._instance_lookup = np.arange(self._instances.size)

    def validate(self):
        """Validate format."""
        verify_multidim(self.ds, self.instance_dim, self.element_dim)

    @property
    def ds(self):
        return self._data

    def sel_instance(self, instance_id: int):
        """Read time series"""
        return self.ds.sel({self.instance_dim: instance_id})

    def __iter__(self):
        """Iterator over time series"""
        for instance in self.ds[self.instance_dim]:
            yield self.sel_instance(instance)

    def iter(self):
        """Explicit iterator method"""
        return self.__iter__()

    def apply(self, func):
        """Apply a function to each instance time series."""
        return [func(ts) for ts in self]

    def _to_contiguous_ds(self, count_var, sample_dim):
        """Subclass hook: convert to a contiguous ragged array dataset."""
        raise NotImplementedError

    def to_contiguous(self, count_var: str = "row_size",
                      sample_dim: str = "obs"):
        """
        Convert to a contiguous ragged array.

        Returns
        -------
        data : ContiguousRaggedArray
            Contiguous ragged array time series.
        """
        new_ds = self._to_contiguous_ds(count_var, sample_dim)
        return ContiguousRaggedArray(new_ds, count_var, self.instance_dim)

    def to_indexed(self, count_var: str = "row_size",
                   sample_dim: str = "obs", index_var: str = "locationIndex"):
        """
        Convert to an indexed ragged array (via a contiguous ragged array).

        Returns
        -------
        data : IndexedRaggedArray
            Indexed ragged array time series.
        """
        return self.to_contiguous(
            count_var=count_var, sample_dim=sample_dim).to_indexed(index_var)

    def to_point_data(self, count_var: str = "row_size",
                      sample_dim: str = "obs"):
        """
        Convert to point data (via a contiguous ragged array).

        Returns
        -------
        data : PointData
            Point data.
        """
        return self.to_contiguous(
            count_var=count_var, sample_dim=sample_dim).to_point_data()


class IncompleteMultidimArray(MultidimArray):
    """
    Incomplete multidimensional array representation (CF 9.3.2).

    Instances may have different numbers of elements (and different element
    coordinates); the dense ``(instance, element)`` array is padded with the
    dtype fill value. The element dimension is a positional index.
    """

    def _to_contiguous_ds(self, count_var, sample_dim):
        return incomplete_to_contiguous(
            self.ds,
            self.instance_dim,
            self.element_dim,
            count_var=count_var,
            sample_dim=sample_dim,
        )


class OrthogonalMultidimArray(MultidimArray):
    """
    Orthogonal multidimensional array representation (CF 9.3.1).

    All instances share the same element coordinate axis (e.g. one time axis),
    stored as a single shared 1-D coordinate; the array is complete (no
    padding).

    Parameters
    ----------
    element_coord : str, optional
        Name of the shared element coordinate variable (default: the element
        dimension name).
    """

    def __init__(self,
                 ds: xr.Dataset,
                 instance_dim: str = "loc",
                 element_dim: str = "time",
                 element_coord: str = None):
        self.element_coord = element_coord or element_dim
        super().__init__(ds, instance_dim, element_dim)

    def _to_contiguous_ds(self, count_var, sample_dim):
        return orthogonal_to_contiguous(
            self.ds,
            self.instance_dim,
            self.element_dim,
            element_coord=self.element_coord,
            count_var=count_var,
            sample_dim=sample_dim,
        )


class ContiguousRaggedArray:
    """
    Contiguous ragged array representation (CF convention).

    In an contiguous ragged array representation, the dataset for all time
    series are stored in a single 1D array. Additional variables or
    dimensions provide the metadata needed to map these values back to their
    respective time series.

    The contiguous ragged array representation can be used only if the size
    of each instance is known at the time that it is created. In this
    representation the data for each instance will be contiguous on disk.

    If the instance dimension exists as a variable, it is assumed that the
    values represent the identifiers for each instance otherwise they are
    count upwards from 0.

    Attributes
    ----------
    instance_dim : str
        Name of the instance dimension.
    sample_dim : str
        Name of the sample dimension. The variable bearing the
        sample_dimension attribute (i.e. count_var) must have the instance
        dimension as its single dimension, and must have an integer type.
    count_var : str
        Name of the count variable. The count variable must be an integer
        type and must have the instance dimension as its sole dimension.
        The count variable are identifiable by the presence of an attribute,
        sample_dimension, found on the count variable, which names the sample
        dimension being counted.
    ds : xarray.Dataset
        Contiguous ragged array dataset.
    instance_variables : list
        List of instance variables.
    instance_ids : list
        List of instance ids.

    Methods
    -------
    sel_instance(i)
        Read time series for given instance.
    iter()
        Yield time series for each instance.
    """

    def __init__(self,
                 ds: xr.Dataset,
                 count_var: str,
                 instance_dim: str,
                 instance_id_var: str = None):
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data stored in contiguous ragged array format.
        count_var : str
            Count variable name.
        instance_dim : str
            Instance dimension name.
        instance_id_var: str, optional
            Variable used as instance identifier (default: None).
        """
        self.count_var = count_var
        self.instance_dim = instance_dim
        self._data = ds
        self.validate()

        self.sample_dim = ds[count_var].attrs["sample_dimension"]
        self.instance_id_var = instance_id_var
        self._lut = None

        # cache row_size and instance_ids data
        self._row_size = self.ds[self.count_var].to_numpy()

        if self.instance_id_var is None:
            self._instance_ids = self.ds[self.instance_dim].to_numpy()
        else:
            self._instance_ids = self.ds[self.instance_id_var].to_numpy()

        self._set_instance_lut()

    def validate(self):
        """Validate format."""
        verify_contiguous_ragged(self.ds, self.count_var, self.instance_dim)

    def _set_instance_lut(self):
        """
        Set instance lookup-table.
        """
        self._instance_lookup = np.zeros(
            self._instance_ids.max() + 1,
            dtype=np.int64) + self._instance_ids.size
        self._instance_lookup[self._instance_ids] = np.arange(
            self._instance_ids.size)

        self._lut = np.zeros(self._instance_ids.max() + 1, dtype=bool)

    @classmethod
    def from_file(cls,
                  filename: str,
                  count_var: str,
                  instance_dim: str,
                  instance_id_var: str = None,
                  **kwargs):
        """
        Load time series from file.

        Parameters
        ----------
        filename : str
            Filename.
        count_var : str
            Count variable name.
        instance_dim : str
            Instance dimension name.
        instance_id_var: str, optional
            Variable used as instance identifier (default: None).

        Returns
        -------
        data : ContiguousRaggedArray
            ContiguousRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename, **kwargs)
        verify_contiguous_ragged(ds, count_var, instance_dim)

        return cls(ds, count_var, instance_dim, instance_id_var)

    @property
    def ds(self):
        """
        Dataset.

        Returns
        -------
        ds : xr.Dataset
            Contiguous ragged array dataset.
        """
        return self._data

    @property
    def size(self) -> list:
        """
        Number of instances.

        Returns
        -------
        instance_ids : int
            Number of instance.
        """
        return self._instance_ids.size

    @property
    def instance_ids(self) -> list:
        """
        Instance ids

        Returns
        -------
        instance_ids : list of int
            Instance ids.
        """
        return self._instance_ids

    @property
    def instance_variables(self) -> list:
        """
        Instance variables.

        Returns
        -------
        instance_variables : list of str
            Instance variables.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.sample_dim,)) and
            (var != self.sample_dim)
        ]

    def get_instance_variables(self, include_dtype: bool = False) -> list:
        """
        Instance variables.

        Returns
        -------
        instance_variables : list of str
            Instance variables.
        """
        if include_dtype:
            instance_variables = [
                (var, self.ds[var].dtype)
                for var in self.ds.variables
                if (self.ds[var].dims == (self.sample_dim,)) and
                (var != self.sample_dim)
            ]
        else:
            instance_variables = [
                var for var in self.ds.variables
                if (self.ds[var].dims == (self.sample_dim,)) and
                (var != self.sample_dim)
            ]

        return instance_variables

    def sel_instance(self, i: int):
        """Read time series"""
        try:
            idx = self._instance_lookup[i]
        except IndexError:
            data = None
        else:
            if idx == self._instance_ids.size:
                data = None
            else:
                start = self._row_size[:idx].sum()
                end = start + self._row_size[idx]
                data = self.ds.isel({
                    self.sample_dim: slice(start, end),
                    self.instance_dim: idx
                })

        return data

    def sel_instances(self, i: np.ndarray) -> xr.Dataset:
        """
        Read time series for given instance IDs using a LUT and preserve order.

        Parameters
        ----------
        i : np.ndarray
            Array of instance IDs.

        Returns
        -------
        ds : xr.Dataset
            Dataset containing the selected instances in the correct order.
        """
        i = np.asarray(i)
        idx_array = self._instance_lookup[i]

        # mark selected instances in lookup table
        self._lut[idx_array] = True

        # map each sample index to its instance index
        obs = np.repeat(np.arange(len(self._row_size)), self._row_size)

        data = self.ds.sel({
            self.sample_dim: self._lut[obs],
            self.instance_dim: idx_array,
        })

        # reset lookup table
        self._lut[:] = False

        # sort the selected data to match the order in i
        # assuming instance_dim is coordinate-based and one-dimensional
        _, order = np.unique(i, return_inverse=True)

        data = data.isel({self.instance_dim: order})
        data = data.assign_coords({self.instance_dim: i})

        return data

    def __iter__(self):
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self):
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_indexed(self, index_var: str = "locationIndex"):
        """
        Convert to indexed ragged array.

        Parameters
        ----------
        index_var : str, optional
            Name of the index variable to create (default: "locationIndex").

        Returns
        -------
        data : IndexedRaggedArray
            Indexed ragged array time series.
        """
        ds = contiguous_to_indexed(self.ds, self.sample_dim, self.instance_dim,
                                   self.count_var, index_var)

        return IndexedRaggedArray(ds, index_var, self.sample_dim)

    def to_incomplete(self):
        """
        Convert to an incomplete multidimensional array (CF 9.3.2).

        Each instance's samples are packed into the leading columns of a dense
        (instance x element) array and the rest padded with fill values.

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array time series.
        """
        instance_id_var = (
            self.instance_id_var if self.instance_id_var is not None
            else self.instance_dim
        )
        reshaped_ds = contiguous_to_incomplete(
            self.ds,
            self.sample_dim,
            self.instance_dim,
            self.count_var,
            element_dim=self.sample_dim,
            instance_id_var=instance_id_var,
        )
        return IncompleteMultidimArray(
            reshaped_ds, self.instance_dim, self.sample_dim)

    def to_orthogonal(self, element_coord: str, element_dim: str = None,
                      strict: bool = True):
        """
        Convert to an orthogonal multidimensional array (CF 9.3.1).

        All instances must share the same set of ``element_coord`` values (e.g.
        the same time axis); the samples are pivoted onto that shared
        coordinate.

        Parameters
        ----------
        element_coord : str
            Name of the per-sample coordinate defining the shared element axis.
        element_dim : str, optional
            Name of the resulting element dimension (default: ``element_coord``).
        strict : bool, optional
            If True (default), raise if the instances do not form a complete
            grid on ``element_coord``.

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array time series.
        """
        element_dim = element_dim or element_coord
        instance_id_var = (
            self.instance_id_var if self.instance_id_var is not None
            else self.instance_dim
        )
        reshaped_ds = contiguous_to_orthogonal(
            self.ds,
            self.sample_dim,
            self.instance_dim,
            self.count_var,
            element_coord,
            element_dim=element_dim,
            instance_id_var=instance_id_var,
            strict=strict,
        )
        return OrthogonalMultidimArray(
            reshaped_ds, self.instance_dim, element_dim,
            element_coord=element_coord)

    def to_point_data(self):
        """
        Convert to point data.

        Instance-level variables are broadcast to the sample dimension so that
        every observation carries its instance's coordinates.

        Returns
        -------
        data : PointData
            Point data.
        """
        ds = contiguous_to_point(self.ds, self.sample_dim, self.instance_dim,
                                 self.count_var)
        return PointData(ds, self.sample_dim)

    def apply(self, func):
        """
        Apply function on each instance.
        """
        return self.ds.groupby(self.sample_dim).map(func)


class IndexedRaggedArray:
    """
    Indexed ragged array representation (CF convention).

    In an indexed ragged array representation, the dataset is structured
    to store variable-length data (e.g., time series with varying lengths)
    compactly. To achieve this, auxiliary indexing variables that map the
    flat array storage to meaningful groups (e.g. locations).

    If the instance dimension exists as a variable, it is assumed that the
    values represent the identfiers for each instance otherwise they counting
    upwards from 0.

    Attributes
    ----------
    index_var : str
        The indexed ragged array representation must contain an index
        variable, which must be an integer type, and must have the sample
        dimension as its single dimension.
        The index variable can be identified by having an attribute
        'instance_dimension' whose value is the instance dimension.
    sample_dim : str
        Name of the sample dimension. The sample dimension indicates the
        number of instances (e.g. stations, locations).
    instance_dim : str
        The name of the instance dimension. The value is defined by
        the 'instance_dimension' attribute, which must be present on the
        index variable. All variables having the instance dimension are
        instance variables, i.e. variables holding time series data.
    ds : xarray.Dataset
        Indexed ragged array dataset.
    instance_variables : list
        List of instance variables.
    instance_ids : list
        List of instance ids.

    Methods
    -------
    sel_instance(i)
        Read time series for given instance.
    iter()
        Yield time series for each instance.
    """

    def __init__(self, ds: xr.Dataset, index_var: str, sample_dim: str):
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data in indexed ragged array structure.
        index_var : str
            Index variable name.
        sample_dim : str
            Sample dimension name.
        """
        self.index_var = index_var
        self.sample_dim = sample_dim
        self._data = ds.set_coords(self.index_var).set_xindex(self.index_var)
        self.validate()

        self.instance_dim = ds[index_var].attrs["instance_dimension"]
        self._instance_lut = None
        self._lut = None
        self._set_instance_lut()

    def validate(self):
        """Validate format."""
        verify_indexed_ragged(self.ds, self.index_var, self.sample_dim)

    def __repr__(self):
        """"""
        return self.ds.__repr__()

    def _set_instance_lut(self):
        """
        Set instance lookup-table.
        """
        if self.instance_dim in self.ds:
            instance_ids = self.ds[self.instance_dim].to_numpy()
            self._instance_lut = np.zeros(
                instance_ids.max() + 1, dtype=np.int64) - 1
            self._instance_lut[instance_ids] = np.arange(instance_ids.size)
        else:
            instance_ids = np.unique(self.ds[self.index_var])
            self._instance_lut = np.arange(instance_ids.size)

        self._lut = np.zeros(instance_ids.max() + 1, dtype=bool)

    @classmethod
    def from_file(cls, filename: str, index_var: str, sample_dim: str):
        """
        Read data from file.

        Parameters
        ----------
        filename : str
            Filename.
        index_var : str
            Index variable name.
        sample_dim : str
            Sample dimension name.

        Returns
        -------
        data : IndexRaggedArray
            IndexRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename)
        verify_indexed_ragged(ds, index_var, sample_dim)
        return cls(ds, index_var, sample_dim)

    def save(self, filename: str):
        """
        Write data to file.

        Parameters
        ----------
        filename : str
            Filename.
        """
        suffix = Path(filename).suffix

        if suffix == ".nc":
            self.ds.to_netcdf(filename)
        elif suffix == ".zarr":
            self.ds.to_zarr(filename)
        else:
            raise ValueError(f"Unknown file suffix '{suffix}' "
                             "(.nc and .zarr supported)")

    @property
    def ds(self) -> xr.Dataset:
        """
        Dataset.

        Returns
        -------
        ds : xr.Dataset
            Indexed ragged array dataset.
        """
        return self._data

    @property
    def size(self) -> list:
        """
        Number of instances.

        Returns
        -------
        instance_ids : int
            Number of instance.
        """
        return self.instance_ids.size

    @property
    def instance_ids(self) -> list:
        """
        Instance ids.

        Returns
        -------
        instance_ids : list of int
            Instance ids.
        """
        return self.ds[self.instance_dim].values

    @property
    def instance_variables(self) -> list:
        """
        Instance variables.

        Returns
        -------
        instance_variables : list of str
            Instance variables.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.sample_dim,)) and
            (var != self.sample_dim)
        ]

    def sel_instance(self, i: int) -> xr.Dataset:
        """
        Read time series.

        Parameters
        ----------
        i : int
            Instance identifier.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        data = self.ds.sel({
            self.index_var: self._instance_lut[i],
            self.instance_dim: i
        })

        # reset index variable or drop index variable (my preference)?
        data[self.index_var] = (self.sample_dim,
                                np.zeros(data[self.index_var].size, dtype=int))

        return data

    def sel_instances(self,
                      i: np.array,
                      ignore_missing: bool = True) -> xr.Dataset:
        """
        Select multiple instances (time series).

        Parameters
        ----------
        i : numpy.array
            Instance identifier.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        # check for duplicates in instance identifier?
        i = np.asarray(i)

        # check if any missing instance ids have been selected
        if ignore_missing:
            valid = np.where(self._instance_lut[i] != -1)[0]
            if valid.size == 0:
                raise ValueError("No valid instances selected")
        else:
            if np.any(self._instance_lut[np.asarray(i)] == -1):
                raise ValueError("Missing instances selected")
            else:
                valid = np.ones(i.size, dtype=bool)

        i = i[valid]

        # initialize look-up table
        self._lut[self._instance_lut[i]] = True

        data = self.ds.sel(
            {self.index_var: self._lut[self.ds[self.index_var]]})

        # reset index variable
        index = data[self.instance_dim][data[self.index_var]].to_numpy()
        lut = np.zeros(index.max() + 1, dtype=np.int64)
        # n_inst = self.ds.sel({self.instance_dim: i}).sizes[self.instance_dim]
        n_inst = i.size
        lut[i] = np.arange(n_inst)
        data[self.index_var] = (self.sample_dim, lut[index])
        data = data.sel({self.instance_dim: i})

        # copy attributes
        data[self.index_var].attrs = self.ds[self.index_var].attrs

        # reset look-up table
        self._lut[:] = False

        return IndexedRaggedArray(data, self.index_var, self.sample_dim)

    def __iter__(self) -> xr.Dataset:
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self) -> xr.Dataset:
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_contiguous(self,
                      count_var: str = "row_size") -> ContiguousRaggedArray:
        """
        Convert to contiguous ragged array.

        Parameters
        ----------
        count_var : str, optional
            Count variable (default: "row_size").

        Returns
        -------
        data : ContiguousRaggedArray
            Contiguous ragged array time series.
        """
        ds = indexed_to_contiguous(self.ds, self.sample_dim, self.instance_dim,
                                   count_var, self.index_var)

        return ContiguousRaggedArray(ds, count_var, self.instance_dim)

    def to_incomplete(self) -> "IncompleteMultidimArray":
        """
        Convert to an incomplete multidimensional array (via a contiguous array).

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array time series.
        """
        return self.to_contiguous().to_incomplete()

    def to_orthogonal(self, element_coord: str, element_dim: str = None,
                      strict: bool = True) -> "OrthogonalMultidimArray":
        """
        Convert to an orthogonal multidimensional array (via a contiguous array).

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array time series.
        """
        return self.to_contiguous().to_orthogonal(
            element_coord, element_dim=element_dim, strict=strict)

    def to_point_data(self):
        """
        Convert to point data.

        Instance-level variables are broadcast to the sample dimension so that
        every observation carries its instance's coordinates.

        Returns
        -------
        data : PointData
            Point data.
        """
        ds = indexed_to_point(self.ds, self.sample_dim, self.instance_dim,
                              self.index_var)
        return PointData(ds, self.sample_dim)

    def apply(self, func):
        """
        Apply function on each instance.
        """
        # return self.ds.groupby(self.sample_dim).apply(func)
        return self.ds.groupby(self.sample_dim).map(func)

    def append(self, ds: xr.Dataset):
        """
        Append indexed ragged array time series.

        Parameters
        ----------
        ds : xarray.Dataset
            Indexed ragged array time series.
        """
        verify_indexed_ragged(ds, self.index_var, self.sample_dim)

        # use instance_id in index variable if instance dimension is a variable
        if self.instance_dim in self.ds:
            self.ds[self.index_var] = self.ds[self.instance_dim][self.ds[
                self.index_var]]

        if self.instance_dim in ds:
            # assume this has been set or not?
            ds = ds.set_coords(self.index_var).set_xindex(self.index_var)
            ds[self.index_var] = (
                self.sample_dim,
                ds[self.instance_dim].values[ds[self.index_var]])

        self._data = xr.combine_nested([self._data, ds], self.sample_dim)

        if self.instance_dim in self.ds:
            instance_ids = self.ds[self.instance_dim].to_numpy()
            lut = np.zeros(instance_ids.max() + 1, dtype=np.int64)
            lut[instance_ids] = np.arange(self.ds[self.instance_dim].size)
            self.ds[self.index_var] = (self.sample_dim,
                                       lut[self.ds[self.index_var]])

        self._set_instance_lut()

