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
from datetime import datetime
from pathlib import Path

import xarray as xr
import numpy as np

from fibgrid.realization import FibGrid
import netCDF4

# By @hmaarrfk on github: https://github.com/pydata/xarray/issues/1672
def _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size):
    # For time deltas, we must ensure that we use the same encoding as
    # what was previously stored.
    # We likely need to do this as well for variables that had custom
    # encodings too
    if hasattr(nc_variable, 'calendar'):

        data.encoding = {
            'units': nc_variable.units,
            'calendar': nc_variable.calendar,
        }
    data_encoded = xr.conventions.encode_cf_variable(data) # , name=name)
    left_slices = data.dims.index(expanding_dim)
    right_slices = data.ndim - left_slices - 1
    nc_slice   = (slice(None),) * left_slices + (slice(nc_shape, nc_shape + added_size),) + (slice(None),) * (right_slices)
    nc_variable[nc_slice] = data_encoded.data

# By @hmaarrfk on github: https://github.com/pydata/xarray/issues/1672
def append_to_netcdf(filename, ds_to_append, unlimited_dims):
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]

    if len(unlimited_dims) != 1:
        # TODO: change this so it can support multiple expanding dims
        raise ValueError(
            "We only support one unlimited dim for now, "
            f"got {len(unlimited_dims)}.")

    unlimited_dims = list(set(unlimited_dims))
    expanding_dim = unlimited_dims[0]

    with netCDF4.Dataset(filename, mode='a') as nc:
        nc.set_auto_maskandscale(False)
        # nc_dims = set(nc.dimensions.keys())
        nc_coord = nc.dimensions[expanding_dim]
        nc_shape = len(nc_coord)

        added_size = ds_to_append.dims[expanding_dim]
        variables, attrs = xr.conventions.encode_dataset_coordinates(ds_to_append)

        for name, data in variables.items():
            if expanding_dim not in data.dims:
                # Nothing to do, data assumed to the identical
                continue

            nc_variable = nc[name]
            _expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size)

class RaggedXArrayCellIOBase:
    """
    Base class for ascat xarray IO classes
    """

    def __init__(self, source, engine, **kwargs):
        self.source = source
        self.engine = engine
        # if filename is a generator, use open_mfdataset
        # else use open_dataset
        if isinstance(source, list):
            self._ds = xr.open_mfdataset(source,
                                         engine=engine,
                                         decode_cf=False,
                                         mask_and_scale=False,
                                         **kwargs)
        elif isinstance(source, (str, Path)):
            self._ds = xr.open_dataset(source,
                                       engine=engine,
                                       decode_cf=False,
                                       mask_and_scale=False,
                                       **kwargs)
        elif isinstance(source, xr.Dataset):
            self._ds = source

        self._kwargs = kwargs

        if "time" in self._ds.dims:
            self._ds = self._ds.rename_dims({"time": "obs"})

        chunks = kwargs.pop("chunks", None)
        if chunks is not None:
            self._ds = self._ds.chunk(chunks)

        if "row_size" in self._ds.data_vars:
            self._ds = self._ensure_indexed(self._ds)

    @property
    def date_range(self):
        """
        Return date range of dataset.

        Returns
        -------
        tuple
            Date range of dataset.
        """

        dates = xr.decode_cf(self._ds[["time"]], mask_and_scale=False)
        return dates.time.min().values, dates.time.max().values

    def read(self, location_id=None, **kwargs):
        """
        Read data from file. Should be implemented by subclasses.

        Parameters
        ----------
        location_id : int, list, optional
            Location id(s) to read.
        **kwargs
            Additional keyword arguments passed to the read method.

        Returns
        -------
        xarray.Dataset
            Dataset containing the data for any specified location_id(s),
            or all location_ids in the file if none are specified.
        """
        raise NotImplementedError

    def write(self, filename, **kwargs):
        """
        Write data to file. Should be implemented by subclasses.

        Parameters
        ----------
        filename : str, Path
            Filename to write data to.
        **kwargs
            Additional keyword arguments passed to the write method.
        """
        raise NotImplementedError

    def _ensure_indexed(self, ds):
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
        # if "time" in ds.dims:
        #     ds = ds.rename_dims({"time": "obs"})

        if ds is None or "locationIndex" in ds.data_vars:
            return ds

        row_size = np.where(ds["row_size"].values > 0,
                            ds["row_size"].values, 0)

        locationIndex = np.repeat(np.arange(row_size.size), row_size)
        ds["locationIndex"] = ("obs", locationIndex)
        ds = ds.drop_vars(["row_size"])
        # if self.ra_type == "contiguous":
        #     row_size = np.where(ds["row_size"].values > 0,
        #                         ds["row_size"].values, 0)

        #     locationIndex = np.repeat(np.arange(row_size.size), row_size)
        #     ds["locationIndex"] = ("obs", locationIndex)
        #     ds = ds.drop_vars(["row_size"])

        return ds[self._var_order(ds)]

    def _ensure_contiguous(self, ds):
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
        if ds is None or "row_size" in ds.data_vars:
            return ds

        ds = ds.sortby(["locationIndex", "time"])
        idxs, sizes = np.unique(ds.locationIndex, return_counts=True)
        row_size = np.zeros_like(ds.location_id.values)
        row_size[idxs] = sizes
        ds["row_size"] = ("locations", row_size)
        ds = ds.drop_vars(["locationIndex"])
        return ds[self._var_order(ds)]

    def close(self):
        """
        Close file.
        """
        if self._ds is not None:
            self._ds.close()
            self._ds = None

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


class AscatNetCDFCellBase(RaggedXArrayCellIOBase):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)

    def read(self, location_id=None, **kwargs):
        if location_id is not None:
            if isinstance(location_id, (int, np.int64, np.int32)):
                return self._ensure_indexed(self._read_location_id(location_id))
            elif isinstance(location_id, list):
                return self._ensure_indexed(self._read_location_ids(location_id))
            else:
                raise ValueError("location_id must be int or list of ints")

        return xr.decode_cf(self._ds, mask_and_scale=False)

    def write(self, filename, ra_type="contiguous", **kwargs):
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
        out_ds = self._ds
        if ra_type == "contiguous":
            out_ds = self._ensure_contiguous(out_ds)
        out_ds = out_ds[self._var_order(out_ds)]
        out_ds = self._set_attributes(out_ds)
        encoding = self._create_encoding(out_ds)
        out_ds.encoding["unlimited_dims"] = []

        out_ds.to_netcdf(filename, encoding=encoding, **kwargs)

    def _read_location_id(self, location_id):
        """
        Read data for a single location_id
        """
        if location_id not in self._ds.location_id.values:
            return None

        idx = np.where(self._ds.location_id.values == location_id)[0][0]
        locationIndex = np.where(self._ds.locationIndex.values == idx)[0]
        ds = self._ds.isel(obs=locationIndex, locations=idx)

        return ds

    def _read_location_ids(self, location_ids):
        """
        Read data for a list of location_ids.
        If there are location_ids not in the dataset,
        they are ignored.
        """
        idxs = np.where(np.isin(self._ds.location_id.values, location_ids))[0]
        locationIndex = np.where(self._ds.locationIndex.values in idxs)[0]
        ds = self._ds.isel(obs=locationIndex, locations=idxs)

        return ds

    def _var_order(self, ds):
        """
        Returns a reasonable variable order for a ragged array dataset,
        based on that used in existing datasets.

        Puts the count/index variable first depending on the ragged array type,
        then lon, lat, alt, location_id, location_description, and time,
        followed by the rest of the variables in the dataset.

        If this order is not appropriate for a particular dataset,
        the user can always reorder the variables manually or override
        this method in a child class specific to their dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset.

        Returns
        -------
        order : list of str
            List of dataset variable names in the determined order.
        """
        if "row_size" in ds.data_vars:
            first_var = "row_size"
        elif "locationIndex" in ds.data_vars:
            first_var = "locationIndex"
        else:
            raise ValueError("No row_size or locationIndex in ds."
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
        order.extend([v for v in ds.data_vars if v not in order])

        return order

    @staticmethod
    def _set_attributes(ds, attributes=None):
        """
        TODO: ideally this should be gotten from the ioclass used to write the dataset (or performed there)
        Set default attributes for a contiguous or indexed ragged dataset.

        Parameters
        ----------
        ds : xarray.Dataset, Path
            Dataset.
        attributes : dict, optional
            Attributes.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with attributes.
        """
        # needed to set custom attributes - but do we need that here?
        attributes = attributes or {}
        # attributes = {}

        if "row_size" in ds.data_vars:
            first_var = "row_size"
        elif "locationIndex" in ds.data_vars:
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
            ds[var] = ds[var].assign_attrs(attrs)
            if var in [
                    "row_size", "locationIndex", "location_id",
                    "location_description"
            ]:
                ds[var].encoding["coordinates"] = None

        date_created = datetime.now().isoformat(" ", timespec="seconds")
        ds.attrs["date_created"] = date_created

        return ds

    @staticmethod
    def _create_encoding(ds, custom_encoding=None):
        """
        TODO: ideally this should be defined in one of the levels of parent class,
        and then passed a custom encoding from the metadata... or something

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
        ds : xarray.Dataset
            Dataset.
        custom_encoding : dict, optional
            Custom encodings.

        Returns
        -------
        ds : xarray.Dataset
            Dataset with encodings.
        """
        if custom_encoding is None:
            custom_encoding = {}

        if "row_size" in ds.data_vars:
            first_var = "row_size"
        elif "locationIndex" in ds.data_vars:
            first_var = "locationIndex"
        else:
            raise ValueError("No row_size or locationIndex in ds."
                             " Cannot determine if indexed or ragged")

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
            for var, dtype in ds.dtypes.items()
        })

        encoding = {**default_encoding, **custom_encoding}

        return encoding


class SwathIOBase:
    """
    Base class for reading swath data.
    Writes ragged array cell data in indexed or contiguous format.
    """

    def __init__(self, source, engine, **kwargs):
        self.source = source
        self.engine = engine
        self._cell_vals = None
        # if filename is a generator, use open_mfdataset
        # else use open_dataset
        chunks = kwargs.pop("chunks", None)
        if isinstance(source, list):
            self._ds = xr.open_mfdataset(source,
                                         engine=engine,
                                         decode_cf=False,
                                         # preprocess=self._preprocess,
                                         concat_dim="obs",
                                         combine="nested",
                                         chunks=(chunks or "auto"),
                                         **kwargs)
            chunks = None

        elif isinstance(source, (str, Path)):
            self._ds = xr.open_dataset(source, engine=engine, **kwargs, decode_cf=False)

        elif isinstance(source, xr.Dataset):
            self._ds = source

        self._kwargs = kwargs

        if chunks is not None:
            self._ds = self._ds.chunk(chunks)

    def read(self, cell=None, location_id=None):
        if location_id is not None:
            return self._read_location_ids(location_id)
        if cell is not None:
            return self._read_cell(cell)
        return xr.decode_cf(self._ds, mask_and_scale=False)

    def write(self, filename, mode="w", **kwargs):
        if mode == "a":
            if os.path.exists(filename):
                append_to_netcdf(filename, self._ds, unlimited_dims=["obs"])
                return
        self._ds.to_netcdf(filename, unlimited_dims=["obs"], **kwargs)

    def _read_location_ids(self, location_ids):
        """
        Read data for a list of location_ids
        """
        idxs = np.in1d(self._ds.location_id, location_ids)
        idxs = np.array([1 if id in location_ids else 0
                         for id in self._ds.location_id.values])
        if not np.any(idxs):
            return None
        ds = self._ds.isel(obs=idxs)
        return ds

    def _read_cell(self, cell):
        """
        Read data for a single cell
        """
        if self._cell_vals is None:
            self._cell_vals = self.grid.gpi2cell(self._ds.location_id.values)
        idxs = np.where(self.cell_vals == cell)[0]
        if not np.any(idxs):
            return None

        ds = self._ds.isel(obs=idxs)
        return ds

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

    def close(self):
        """
        Close the dataset.
        """
        self._ds.close()


class CellGridCache:
    """
    Cache for CellGrid objects.
    """
    def __init__(self):
        self.grids = {}

    def fetch_or_store(self, key, cell_grid_type, *args):
        """
        Get a CellGrid object with the specified arguments.

        Parameters
        ----------
        """
        if key not in self.grids:
            self.grids[key] = dict()
            self.grids[key]["grid"] = cell_grid_type(*args)
            self.grids[key]["possible_cells"] = self.grids[key]["grid"].get_cells()
            self.grids[key]["max_cell"] = self.grids[key]["possible_cells"].max()
            self.grids[key]["min_cell"] = self.grids[key]["possible_cells"].min()

        return self.grids[key]


grid_cache = CellGridCache()


# Define dataset-specific classes.
class AscatH129Cell(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)


class AscatSIG0Cell6250m(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)


class AscatSIG0Cell12500m(AscatNetCDFCellBase):
    grid_info = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        # self.grid = ASCAT_SIG0_12_5_Cell.grid
        # self.grid_cell_size = ASCAT_SIG0_12_5_Cell.grid_cell_size
        # self.fn_format = ASCAT_SIG0_12_5_Cell.fn_format
        # self.possible_cells = ASCAT_SIG0_12_5_Cell.possible_cells
        # self.max_cell = ASCAT_SIG0_12_5_Cell.max_cell
        # self.min_cell = ASCAT_SIG0_12_5_Cell.min_cell


class AscatH129Swath(SwathIOBase):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-6.25-H129_C_LIIB_{date}_*_*____.nc"
    sf_pattern = {"year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = ["backscatter", "incidence_angle", "azimuth_angle", "kp"]
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("backscatter_for", np.float32),
        ("backscatter_mid", np.float32),
        ("backscatter_aft", np.float32),
        ("incidence_angle_for", np.float32),
        ("incidence_angle_mid", np.float32),
        ("incidence_angle_aft", np.float32),
        ("azimuth_angle_for", np.float32),
        ("azimuth_angle_mid", np.float32),
        ("azimuth_angle_aft", np.float32),
        ("kp_for", np.float32),
        ("kp_mid", np.float32),
        ("kp_aft", np.float32),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("backscatter40", np.float32),
        ("slope40", np.float32),
        ("curvature40", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("surface_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
    ])

    # @staticmethod
    # def metadata(resolution):
    #     return {
    #         "fn_pattern": f"W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-{resolution:g}"+"-H129_C_LIIB_{date}_*_*____.nc",
    #         "sf_pattern": {"year_folder": "{year}"},
    #         "date_format": "%Y%m%d%H%M%S",
    #         "grid": grid_cache[resolution]["grid"],
    #         "grid_cell_size": 5,
    #         "cell_fn_format": "{:04d}.nc",
    #         "beams_vars": ["backscatter", "incidence_angle", "azimuth_angle", "kp"],
    #         "ts_dtype": np.dtype([
    #             ("sat_id", np.int8),
    #             ("as_des_pass", np.int8),
    #             ("swath_indicator", np.int8),
    #             ("backscatter_for", np.float32),
    #             ("backscatter_mid", np.float32),
    #             ("backscatter_aft", np.float32),
    #             ("incidence_angle_for", np.float32),
    #             ("incidence_angle_mid", np.float32),
    #             ("incidence_angle_aft", np.float32),
    #             ("azimuth_angle_for", np.float32),
    #             ("azimuth_angle_mid", np.float32),
    #             ("azimuth_angle_aft", np.float32),
    #             ("kp_for", np.float32),
    #             ("kp_mid", np.float32),
    #             ("kp_aft", np.float32),
    #             ("surface_soil_moisture", np.float32),
    #             ("surface_soil_moisture_noise", np.float32),
    #             ("backscatter40", np.float32),
    #             ("slope40", np.float32),
    #             ("curvature40", np.float32),
    #             ("surface_soil_moisture_sensitivity", np.float32),
    #             ("correction_flag", np.uint8),
    #             ("processing_flag", np.uint8),
    #             ("surface_flag", np.uint8),
    #             ("snow_cover_probability", np.int8),
    #             ("frozen_soil_probability", np.int8),
    #             ("wetland_fraction", np.int8),
    #             ("topographic_complexity", np.int8),
    #         ]),
    #     }

    @staticmethod
    def fn_read_fmt(timestamp):
        return {"date": timestamp.strftime("%Y%m%d*")}

    @staticmethod
    def sf_read_fmt(timestamp):
        return {"year_folder": {"year": f"{timestamp.year}"}}

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)
        # metadata = self.metadata(resolution)
        # self.resolution = resolution
        # self.fn_pattern = metadata["fn_pattern"]
        # self.sf_pattern = metadata["sf_pattern"]
        # self.date_format = metadata["date_format"]
        # self.grid = metadata["grid"]
        # self.grid_cell_size = metadata["grid_cell_size"]
        # self.ts_dtype = metadata["ts_dtype"]


class AscatSIG0Swath6250m(SwathIOBase):
    """
    Class for reading and writing ASCAT sigma0 swath data.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP[ABC]-6.25_C_LIIB_*_*_{date}____*.nc"
    sf_pattern = {"year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid = grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = [
        "backscatter",
        "backscatter_std",
        "incidence_angle",
        "azimuth_angle",
        "kp",
        "n_echos",
        "all_backscatter",
        "all_backscatter_std",
        "all_incidence_angle",
        "all_azimuth_angle",
        "all_kp",
        "all_n_echos",
    ],
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("backscatter_for", np.float32),
        ("backscatter_mid", np.float32),
        ("backscatter_aft", np.float32),
        ("backscatter_std_for", np.float32),
        ("backscatter_std_mid", np.float32),
        ("backscatter_std_aft", np.float32),
        ("incidence_angle_for", np.float32),
        ("incidence_angle_mid", np.float32),
        ("incidence_angle_aft", np.float32),
        ("azimuth_angle_for", np.float32),
        ("azimuth_angle_mid", np.float32),
        ("azimuth_angle_aft", np.float32),
        ("kp_for", np.float32),
        ("kp_mid", np.float32),
        ("kp_aft", np.float32),
        ("n_echos_for", np.int8),
        ("n_echos_mid", np.int8),
        ("n_echos_aft", np.int8),
        ("all_backscatter_for", np.float32),
        ("all_backscatter_mid", np.float32),
        ("all_backscatter_aft", np.float32),
        ("all_backscatter_std_for", np.float32),
        ("all_backscatter_std_mid", np.float32),
        ("all_backscatter_std_aft", np.float32),
        ("all_incidence_angle_for", np.float32),
        ("all_incidence_angle_mid", np.float32),
        ("all_incidence_angle_aft", np.float32),
        ("all_azimuth_angle_for", np.float32),
        ("all_azimuth_angle_mid", np.float32),
        ("all_azimuth_angle_aft", np.float32),
        ("all_kp_for", np.float32),
        ("all_kp_mid", np.float32),
        ("all_kp_aft", np.float32),
        ("all_n_echos_for", np.int8),
        ("all_n_echos_mid", np.int8),
        ("all_n_echos_aft", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        """
        Format a timestamp to search as YYYYMMDD*, for use in a regex
        that will match all files covering a single given date.

        Parameters
        ----------
        timestamp: datetime.datetime
            Timestamp to format

        Returns
        -------
        dict
            Dictionary of formatted strings
        """
        return {"date": timestamp.strftime("%Y%m%d*")}

    @staticmethod
    def sf_read_fmt(timestamp):
        return {"year_folder": {"year": f"{timestamp.year}"}}

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)

class AscatSIG0Swath12500m(SwathIOBase):
    """
    Class for reading and writing ASCAT sigma0 swath data.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP[ABC]-12.5_C_LIIB_*_*_{date}____*.nc"
    sf_pattern = {"year_folder": "{year}"}
    date_format = "%Y%m%d%H%M%S"
    grid = grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)["grid"]
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = [
        "backscatter",
        "backscatter_std",
        "incidence_angle",
        "azimuth_angle",
        "kp",
        "n_echos",
        "all_backscatter",
        "all_backscatter_std",
        "all_incidence_angle",
        "all_azimuth_angle",
        "all_kp",
        "all_n_echos",
    ],
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("backscatter_for", np.float32),
        ("backscatter_mid", np.float32),
        ("backscatter_aft", np.float32),
        ("backscatter_std_for", np.float32),
        ("backscatter_std_mid", np.float32),
        ("backscatter_std_aft", np.float32),
        ("incidence_angle_for", np.float32),
        ("incidence_angle_mid", np.float32),
        ("incidence_angle_aft", np.float32),
        ("azimuth_angle_for", np.float32),
        ("azimuth_angle_mid", np.float32),
        ("azimuth_angle_aft", np.float32),
        ("kp_for", np.float32),
        ("kp_mid", np.float32),
        ("kp_aft", np.float32),
        ("n_echos_for", np.int8),
        ("n_echos_mid", np.int8),
        ("n_echos_aft", np.int8),
        ("all_backscatter_for", np.float32),
        ("all_backscatter_mid", np.float32),
        ("all_backscatter_aft", np.float32),
        ("all_backscatter_std_for", np.float32),
        ("all_backscatter_std_mid", np.float32),
        ("all_backscatter_std_aft", np.float32),
        ("all_incidence_angle_for", np.float32),
        ("all_incidence_angle_mid", np.float32),
        ("all_incidence_angle_aft", np.float32),
        ("all_azimuth_angle_for", np.float32),
        ("all_azimuth_angle_mid", np.float32),
        ("all_azimuth_angle_aft", np.float32),
        ("all_kp_for", np.float32),
        ("all_kp_mid", np.float32),
        ("all_kp_aft", np.float32),
        ("all_n_echos_for", np.int8),
        ("all_n_echos_mid", np.int8),
        ("all_n_echos_aft", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        """
        Format a timestamp to search as YYYYMMDD*, for use in a regex
        that will match all files covering a single given date.

        Parameters
        ----------
        timestamp: datetime.datetime
            Timestamp to format

        Returns
        -------
        dict
            Dictionary of formatted strings
        """
        return {"date": timestamp.strftime("%Y%m%d*")}

    @staticmethod
    def sf_read_fmt(timestamp):
        return {"year_folder": {"year": f"{timestamp.year}"}}

    def __init__(self, filename,  **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)


ascat_io_classes = {
    "H129": {
        # "cell": ASCAT_H129_Cell,
        "cell": AscatH129Cell,
        "swath": AscatH129Swath,
    },
    "SIG0_6.25": {
        "cell": AscatSIG0Cell6250m,
        "swath": AscatSIG0Swath6250m,
    },
    "SIG0_12.5": {
        "cell": AscatSIG0Cell12500m,
        "swath": AscatSIG0Swath12500m,
    },
}
