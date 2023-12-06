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

from datetime import datetime
from pathlib import Path

import xarray as xr
import numpy as np

from fibgrid.realization import FibGrid


class RaggedXArrayIOBase:
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


class ASCAT_NetCDF4(RaggedXArrayIOBase):
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


# nice to define these outside the classes so there's no need to create multiple
grid_6_25 = FibGrid(6.25)
grid_12_5 = FibGrid(12.5)

class ASCAT_H129_Cell(ASCAT_NetCDF4):
    grid = grid_6_25
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid.get_cells()
    max_cell = possible_cells.max()
    min_cell = possible_cells.min()

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.grid = ASCAT_H129_Cell.grid
        self.grid_cell_size = ASCAT_H129_Cell.grid_cell_size
        self.fn_format = ASCAT_H129_Cell.fn_format
        self.possible_cells = ASCAT_H129_Cell.possible_cells
        self.max_cell = ASCAT_H129_Cell.max_cell
        self.min_cell = ASCAT_H129_Cell.min_cell


class ASCAT_SIG0_6_25_Cell(ASCAT_NetCDF4):
    grid = grid_6_25
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid.get_cells()
    max_cell = possible_cells.max()
    min_cell = possible_cells.min()

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.grid = ASCAT_SIG0_6_25_Cell.grid
        self.grid_cell_size = ASCAT_SIG0_6_25_Cell.grid_cell_size
        self.fn_format = ASCAT_SIG0_6_25_Cell.fn_format
        self.possible_cells = ASCAT_SIG0_6_25_Cell.possible_cells
        self.max_cell = ASCAT_SIG0_6_25_Cell.max_cell
        self.min_cell = ASCAT_SIG0_6_25_Cell.min_cell


class ASCAT_SIG0_12_5_Cell(ASCAT_NetCDF4):
    grid = grid_12_5
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid.get_cells()
    max_cell = possible_cells.max()
    min_cell = possible_cells.min()

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.grid = ASCAT_SIG0_12_5_Cell.grid
        self.grid_cell_size = ASCAT_SIG0_12_5_Cell.grid_cell_size
        self.fn_format = ASCAT_SIG0_12_5_Cell.fn_format
        self.possible_cells = ASCAT_SIG0_12_5_Cell.possible_cells
        self.max_cell = ASCAT_SIG0_12_5_Cell.max_cell
        self.min_cell = ASCAT_SIG0_12_5_Cell.min_cell

class SwathIOBase:
    """
    Base class for reading swath data.
    Writes ragged array cell data in indexed or contiguous format.
    """

    def __init__(self, source, engine, **kwargs):
        self.source = source
        self.engine = engine
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
            # print(self._ds.chunks)

            chunks = None

        elif isinstance(source, (str, Path)):
            self._ds = xr.open_dataset(source, engine=engine, **kwargs, decode_cf=False)

        elif isinstance(source, xr.Dataset):
            self._ds = source

        self._kwargs = kwargs

        # if "time" in self._ds.dims:
        #     self._ds = self._ds.rename_dims({"time": "obs"})

        if chunks is not None:
            self._ds = self._ds.chunk(chunks)

        # if "row_size" in self._ds.data_vars:
        #     self._ds = self._ensure_indexed(self._ds)
        # else:
            # self.ra_type = "indexed"


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


class ASCAT_H129_Swath(SwathIOBase):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOPA-6.25-H129_C_LIIB_{date}_*_*____.nc"
    fn_read_fmt = lambda timestamp: {"date": timestamp.strftime("%Y%m%d*")}
    sf_pattern = {"year_folder": "{year}"}
    sf_read_fmt = lambda timestamp: {"year_folder": {"year": f"{timestamp.year}"}}
    date_format = "%Y%m%d%H%M%S"
    grid = grid_6_25
    grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
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

    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)
        self.fn_pattern = ASCAT_H129_Swath.fn_pattern
        self.fn_read_fmt = ASCAT_H129_Swath.fn_read_fmt
        self.sf_pattern = ASCAT_H129_Swath.sf_pattern
        self.sf_read_fmt = ASCAT_H129_Swath.sf_read_fmt
        self.date_format = ASCAT_H129_Swath.date_format
        self.grid = ASCAT_H129_Swath.grid
        self.grid_cell_size = ASCAT_H129_Swath.grid_cell_size
        self.ts_dtype = ASCAT_H129_Swath.ts_dtype
        # self.cell_vals = self.grid.gpi2cell(self._ds.location_id.values)

    def _preprocess(self, ds):
        """
        Preprocessing function for opening multifile datasets
        """
        return ds

    def _read_location_ids(self, location_ids):
        """
        Read data for a list of location_ids
        """
        idxs = np.in1d(self._ds.location_id, location_ids, kind='table')
        idxs = np.array([1 if id in location_ids else 0 for id in self._ds.location_id.values])
        if not np.any(idxs):
            return None
        ds = self._ds.isel(obs=idxs)
        # print(self._ds)
        # print(location_ids)
        # lons, lats = self.grid.gpi2lonlat(location_ids)
        # ds = self._ds.sel({"lon": lons, "lat": lats})

        return ds

    def _read_cell(self, cell):
        """
        Read data for a single cell
        """
        idxs = np.where(self.cell_vals==cell)[0]
        if not np.any(idxs):
            return None

        ds = self._ds.isel(obs=idxs)
        return ds

    def read(self, cell=None, location_id=None, **kwargs):
        if location_id is not None:
            return self._read_location_ids(location_id)
        if cell is not None:
            return self._read_cell(cell)
        return xr.decode_cf(self._ds, mask_and_scale=False)

    def write(self, filename, **kwargs):
        self._ds.to_netcdf(filename, **kwargs)

ascat_io_classes = {
    "H129": {
        "cell": ASCAT_H129_Cell,
        "swath": ASCAT_H129_Swath,
    }
}
