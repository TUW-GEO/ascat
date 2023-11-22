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
            self._ds = xr.open_mfdataset(source, engine=engine, **kwargs, decode_cf=False)
        elif isinstance(source, (str, Path)):
            self._ds = xr.open_dataset(source, engine=engine, **kwargs, decode_cf=False)
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

    @property
    def date_range(self):
        dates = xr.decode_cf(self._ds[["time"]])
        return dates.time.min().values, dates.time.max().values

    def _ensure_indexed(self, ds):
        """
        Convert a contiguous dataset to indexed dataset,
        if necessary.
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

        return ds

    def read(self, location_id=None, **kwargs):
        raise NotImplementedError
        # if location_id is not None:
        #     if isinstance(location_id, int):
        #         return self._ensure_indexed(self._read_location_id(location_id))
        #     elif isinstance(location_id, list):
        #         return self._ensure_indexed(self._read_location_ids(location_id))
        #     else:
        #         raise ValueError("location_id must be int or list of ints")
        # return xr.decode_cf(self._ds)

    def write(self, filename, **kwargs):
        raise NotImplementedError

    def close(self):
        """
        Close file.
        """
        if self._ds is not None:
            self._ds.close()
            self._ds = None

class ASCAT_NetCDF4(RaggedXArrayIOBase):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)
        self.fn_format = "{:04d}.nc"

    def _read_location_id(self, location_id):
        """
        Read data for a single location_id
        """
        if location_id not in self._ds.location_id.values:
            return None

        # if self.ra_type == "contiguous":
        #     idx = np.where(self._ds.location_id.values == location_id)[0][0]
        #     row_start = self._ds.row_size.values[:idx].sum()
        #     row_end = row_start + self._ds.row_size.values[idx]
        #     ds = self._ds.isel(obs=slice(row_start, row_end), locations=idx)

        idx = np.where(self._ds.location_id.values == location_id)[0][0]
        locationIndex = np.where(self._ds.locationIndex.values == idx)[0]
        ds = self._ds.isel(obs=np.where(self._ds.locationIndex.values==idx)[0], locations=idx)

        return ds

    def _read_location_ids(self, location_ids):
        """
        Read data for a list of location_ids
        """
        if location_id not in self._ds.location_id.values:
            return None

        # if self.ra_type == "contiguous":
        #     idxs = np.where(self._ds.location_id.values in location_ids)[0]
        #     row_starts = self._ds.row_size.values[:idxs].sum()
        #     row_ends = row_start + self._ds.row_size.values[idx]
        #     ds = self._ds.isel(obs=slice(row_start, row_end), locations=idx)

        idxs = np.where(self._ds.location_id.values == location_ids)[0]
        locationIndex = np.where(self._ds.locationIndex.values in idxs)[0]
        ds = self._ds.isel(obs=np.where(self._ds.locationIndex.values==idxs)[0], locations=idxs)

        return ds

    def read(self, location_id=None, **kwargs):
        if location_id is not None:
            if isinstance(location_id, int):
                return self._ensure_indexed(self._read_location_id(location_id))
            elif isinstance(location_id, list):
                return self._ensure_indexed(self._read_location_ids(location_id))
            else:
                raise ValueError("location_id must be int or list of ints")
        return xr.decode_cf(self._ds)

    def write(self, filename, **kwargs):
        self._ds.to_netcdf(filename, **kwargs)

    # def read_indexed(self, location_id=None, **kwargs):
    #     if location_id is not None:
    #         print(f"location_id: {location_id}")
    #         print(self._ds.location_id)
    #     return self._ds

    # def read_contiguous(self, location_id=None, **kwargs):
    #     if location_id is not None:
    #         print(f"location_id: {location_id}")
    #         print(self._ds.location_id)
    #     return self._ds

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
                                         preprocess=self._preprocess,
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

class ASCAT_Swath(SwathIOBase):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "netcdf4", **kwargs)

    def _preprocess(self, ds):
        """
        Preprocessing function for opening multifile datasets
        """
        return ds

    def _read_location_id(self, location_id):
        """
        Read data for a single location_id
        """
        if location_id not in self._ds.location_id.values:
            return None

        # if self.ra_type == "contiguous":
        #     idx = np.where(self._ds.location_id.values == location_id)[0][0]
        #     row_start = self._ds.row_size.values[:idx].sum()
        #     row_end = row_start + self._ds.row_size.values[idx]
        #     ds = self._ds.isel(obs=slice(row_start, row_end), locations=idx)

        idx = np.where(self._ds.location_id.values == location_id)[0]
        # locationIndex = np.where(self._ds.locationIndex.values == idx)[0]
        ds = self._ds.isel(obs=idx)
        # ds = self._ds.sel(location_id=location_id)

        return ds

    def _read_location_ids(self, location_ids):
        """
        Read data for a list of location_ids
        """
        idxs = np.isin(self._ds.location_id.values, location_ids)
        if not np.any(idxs):
            return None
        ds = self._ds.isel(obs=idxs)

        return ds

    def read(self, location_id=None, **kwargs):
        if location_id is not None:
            if isinstance(location_id, int):
                return self._read_location_id(location_id)
            elif isinstance(location_id, list):
                return self._read_location_ids(location_id)
            else:
                raise ValueError("location_id must be int or list of ints")
        return xr.decode_cf(self._ds)

    def write(self, filename, **kwargs):
        self._ds.to_netcdf(filename, **kwargs)
