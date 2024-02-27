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

import datetime

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from flox import groupby_reduce

from dask.array import vstack

import ascat.read_native.ragged_array_ts as rat
from ascat.read_native.xarray_io import get_swath_product_id
from ascat.regrid import fib_to_standard_ds, fib_to_standard

progress_to_stdout = False


class TemporalSwathAggregator:
    """Class to aggregate ASCAT data its location ids over time."""

    def __init__(
        self,
        filepath,
        start_dt,
        end_dt,
        t_delta,
        agg,
        snow_cover_mask=None,
        frozen_soil_mask=None,
        subsurface_scattering_mask=None,
        regrid_degrees=None,
    ):
        """Initialize the class.

        Parameters
        ----------
        filepath : str
            Path to the data.
        start_dt : str
            Start date and time (formatted e.g. 2020-01-01T00:00:00).
        end_dt : str
            End date and time (formatted e.g. 2020-02-01T00:00:00).
        t_delta : str
            Time period for aggregation (e.g. 1D, 1W, 1M, 1Y, 2D, 3M, 4Y, etc.).
        agg : str
            Aggregation.
        snow_cover_mask : int, optional
            Snow cover probability value above which to mask the source data.
        frozen_soil_mask : int, optional
            Frozen soil probability value above which to mask the source data.
        subsurface_scattering_mask : int, optional
            Subsurface scattering probability value above which to mask the source data.
        """
        self.filepath = filepath
        self.start_dt = datetime.datetime.strptime(start_dt, "%Y-%m-%dT%H:%M:%S")
        self.end_dt = datetime.datetime.strptime(end_dt, "%Y-%m-%dT%H:%M:%S")
        self.timedelta = pd.Timedelta(t_delta)
        if agg in [
            "mean",
            "median",
            "mode",
            "std",
            "min",
            "max",
            "argmin",
            "argmax",
            "quantile",
            "first",
            "last",
        ]:
            agg = "nan" + agg
        self.agg = agg
        self.regrid_degrees = regrid_degrees

        # assumes ONLY swath files are in the folder
        first_fname = str(next(Path(filepath).rglob("*.nc")).name)
        product = get_swath_product_id(first_fname)
        self.product = product

        self.collection = rat.SwathFileCollection.from_product_id(
            Path(filepath), product
        )

        self.grid = self.collection.grid
        self.data = None
        self.agg_vars = [
            "surface_soil_moisture",
            "backscatter40",
        ]
        self.mask_probs = {
            "snow_cover_probability": (
                80 if snow_cover_mask is None else snow_cover_mask
            ),
            "frozen_soil_probability": (
                80 if frozen_soil_mask is None else frozen_soil_mask
            ),
            "subsurface_scattering_probability": (
                90 if subsurface_scattering_mask is None else subsurface_scattering_mask
            ),
        }

    def _read_data(self):
        if progress_to_stdout:
            print("reading data, this may take some time...")
        self.data = self.collection.read(
            date_range=(self.start_dt, self.end_dt),
        )
        if progress_to_stdout:
            print("done reading data")

    def _set_metadata(self, ds):
        """Add appropriate metadata to datasets."""
        return ds

    def write_time_chunks(self, out_dir):
        """Loop through time chunks and write them to file."""
        product_id = self.product.lower().replace("_", "-")
        if self.regrid_degrees is None:
            grid_sampling = self.collection.ioclass.grid_sampling_km + "km"
        else:
            grid_sampling = str(self.regrid_degrees) + "deg"

        if self.agg is not None:
            yield_func = self.yield_aggregated_time_chunks
            agg_str = f"_{self.agg}"
        else:
            yield_func = self.yield_time_chunks
            agg_str = "_data"

        for ds in yield_func():
            chunk_start_str = (
                np.datetime64(ds.attrs["start_time"])
                .astype(datetime.datetime)
                .strftime("%Y%m%d%H%M%S")
            )
            chunk_end_str = (
                np.datetime64(ds.attrs["end_time"])
                .astype(datetime.datetime)
                .strftime("%Y%m%d%H%M%S")
            )
            ds = self._set_metadata(ds)
            out_name = (
                f"ascat"
                f"_{product_id}"
                f"_{grid_sampling}"
                f"{agg_str}"
                f"_{chunk_start_str}"
                f"_{chunk_end_str}.nc"
            )

            ds.to_netcdf(
                Path(out_dir) / out_name,
            )
        print("complete                     ")

    def yield_time_chunks(self):
        """Loop through time chunks of the range, yield the merged data unmodified."""
        time_chunks = pd.date_range(
            start=self.start_dt, end=self.end_dt, freq=self.timedelta
        )
        if self.data is not None:
            # I don't know why this case would exist, but if it does...
            ds = self.data
            for timechunk in time_chunks:
                chunk_start = timechunk
                chunk_end = timechunk + self.timedelta - pd.Timedelta("1s")
                ds_chunk = ds.sel(time=slice(chunk_start, chunk_end))
                ds_chunk.attrs["start_time"] = np.datetime64(chunk_start).astype(str)
                ds_chunk.attrs["end_time"] = np.datetime64(chunk_end).astype(str)
                yield ds_chunk
        if self.data is None:
            for timechunk in time_chunks:
                chunk_start = timechunk
                chunk_end = timechunk + self.timedelta
                ds_chunk = self.collection.read(date_range=(chunk_start, chunk_end))
                chunk_end = chunk_end - pd.Timedelta("1s")
                ds_chunk.attrs["start_time"] = np.datetime64(chunk_start).astype(str)
                ds_chunk.attrs["end_time"] = np.datetime64(chunk_end).astype(str)
                yield ds_chunk

    def yield_aggregated_time_chunks(self):
        """Loop through data in time chunks, aggregating it over time."""
        if self.data is None:
            self._read_data()
        ds = self.data
        if progress_to_stdout:
            print("masking data...", end="\r")
        # mask ds where "surface_flag" is 1, "snow_cover_probability" is > 90, or "frozen_soil_probability" is > 90
        mask = (
            (ds.surface_flag != 0)
            | (ds.snow_cover_probability > self.mask_probs["snow_cover_probability"])
            | (ds.frozen_soil_probability > self.mask_probs["frozen_soil_probability"])
            | (
                ds.subsurface_scattering_probability
                > self.mask_probs["subsurface_scattering_probability"]
            )
        )
        ds = ds.where(~mask, drop=False)
        ds["time_chunks"] = (
            ds.time - np.datetime64(self.start_dt, "ns")
        ) // self.timedelta

        present_agg_vars = [var for var in self.agg_vars if var in ds.variables]

        agg_vars_stack = vstack([ds[var].data for var in present_agg_vars])

        if progress_to_stdout:
            print("constructing groups...", end="\r")
        grouped_data, time_groups, loc_groups = groupby_reduce(
            agg_vars_stack, ds["time_chunks"], ds["location_id"], func=self.agg
        )
        # shape of grouped_data is (n_agg_vars, n_time_chunks, n_locations)
        # now we need to rebuild an xarray dataset from this
        # we can use the time_groups and loc_groups to do this
        grouped_ds = xr.Dataset(
            {
                var: (("time_chunks", "location_id"), grouped_data[i])
                for i, var in enumerate(present_agg_vars)
            },
            coords={
                "time_chunks": time_groups,
                "location_id": loc_groups.astype(int),
            },
        )

        if self.regrid_degrees is not None:
            print("regridding             ")
            grouped_ds = fib_to_standard_ds(grouped_ds, self.grid, self.regrid_degrees)

        else:
            lons, lats = self.grid.gpi2lonlat(grouped_ds.location_id.values)
            grouped_ds["lon"] = ("location_id", lons)
            grouped_ds["lat"] = ("location_id", lats)
            grouped_ds = grouped_ds.set_coords(["lon", "lat"])

        for timechunk, group in grouped_ds.groupby("time_chunks"):
            if progress_to_stdout:
                print(
                    f"processing time chunk {timechunk + 1}/{len(time_groups)}...      ",
                    end="\r",
                )
            chunk_start = self.start_dt + self.timedelta * timechunk
            chunk_end = (
                self.start_dt + self.timedelta * (timechunk + 1) - pd.Timedelta("1s")
            )
            group.attrs["start_time"] = np.datetime64(chunk_start).astype(str)
            group.attrs["end_time"] = np.datetime64(chunk_end).astype(str)
            group["time_chunks"] = np.datetime64(chunk_start, "ns")
            group = group.rename({"time_chunks": "time"})
            yield group

        # # alternative implementation: loop through time chunks and THEN build the dataset
        # # for each
        # lons, lats = self.grid.gpi2lonlat(loc_groups)
        # for timegroup in time_groups:
        #     print(f"processing time chunk {timegroup + 1}/{len(time_groups)}...      ", end="\r")
        #     group_ds = xr.Dataset(
        #         {
        #             var: (("location_id",), grouped_data[i, timegroup])
        #             for i, var in enumerate(present_agg_vars)
        #         },
        #         coords={
        #             "location_id": loc_groups,
        #             "lon": ("location_id", lons),
        #             "lat": ("location_id", lats),
        #         },
        #     )
        #     chunk_start = self.start_dt + self.timedelta * timegroup
        #     chunk_end = (
        #         self.start_dt + self.timedelta * (timegroup + 1) - pd.Timedelta("1s")
        #     )
        #     group_ds.attrs["start_time"] = np.datetime64(chunk_start).astype(str)
        #     group_ds.attrs["end_time"] = np.datetime64(chunk_end).astype(str)
        #     group_ds["time"] = np.datetime64(chunk_start, "ns")
        #     yield group_ds
