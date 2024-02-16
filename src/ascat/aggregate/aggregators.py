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

from flox.xarray import xarray_reduce

import ascat.read_native.ragged_array_ts as rat

progress_to_stdout = False

class TemporalSwathAggregator:
    """ Class to aggregate ASCAT data its location ids over time."""

    def __init__(self, filepath, outpath, start_dt, end_dt, t_delta, agg, product):
        """ Initialize the class.

        Parameters
        ----------
        filepath : str
            Path to the data.
        outpath : str
            Path to the output data.
        start_dt : str
            Start date and time (formatted e.g. 2020-01-01T00:00:00).
        end_dt : str
            End date and time (formatted e.g. 2020-02-01T00:00:00).
        t_delta : str
            Time period for aggregation (e.g. 1D, 1W, 1M, 1Y, 2D, 3M, 4Y, etc.).
        agg : str
            Aggregation.
        product : str
            Product id.
        """
        self.filepath = filepath
        self.start_dt = datetime.datetime.strptime(start_dt, "%Y-%m-%dT%H:%M:%S")
        self.end_dt = datetime.datetime.strptime(end_dt, "%Y-%m-%dT%H:%M:%S")
        self.timedelta = pd.Timedelta(t_delta)
        self.agg = agg
        self.product = product
        self.collection = rat.SwathFileCollection.from_product_id(
            Path(filepath), product
        )
        self.grid = self.collection.grid
        self.data = None
        self.agg_vars = [
            "surface_soil_moisture",
            "surface_soil_moisture_noise",
            "backscatter40",
            "slope40",
            "curvature40",
            "surface_soil_moisture_sensitivity",
        ]

    def _read_data(self):
        if progress_to_stdout:
            print("reading data, this may take some time...")
        self.data = self.collection.read(date_range=(self.start_dt, self.end_dt))
        if progress_to_stdout:
            print("done reading data")

    def _set_metadata(self, ds):
        """Add appropriate metadata to datasets."""
        return ds

    def write_time_chunks(self, out_dir):
        """Loop through time chunks and write them to file."""
        for ds in self.yield_aggregated_time_chunks():
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
            ds["location_id"] = ds["location_id"].astype(int)
            ds = self._set_metadata(ds)
            ds.to_netcdf(
                f"{out_dir}/ascat_{self.agg}_{chunk_start_str}_{chunk_end_str}.nc",
                # encoding={"location_id": {"dtype": int}}
            )

    def yield_aggregated_time_chunks(self):
        """Loop through data in time chunks, aggregating it over time."""
        if self.data is None:
            self._read_data()
        ds = self.data
        ds["time_chunks"] = (
            ds.time - np.datetime64(self.start_dt, "ns")
        ) // self.timedelta

        present_agg_vars = [var for var in self.agg_vars if var in ds.variables]
        total_time_chunks = int(
            (self.end_dt - self.start_dt) / self.timedelta
        ) + 1
        for timechunk, group in ds.groupby("time_chunks"):
            if progress_to_stdout:
                print(f"processing time chunk {timechunk + 1}/{total_time_chunks}...      ", end="\r")
            chunk_means = xarray_reduce(
                group[present_agg_vars + ["location_id"]],
                group.location_id,
                expected_groups=(np.unique(group.location_id.values),),
                # dims = ["obs"],
                func=self.agg,
            )

            lons, lats = self.grid.gpi2lonlat(chunk_means.location_id.values)
            chunk_means["lon"] = ("location_id", lons)
            chunk_means["lat"] = ("location_id", lats)
            chunk_means = chunk_means.set_coords(["lon", "lat"])

            chunk_start = self.start_dt + self.timedelta * timechunk
            chunk_end = (
                self.start_dt + self.timedelta * (timechunk + 1) - pd.Timedelta("1s")
            )
            chunk_means.attrs["start_time"] = np.datetime64(chunk_start).astype(str)
            chunk_means.attrs["end_time"] = np.datetime64(chunk_end).astype(str)
            yield chunk_means
