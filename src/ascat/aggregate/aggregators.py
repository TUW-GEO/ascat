# Copyright (c) 2024, TU Wien
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
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from flox.xarray import xarray_reduce
from dask.array import unique as da_unique

import ascat.read_native.ragged_array_ts as rat
from ascat.read_native.xarray_io import get_swath_product_id
from ascat.read_native.xarray_io import dtype_to_nan


class TemporalSwathAggregator:
    """Class to aggregate ASCAT data its location ids over time."""

    def __init__(
        self,
        filepath,
        start_dt,
        end_dt,
        t_delta,
        agg,
        snow_cover_mask=80,
        frozen_soil_mask=80,
        subsurface_scattering_mask=5,
        ssm_sensitivity_mask=1,
        no_masking=False,
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
            Time period for aggregation (e.g. 1D, 1W, 1M, 1Y, 3M, 4Y, etc.).
        agg : str
            Aggregation method (e.g. mean, median, std).
        snow_cover_mask : int, optional
            Snow cover probability value above which to mask the source data.
        frozen_soil_mask : int, optional
            Frozen soil probability value above which to mask the source data.
        subsurface_scattering_mask : int, optional
            Subsurface scattering probability value above which to mask
            the source data.
        ssm_sensitivity_mask : float, optional
            Soil moisture sensitivity value above which to mask
            the source data.
        no_masking : boolean, optional
            Ignore all masks (default: False).
        """
        self.filepath = filepath

        fmt = "%Y-%m-%dT%H:%M:%S"
        self.start_dt = datetime.datetime.strptime(start_dt, fmt)
        self.end_dt = datetime.datetime.strptime(end_dt, fmt)
        self.timedelta = pd.Timedelta(t_delta)
        self.no_masking = no_masking

        agg_methods = [
            "mean", "median", "mode", "std", "min", "max", "argmin", "argmax",
            "quantile", "first", "last"
        ]

        if agg in agg_methods:
            agg = "nan" + agg

        self.agg = agg

        # assumes ONLY swath files are in the folder
        first_fname = str(next(Path(filepath).rglob("*.nc")).name)
        product = get_swath_product_id(first_fname)
        self.product = product

        self.collection = rat.SwathFileCollection.from_product_id(
            Path(filepath), product)

        self.grid = self.collection.grid
        self.data = None
        self.agg_vars = {
            "surface_soil_moisture": {
                "dtype": np.dtype("int16"),
                "scale_factor": 1e-2,
            },
            "backscatter40": {
                "dtype": np.dtype("int32"),
                "scale_factor": 1e-7,
            },
        }
        self.mask_probs = {
            "snow_cover_probability": snow_cover_mask,
            "frozen_soil_probability": frozen_soil_mask,
            "subsurface_scattering_probability": subsurface_scattering_mask,
            "surface_soil_moisture_sensitivity": ssm_sensitivity_mask,
        }

    def _read_data(self):
        """Read data."""
        print("constructing dataset, this may take some time...")

        self.data = self.collection.read(
            date_range=(self.start_dt, self.end_dt),)

        print("done constructing dataset")

    def _set_metadata(self, ds):
        """Add appropriate metadata to datasets."""
        return ds

    def _create_output_encoding(self):
        """Create NetCDF encoding."""

        output_encoding = {
            "latitude": {
                "dtype": np.dtype("int32"),
                "scale_factor": 1e-6,
                "zlib": True,
                "complevel": 4,
                "_FillValue": dtype_to_nan[np.dtype("int32")],
                "missing_value": dtype_to_nan[np.dtype("int32")],
            },
            "longitude": {
                "dtype": np.dtype("int32"),
                "scale_factor": 1e-6,
                "zlib": True,
                "complevel": 4,
                "_FillValue": dtype_to_nan[np.dtype("int32")],
                "missing_value": dtype_to_nan[np.dtype("int32")],
            },
            "time": {
                "dtype": np.dtype("float64"),
                "zlib": True,
                "complevel": 4,
                "_FillValue": 0,
                "missing_value": 0,
            },
        }

        for var in self.agg_vars:
            if var in output_encoding:
                continue

            output_encoding[var] = {
                "dtype": self.agg_vars[var]["dtype"],
                "scale_factor": self.agg_vars[var]["scale_factor"],
                "zlib": True,
                "complevel": 4,
                "_FillValue": dtype_to_nan[self.agg_vars[var]["dtype"]],
                "missing_value": dtype_to_nan[self.agg_vars[var]["dtype"]],
            }

        return output_encoding

    def write_time_steps(self, outpath):
        """
        Loop through time steps and write them to file.

        Parameters
        ----------
        outpath : str
            Output path.
        """
        product_id = self.product.lower().replace("_", "-")
        grid_sampling = str(self.collection.ioclass.grid_sampling_km) + "km"

        if self.agg is not None:
            datasets = self.get_aggregated_time_steps()
            agg_str = f"_{self.agg}"
        else:
            datasets = self.get_time_steps()
            agg_str = "_data"

        fmt = "%Y%m%d%H%M%S"

        paths = []
        for ds in datasets:
            step_start_str = (
                np.datetime64(ds.attrs["start_time"]).astype(
                    datetime.datetime).strftime(fmt))
            step_end_str = (
                np.datetime64(ds.attrs["end_time"]).astype(
                    datetime.datetime).strftime(fmt))

            out_name = (f"ascat"
                        f"_{product_id}"
                        f"_{grid_sampling}"
                        f"{agg_str}"
                        f"_{step_start_str}"
                        f"_{step_end_str}.nc")

            paths.append(Path(outpath) / out_name)

        print("saving datasets...", end="\r")

        output_encoding = self._create_output_encoding()
        xr.save_mfdataset(datasets, paths, encoding=output_encoding)
        print("complete                     ")

        return paths

    def get_time_steps(self):
        """
        Loop through time steps of the range, return the merged data
        for each unmodified.
        """
        time_steps = pd.date_range(
            start=self.start_dt, end=self.end_dt, freq=self.timedelta)
        datasets = []

        if self.data is not None:
            # I don't know why this case would exist, but if it does...
            ds = self.data
            for timestep in time_steps:
                step_start = timestep
                step_end = timestep + self.timedelta - pd.Timedelta("1s")
                ds_step = ds.sel(time=slice(step_start, step_end))
                ds_step.attrs["start_time"] = np.datetime64(step_start).astype(
                    str)
                ds_step.attrs["end_time"] = np.datetime64(step_end).astype(str)
                datasets.append(ds_step)

        if self.data is None:
            for timestep in time_steps:
                step_start = timestep
                step_end = timestep + self.timedelta
                ds_step = self.collection.read(
                    date_range=(step_start, step_end))
                step_end = step_end - pd.Timedelta("1s")
                ds_step.attrs["start_time"] = np.datetime64(step_start).astype(
                    str)
                ds_step.attrs["end_time"] = np.datetime64(step_end).astype(str)
                datasets.append(ds_step)

        return datasets

    def get_aggregated_time_steps(self):
        """Loop through data in time steps, aggregating it over time."""
        if self.data is None:
            self._read_data()

        ds = self.data

        present_agg_vars = [
            var for var in self.agg_vars if var in ds.variables
        ]

        print("masking data...", end="\r")

        global_mask = (ds.surface_flag != 0)

        ds = ds.where(~global_mask, drop=False)

        if not self.no_masking:
            variable_masks = {
                "surface_soil_moisture": (
                    (ds["frozen_soil_probability"]
                    > self.mask_probs["frozen_soil_probability"])
                    | (ds["snow_cover_probability"]
                    > self.mask_probs["snow_cover_probability"])
                    | (ds["subsurface_scattering_probability"]
                    > self.mask_probs["subsurface_scattering_probability"])
                    | (ds["surface_soil_moisture_sensitivity"]
                    < self.mask_probs["surface_soil_moisture_sensitivity"])),
            }

            for var, var_mask in variable_masks.items():
                ds[var] = ds[var].where(~var_mask, drop=False)

        print("grouping data...           ")

        # discretize time into integer-labeled steps according to our desired frequency
        ds["time_steps"] = (
            ds.time - np.datetime64(self.start_dt, "ns")) // self.timedelta

        # get unique time_step and location_id values so we can tell xarray_reduce
        # what to expect.
        expected_time_steps = da_unique(ds["time_steps"].data).compute()
        expected_location_ids = da_unique(ds["location_id"].data).compute()

        # remove NaN from the expected location ids (this was introduced by the masking)
        expected_location_ids = expected_location_ids[
            ~np.isnan(expected_location_ids)]

        # group the data by time_steps and location_id and aggregate it
        grouped_ds = xarray_reduce(
            ds[present_agg_vars],
            ds["time_steps"],
            ds["location_id"],
            expected_groups=(expected_time_steps, expected_location_ids),
            func=self.agg)

        # convert the location_id back to an integer
        grouped_ds["location_id"] = grouped_ds["location_id"].astype(int)

        lons, lats = self.grid.gpi2lonlat(grouped_ds.location_id.values)
        grouped_ds["longitude"] = ("location_id", lons)
        grouped_ds["latitude"] = ("location_id", lats)
        grouped_ds = grouped_ds.set_coords(["longitude", "latitude"])
        grouped_ds = grouped_ds.chunk({"time_steps": 1})

        # Compute the results, write to a temporary directory, and then read it back in.
        # If we don't do this, dask will try to compute the entire dataset
        # for each step of time before writing it to disk later.

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "ascat.nc"
            print("computing results...             ")
            grouped_ds.to_netcdf(temp_path)
            grouped_ds.close()
            grouped_ds = xr.open_dataset(temp_path)

        groups = []
        for timestep, group in grouped_ds.groupby("time_steps", squeeze=False):
            group = group.squeeze("time_steps")
            print(
                f"writing time step {timestep + 1}/{len(grouped_ds['time_steps'])}...      ",
                end="\r",
            )
            step_start = self.start_dt + self.timedelta * timestep
            step_end = (
                self.start_dt + self.timedelta * (timestep + 1) -
                pd.Timedelta("1s"))
            group.attrs["start_time"] = np.datetime64(step_start).astype(str)
            group.attrs["end_time"] = np.datetime64(step_end).astype(str)
            group["time_steps"] = np.datetime64(step_start, "ns")
            group = group.rename({"time_steps": "time"})
            group = self._set_metadata(group)
            groups.append(group)

        return groups
