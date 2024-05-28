#!/usr/bin/env python3

import re

import numpy as np

from fibgrid.realization import FibGrid
from ascat.read_native.xarray_io import grid_cache
# from ascat.read_native.swath_collection import SwathGridFiles, SwathFile

class CellGridCache:
    """
    Cache for CellGrid objects.
    """

    def __init__(self):
        self.grids = {}

    def fetch_or_store(self, key, grid=None, attrs=None):
        """
        Fetch a CellGrid object from the cache given a key,
        or store a new one.

        Parameters
        ----------
        """
        if key not in self.grids:
            if grid is None:
                raise ValueError(
                    "Key not in cache, please specify cell_grid_type and arguments"
                    " to create a new CellGrid object and add it to the cache under"
                    " the given key.")
            self.grids[key] = {}
            self.grids[key]["grid"] = grid
            self.grids[key]["possible_cells"] = self.grids[key][
                "grid"].get_cells()
            self.grids[key]["max_cell"] = self.grids[key][
                "possible_cells"].max()
            self.grids[key]["min_cell"] = self.grids[key][
                "possible_cells"].min()
            self.grids[key]["attrs"] = attrs

        return self.grids[key]


grid_cache = CellGridCache()
grid_cache.fetch_or_store("Fib6.25", FibGrid(6.25), {"grid_sampling_km": 6.25})
grid_cache.fetch_or_store("Fib12.5", FibGrid(12.5), {"grid_sampling_km": 12.5})


# Define dataset-specific classes.
class AscatH129Cell():
    grid_name = "Fib6.25"
    grid_info = grid_cache.fetch_or_store(grid_name)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]


class AscatH129v1Cell():
    grid_name = "Fib6.25"
    grid_info = grid_cache.fetch_or_store(grid_name, FibGrid, 6.25)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]


class AscatH121v1Cell():
    grid_name = "Fib12.5"
    grid_info = grid_cache.fetch_or_store(grid_name, FibGrid, 12.5)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]


class AscatH122Cell():
    grid_name = "Fib6.25"
    grid_info = grid_cache.fetch_or_store(grid_name, FibGrid, 6.25)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]


class AscatSIG0Cell6250m():
    grid_name = "Fib6.25"
    grid_info = grid_cache.fetch_or_store(grid_name, FibGrid, 6.25)
    grid = grid_info["grid"]
    # grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]


class AscatSIG0Cell12500m():
    grid_name = "Fib12.5"
    grid_info = grid_cache.fetch_or_store(grid_name, FibGrid, 12.5)
    grid = grid_info["grid"]
    grid_cell_size = 5
    fn_format = "{:04d}.nc"
    possible_cells = grid_info["possible_cells"]
    max_cell = grid_info["max_cell"]
    min_cell = grid_info["min_cell"]


class AscatH129Swath():
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H129_C_LIIB_{date}_{placeholder}_{placeholder1}____.nc"
    sf_pattern = {
        # "satellite_folder": "metop_[abc]",
        "year_folder": "{year}"
    }
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
    # grid_sampling_km = 6.25
    # grid = grid_cache.fetch_or_store(grid_name, FibGrid,
    #                                  grid_sampling_km)["grid"]
    # grid_cell_size = 5
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

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            # "satellite_folder": {
            #     "satellite": "metop_[abc]"
            # },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    # def __init__(
    #         self,
    #         root_path,
    #         fn_templ=fn_pattern,
    #         sf_templ=sf_pattern,
    #         grid=grid,
    #         cls_kwargs=None,
    #         err=True,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=None,
    #         sf_write_fmt=None,
    #         cache_size=0
    # ):
    #     super().__init__(
    #         root_path,
    #         SwathFile,
    #         fn_templ,
    #         sf_templ,
    #         grid=grid,
    #         cls_kwargs=cls_kwargs,
    #         err=err,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=fn_write_fmt,
    #         sf_write_fmt=sf_write_fmt,
    #         cache_size=cache_size
    #     )

class AscatH129v1Swath():
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H129_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
    # grid_sampling_km = 6.25
    # grid = grid_cache.fetch_or_store(grid_name, FibGrid,
    #                                  grid_sampling_km)["grid"]
    # grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = []
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("backscatter40", np.float32),
        ("slope40", np.float32),
        ("curvature40", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("backscatter_flag", np.uint8),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("surface_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
        ("subsurface_scattering_probability", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    # def __init__(
    #         self,
    #         root_path,
    #         fn_templ=fn_pattern,
    #         sf_templ=sf_pattern,
    #         grid=grid,
    #         cls_kwargs=None,
    #         err=True,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=None,
    #         sf_write_fmt=None,
    #         cache_size=0
    # ):
    #     super().__init__(
    #         root_path,
    #         SwathFile,
    #         fn_templ,
    #         sf_templ,
    #         grid=grid,
    #         cls_kwargs=cls_kwargs,
    #         err=err,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=fn_write_fmt,
    #         sf_write_fmt=sf_write_fmt,
    #         cache_size=cache_size
    #     )


class AscatH121v1Swath():
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-12.5km-H121_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib12.5"
    # grid_sampling_km = 12.5
    # grid = grid_cache.fetch_or_store(grid_name, FibGrid,
    #                                  grid_sampling_km)["grid"]
    # grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = []
    ts_dtype = np.dtype([
        ("sat_id", np.int8),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("backscatter40", np.float32),
        ("slope40", np.float32),
        ("curvature40", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("backscatter_flag", np.uint8),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("surface_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
        ("subsurface_scattering_probability", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    # def __init__(
    #         self,
    #         root_path,
    #         fn_templ=fn_pattern,
    #         sf_templ=sf_pattern,
    #         grid=grid,
    #         cls_kwargs=None,
    #         err=True,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=None,
    #         sf_write_fmt=None,
    #         cache_size=0
    # ):
    #     super().__init__(
    #         root_path,
    #         SwathFile,
    #         fn_templ,
    #         sf_templ,
    #         grid=grid,
    #         cls_kwargs=cls_kwargs,
    #         err=err,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=fn_write_fmt,
    #         sf_write_fmt=sf_write_fmt,
    #         cache_size=cache_size
    #     )

class AscatH122Swath():
    fn_pattern = "ascat_ssm_nrt_6.25km_{placeholder}Z_{date}Z_metop-{sat}_h122.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
    # grid_sampling_km = 6.25
    # grid = grid_cache.fetch_or_store(grid_name, FibGrid,
    #                                  grid_sampling_km)["grid"]
    # grid_cell_size = 5
    cell_fn_format = "{:04d}.nc"
    beams_vars = []
    ts_dtype = np.dtype([
        ("sat_id", np.int64),
        ("as_des_pass", np.int8),
        ("swath_indicator", np.int8),
        ("surface_soil_moisture", np.float32),
        ("surface_soil_moisture_noise", np.float32),
        ("sigma40", np.float32),
        ("sigma40_noise", np.float32),
        ("slope40", np.float32),
        ("slope40_noise", np.float32),
        ("curvature40", np.float32),
        ("curvature40_noise", np.float32),
        ("dry40", np.float32),
        ("dry40_noise", np.float32),
        ("wet40", np.float32),
        ("wet40_noise", np.float32),
        ("surface_soil_moisture_sensitivity", np.float32),
        ("surface_soil_moisture_climatology", np.float32),
        ("correction_flag", np.uint8),
        ("processing_flag", np.uint8),
        ("snow_cover_probability", np.int8),
        ("frozen_soil_probability", np.int8),
        ("wetland_fraction", np.int8),
        ("topographic_complexity", np.int8),
    ])

    @staticmethod
    def fn_read_fmt(timestamp):
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    # def __init__(
    #         self,
    #         root_path,
    #         fn_templ=fn_pattern,
    #         sf_templ=sf_pattern,
    #         grid=grid,
    #         cls_kwargs=None,
    #         err=True,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=None,
    #         sf_write_fmt=None,
    #         cache_size=0
    # ):
    #     super().__init__(
    #         root_path,
    #         SwathFile,
    #         fn_templ,
    #         sf_templ,
    #         grid=grid,
    #         cls_kwargs=cls_kwargs,
    #         err=err,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=fn_write_fmt,
    #         sf_write_fmt=sf_write_fmt,
    #         cache_size=cache_size
    #     )


class AscatSIG0Swath6250m():
    """
    Class for reading ASCAT sigma0 swath data and writing it to cells.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP{sat}-6.25_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
    # grid_sampling_km = 6.25
    # grid = grid_cache.fetch_or_store(grid_name, FibGrid,
    #                                  grid_sampling_km)["grid"]
    # grid_cell_size = 5
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
    ]
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
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    # def __init__(
    #         self,
    #         root_path,
    #         fn_templ=fn_pattern,
    #         sf_templ=sf_pattern,
    #         grid=grid,
    #         cls_kwargs=None,
    #         err=True,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=None,
    #         sf_write_fmt=None,
    #         cache_size=0
    # ):
    #     super().__init__(
    #         root_path,
    #         SwathFile,
    #         fn_templ,
    #         sf_templ,
    #         grid=grid,
    #         cls_kwargs=cls_kwargs,
    #         err=err,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=fn_write_fmt,
    #         sf_write_fmt=sf_write_fmt,
    #         cache_size=cache_size
    #     )


class AscatSIG0Swath12500m():
    """
    Class for reading and writing ASCAT sigma0 swath data.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP{sat}-12.5_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib12.5"
    # grid_sampling_km = 12.5
    # grid = grid_cache.fetch_or_store(grid_name, FibGrid,
    #                                  grid_sampling_km)["grid"]
    # grid_cell_size = 5
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
    ]
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
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": "[ABC]",
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp):
        return {
            "satellite_folder": {
                "satellite": "metop_[abc]"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }

    # def __init__(
    #         self,
    #         root_path,
    #         fn_templ=fn_pattern,
    #         sf_templ=sf_pattern,
    #         grid=grid,
    #         cls_kwargs=None,
    #         err=True,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=None,
    #         sf_write_fmt=None,
    #         cache_size=0
    # ):
    #     super().__init__(
    #         root_path,
    #         SwathFile,
    #         fn_templ,
    #         sf_templ,
    #         grid=grid,
    #         cls_kwargs=cls_kwargs,
    #         err=err,
    #         fn_read_fmt=fn_read_fmt,
    #         sf_read_fmt=sf_read_fmt,
    #         fn_write_fmt=fn_write_fmt,
    #         sf_write_fmt=sf_write_fmt,
    #         cache_size=cache_size
    #     )


cell_io_catalog = {
    "H129": AscatH129Cell,
    "H129_V1.0": AscatH129v1Cell,
    "H121_V1.0": AscatH121v1Cell,
    "H122": AscatH122Cell,
    "SIG0_6.25": AscatSIG0Cell6250m,
    "SIG0_12.5": AscatSIG0Cell12500m,
}

swath_io_catalog = {
    "H129": AscatH129Swath,
    "H129_V1.0": AscatH129v1Swath,
    "H121_V1.0": AscatH121v1Swath,
    "H122": AscatH122Swath,
    "SIG0_6.25": AscatSIG0Swath6250m,
    "SIG0_12.5": AscatSIG0Swath12500m,
}

swath_fname_regex_lookup = {
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-6.25-H129_C_LIIB_.*_.*_.*____.nc":
        "H129",
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-6.25km-H129_C_LIIB_.*_.*_.*____.nc":
        "H129_V1.0",
    "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP[ABC]-12.5km-H121_C_LIIB_.*_.*_.*____.nc":
        "H121_V1.0",
    "ascat_ssm_nrt_6.25km_.*Z_.*Z_metop-[ABC]_h122.nc":
        "H122",
    "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP[ABC]-6.25_C_LIIB_.*_.*_.*____.nc":
        "SIG0_6.25",
    "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP[ABC]-12.5_C_LIIB_.*_.*_.*____.nc":
        "SIG0_12.5",
}


def get_swath_product_id(filename):
    for pattern, swath_product_id in swath_fname_regex_lookup.items():
        if re.match(pattern, filename):
            return swath_product_id
    return None
