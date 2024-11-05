#!/usr/bin/env python3

import re

import numpy as np

from fibgrid.realization import FibGrid

from pygeogrids.grids import BasicGrid
from pygeogrids.netcdf import load_grid

from pathlib import Path

from ascat.read_native.cell_collection import RaggedArrayCell
from ascat.read_native.cell_collection import OrthoMultiCell

from ascat.read_native.grid_registry import grid_registry


def register_cell_grid_product(product_class, grid, product_id, grid_attrs=None):
    """
    Register a product class for a specific grid.

    Parameters
    ----------
    product_class: class
        Class to register
    grid: pygeogrids.BasicGrid
        Grid object to register the class for
    """
    grid_attrs = grid_attrs or {}
    if isinstance(grid, (str, Path)):
        grid = load_grid(grid)
    elif not isinstance(grid, BasicGrid):
        raise ValueError("grid must be a BasicGrid object or a string or Path to a grid file.")

    grid_registry.register(product_class.grid_name, lambda: {"grid": grid, "attrs": {**grid_attrs}})

    cell_io_catalog[product_id] = product_class

def register_swath_grid_product(product_class, grid, product_id, grid_attrs=None):
    """
    Register a product class for a specific grid.

    Parameters
    ----------
    product_class: class
        Class to register
    grid: pygeogrids.BasicGrid
        Grid object to register the class for
    """
    grid_attrs = grid_attrs or {}
    if isinstance(grid, (str, Path)):
        grid = load_grid(grid)
    elif not isinstance(grid, BasicGrid):
        raise ValueError("grid must be a BasicGrid object or a string or Path to a grid file.")

    grid_registry.register(product_class.grid_name, lambda: {"grid": grid, "attrs": {**grid_attrs}})

    swath_io_catalog[product_id] = product_class


class RaggedArrayCellProduct:
    file_class = RaggedArrayCell
    fn_format = "{:04d}.nc"

    
class ErsHCell(RaggedArrayCellProduct):
    grid_name = "Fib12.5"
    #sf_pattern = {"satellite_folder": "ERS-{sat}_WindScatt", "res_folder": "{res}"}


class ErsNCell(RaggedArrayCellProduct):
    grid_name = "Fib25"


class AscatH129Cell(RaggedArrayCellProduct):
    grid_name = "Fib6.25"
    # sf_pattern = {"sat_str": "{sat}"}


class AscatH129v1Cell(RaggedArrayCellProduct):
    grid_name = "Fib6.25"


class AscatH121v1Cell(RaggedArrayCellProduct):
    grid_name = "Fib12.5"


class AscatH122Cell(RaggedArrayCellProduct):
    grid_name = "Fib6.25"


class AscatSIG0Cell6250m(RaggedArrayCellProduct):
    grid_name = "Fib6.25"


class AscatSIG0Cell12500m(RaggedArrayCellProduct):
    grid_name = "Fib12.5"


class SwathProduct:
    from ascat.read_native.swath_collection import Swath
    file_class = Swath

class AscatSwathProduct(SwathProduct):
    @staticmethod
    def preprocess_(ds):
        ds.attrs["global_attributes_flag"] = 1
        if "spacecraft" in ds.attrs:
            # Assumption: the spacecraft attribute is something like "metop-a"
            sat_id = {"a": 3, "b": 4, "c": 5}
            sat = ds.attrs["spacecraft"][-1].lower()
            ds["sat_id"] = ("obs",
                            np.repeat(sat_id[sat], ds["location_id"].size))
            del ds.attrs["spacecraft"]

class AscatH129Swath(AscatSwathProduct):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25-H129_C_LIIB_{date}_{placeholder}_{placeholder1}____.nc"
    sf_pattern = {
        "satellite_folder": "metop_[abc]",
        "year_folder": "{year}"
    }
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
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
    def fn_read_fmt(timestamp, sat="[ABC]"):
        sat = sat.upper()
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": sat,
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp, sat="[abc]"):
        sat = sat.lower()
        return {
            "satellite_folder": {
                "satellite": f"metop_{sat}"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }


class AscatH129v1Swath(AscatSwathProduct):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-6.25km-H129_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
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
    def fn_read_fmt(timestamp, sat="[ABC]"):
        sat = sat.upper()
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": sat,
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp, sat="[abc]"):
        sat = sat.lower()
        return {
            "satellite_folder": {
                "satellite": f"metop_{sat}"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }


class AscatH121v1Swath(AscatSwathProduct):
    fn_pattern = "W_IT-HSAF-ROME,SAT,SSM-ASCAT-METOP{sat}-12.5km-H121_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib12.5"
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
    def fn_read_fmt(timestamp, sat="[ABC]"):
        sat = sat.upper()
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": sat,
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp, sat="[abc]"):
        sat = sat.lower()
        return {
            "satellite_folder": {
                "satellite": f"metop_{sat}"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }


class AscatH122Swath(AscatSwathProduct):
    fn_pattern = "ascat_ssm_nrt_6.25km_{placeholder}Z_{date}Z_metop-{sat}_h122.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
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
    def fn_read_fmt(timestamp, sat="[ABC]"):
        sat = sat.upper()
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": sat,
            "placeholder": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp, sat="[abc]"):
        sat = sat.lower()
        return {
            "satellite_folder": {
                "satellite": f"metop_{sat}"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }


class AscatSIG0Swath6250m(AscatSwathProduct):
    """
    Class for reading ASCAT sigma0 swath data and writing it to cells.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP{sat}-6.25_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib6.25"
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
    def fn_read_fmt(timestamp, sat="[ABC]"):
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
        sat = sat.upper()
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": sat,
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp, sat="[abc]"):
        sat = sat.lower()
        return {
            "satellite_folder": {
                "satellite": f"metop_{sat}"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }


class AscatSIG0Swath12500m(AscatSwathProduct):
    """
    Class for reading and writing ASCAT sigma0 swath data.
    """
    fn_pattern = "W_IT-HSAF-ROME,SAT,SIG0-ASCAT-METOP{sat}-12.5_C_LIIB_{placeholder}_{placeholder1}_{date}____.nc"
    sf_pattern = {"satellite_folder": "metop_[abc]", "year_folder": "{year}"}
    date_field_fmt = "%Y%m%d%H%M%S"
    grid_name = "Fib12.5"
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
    def fn_read_fmt(timestamp, sat="[ABC]"):
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
        sat = sat.upper()
        return {
            "date": timestamp.strftime("%Y%m%d*"),
            "sat": sat,
            "placeholder": "*",
            "placeholder1": "*"
        }

    @staticmethod
    def sf_read_fmt(timestamp, sat="[abc]"):
        sat = sat.lower()
        return {
            "satellite_folder": {
                "satellite": f"metop_{sat}"
            },
            "year_folder": {
                "year": f"{timestamp.year}"
            },
        }


cell_io_catalog = {
    "H129": AscatH129Cell,
    "H129_V1.0": AscatH129v1Cell,
    "H121_V1.0": AscatH121v1Cell,
    "H122": AscatH122Cell,
    "SIG0_6.25": AscatSIG0Cell6250m,
    "SIG0_12.5": AscatSIG0Cell12500m,
    "ERSH": ErsHCell,
    "ERSN": ErsNCell,
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
