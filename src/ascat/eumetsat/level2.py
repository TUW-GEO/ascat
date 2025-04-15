# Copyright (c) 2025, TU Wien
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
"""
Readers for ASCAT Level 2 data for various file formats.
"""

import os
from pathlib import Path

import numpy as np

from ascat.read_native.nc import AscatL2NcFileGeneric
from ascat.read_native.bufr import AscatL2BufrFileGeneric
from ascat.read_native.eps_native import AscatL2EpsFileGeneric
from ascat.utils import get_toi_subset, get_roi_subset
from ascat.utils import get_file_format
from ascat.file_handling import ChronFiles


class AscatL2File:
    """
    Class reading ASCAT Level 2 files.
    """

    def __new__(cls, filename, file_format=None):
        """
        Initialize AscatL2File.

        Parameters
        ----------
        filename : str
            Filename.
        file_format : str, optional
            File format: ".nat", ".nc", ".bfr", ".h5" (default: None).
            If None file format will be guessed based on the file ending.
        """

        if file_format is None:
            if isinstance(filename, (str, Path)):
                file_format = get_file_format(filename)
            else:
                file_format = get_file_format(filename[0])

        if file_format in [".nat", ".nat.gz"]:
            return AscatL2EpsFileGeneric(filename)
        elif file_format in [".nc", ".nc.gz"]:
            return AscatL2NcFileGeneric(filename)
        elif file_format in [".bfr", ".bfr.gz", ".buf", "buf.gz"]:
            return AscatL2BufrFileGeneric(filename)
        else:
            raise RuntimeError("ASCAT Level 2 file format unknown")


class AscatL2BufrFileList(ChronFiles):
    """
    Class reading ASCAT L2 BUFR files.
    """

    def __init__(self, path, sat, product, filename_template=None):
        """
        Initialize.
        """
        sat_lut = {"a": 2, "b": 1, "c": 3}
        self.sat = sat_lut[sat]
        self.product = product

        if filename_template is None:
            filename_template = ("M0{sat}-ASCA-ASC{product}{placeholder1}-{placeholder2}-{placeholder3}-"
                                 "{date}.000000000Z-{placeholder4}-{placeholder5}.bfr")

        super().__init__(path, AscatL2File, filename_template, None)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {
            "date": timestamp.strftime("%Y%m%d%H%M%S"),
            "sat": self.sat,
            "product": self.product.upper(),
            "placeholder1": "*",
            "placeholder2": "*",
            "placeholder3": "*",
            "placeholder4": "*",
            "placeholder5": "*",
        }
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt


class AscatL2NcFileList(ChronFiles):
    """
    Class reading ASCAT L1b NetCDF files.
    """

    def __init__(self, path, sat, product, filename_template=None):
        """
        Initialize.

        Parameters
        ----------
        path : str
            Path to input data.
        sat : str
            Metop satellite ("a", "b", "c").
        product : str
            Product type ("szf", "szr", "szo").
        filename_template : str, optional
            Filename template (default:
            "M0{sat}-ASCA-ASC{product}1B0200-NA-9.1-{date}.000000000Z-{placeholder1}-{placeholder2}.bfr")
        """
        self.sat = sat

        lut = {"smr": "125", "smo": "250"}
        self.product = lut[product]

        if filename_template is None:
            filename_template = (
                "W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOP{sat}+"
                "ASCAT_C_EUMP_{date}_{placeholder}_eps_o_{product}_ssm_l2.nc")

        super().__init__(path, AscatL2File, filename_template, None)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {
            "date": timestamp.strftime("%Y%m%d%H%M%S"),
            "sat": self.sat.upper(),
            "product": self.product.upper(),
            "placeholder": "*",
        }
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt


class AscatL2EpsFileList(ChronFiles):
    """
    Class reading ASCAT L2 Eps files.
    """

    def __init__(self, path, sat, product, filename_template=None):
        """
        Initialize.

        Parameters
        ----------
        path : str
            Path to input data.
        sat : str
            Metop satellite ("a", "b", "c").
        product : str
            Product type ("szf", "szr", "szo").
        filename_template : str, optional
            Filename template (default:
                "ASCA_{product}_02_M0{sat}_{date}Z_*_*_*_*.nat")
        """
        sat_lut = {"a": 2, "b": 1, "c": 3, "?": "?"}
        self.sat = sat_lut[sat]
        self.product = product

        if filename_template is None:
            filename_template = "ASCA_{product}_02_M0{sat}_{date}Z_{placeholder1}_{placeholder2}_{placeholder3}_{placeholder4}.nat"

        super().__init__(path, AscatL2File, filename_template, None)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_read_fmt = {
            "date": timestamp.strftime("%Y%m%d%H%M%S"),
            "sat": self.sat,
            "product": self.product.upper(),
            "placeholder1": "*",
            "placeholder2": "*",
            "placeholder3": "*",
            "placeholder4": "*",
        }
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt
