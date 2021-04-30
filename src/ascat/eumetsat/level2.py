# Copyright (c) 2021, TU Wien, Department of Geodesy and Geoinformation
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
import numpy as np
from datetime import datetime

from ascat.read_native.nc import AscatL2NcFile
from ascat.read_native.bufr import AscatL2BufrFile
from ascat.read_native.eps_native import AscatL2EpsFile
from ascat.utils import get_toi_subset, get_roi_subset
from ascat.file_handling import ChronFiles


class AscatL2File:

    """
    ASCAT Level 2 reader class.
    """

    def __init__(self, filename, file_format=None):
        """
        Initialize AscatL2File.

        Parameters
        ----------
        filename : str
            Filename.
        file_format : str, optional
            File format: '.nat', '.nc', '.bfr', '.h5' (default: None).
            If None file format will be guessed based on the file ending.
        """
        self.filename = filename
        self.fid = None

        if file_format is None:
            file_format = get_file_format(self.filename)

        self.file_format = file_format

        if self.file_format in ['.nat', '.nat.gz']:
            self.fid = AscatL2EpsFile(self.filename)
        elif self.file_format in ['.nc', '.nc.gz']:
            self.fid = AscatL2NcFile(self.filename)
        elif self.file_format in ['.bfr', '.bfr.gz', '.buf', 'buf.gz']:
            self.fid = AscatL2BufrFile(self.filename)
        else:
            raise RuntimeError("ASCAT Level 2 file format unknown")

    def read(self, toi=None, roi=None, generic=True, to_xarray=False):
        """
        Read ASCAT Level 2 data.

        Parameters
        ----------
        toi : tuple of datetime, optional
            Filter data for given time of interest (default: None).
        roi : tuple of 4 float, optional
            Filter data for region of interest (default: None).
            e.g. latmin, lonmin, latmax, lonmax

        Returns
        -------
        ds : xarray.Dataset, numpy.ndarray
            ASCAT Level 2 data.
        """
        ds = self.fid.read(generic=generic, to_xarray=to_xarray)

        if toi:
            ds = get_toi_subset(ds, toi)

        if roi:
            ds = get_roi_subset(ds, roi)

        return ds

    def read_period(self, dt_start, dt_end, **kwargs):
        """
        Read interval.
        """
        return self.read(toi=(dt_start, dt_end), **kwargs)

    def close(self):
        """
        Close file.
        """
        self.fid.close()


def get_file_format(filename):
    """
    Try to guess the file format from the extension.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    file_format : str
        File format indicator.
    """
    if os.path.splitext(filename)[1] == '.gz':
        file_format = os.path.splitext(os.path.splitext(filename)[0])[1]
    else:
        file_format = os.path.splitext(filename)[1]

    return file_format


class AscatL2BufrFileList(ChronFiles):

    def __init__(self, path, sat='A', res='SMR'):
        """
        Initialize.
        """
        sat_lut = {'A': 2, 'B': 1, 'C': 3}
        self.sat = sat_lut[sat]
        self.res = res
        fn_templ = 'M0{sat}-ASCA-ASC{res}*-*-*-{date}.000000000Z-*-*.bfr'

        super().__init__(path, AscatL2File, fn_templ)

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
        fn_read_fmt = {'date': timestamp.strftime('%Y%m%d%H%M%S'),
                       'sat': self.sat, 'res': self.res}
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(os.path.basename(filename)[25:39],
                                 '%Y%m%d%H%M%S')

    def _merge_data(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        return np.hstack(data)


class AscatL2NcFileList(ChronFiles):

    def __init__(self, path, sat='A', res='SMR'):
        """
        Initialize.
        """
        self.sat = sat
        res_lut = {'SMR': '125', 'SMO': '250'}
        self.sampling = res_lut[res]
        fn_templ = 'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOP{sat}+ASCAT_C_EUMP_{date}_*_eps_o_{sampling}_ssm_l2.nc'

        super().__init__(path, AscatL2File, fn_templ)

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
        fn_read_fmt = {'date': timestamp.strftime('%Y%m%d%H%M%S'),
                       'sat': self.sat, 'sampling': self.sampling}
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(os.path.basename(filename)[62:76],
                                 '%Y%m%d%H%M%S')

    def _merge_data(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        return np.hstack(data)


class AscatL2EpsFileList(ChronFiles):

    def __init__(self, path, sat='A', res='SMR'):
        """
        Initialize.
        """
        sat_lut = {'A': 2, 'B': 1, 'C': 3}
        self.sat = sat_lut[sat]
        self.res = res
        fn_templ = 'ASCA_{res}_02_M0{sat}_{date}Z_*_*_*_*.nat'

        super().__init__(path, AscatL2File, fn_templ)

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
        fn_read_fmt = {'date': timestamp.strftime('%Y%m%d%H%M%S'),
                       'sat': self.sat, 'res': self.res}
        fn_write_fmt = None
        sf_read_fmt = None
        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename):
        """
        Parse date from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        date : datetime
            Parsed date.
        """
        return datetime.strptime(os.path.basename(filename)[16:30],
                                 '%Y%m%d%H%M%S')

    def _merge_data(self, data):
        """
        Merge data.

        Parameters
        ----------
        data : list
            List of array.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        return np.hstack(data)
