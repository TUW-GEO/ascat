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
Readers for ASCAT Level 1b data for various file formats.
"""

import os
from datetime import datetime
from collections import defaultdict

import numpy as np

from ascat.read_native.nc import AscatL1bNcFile
from ascat.read_native.hdf5 import AscatL1bHdf5File
from ascat.read_native.bufr import AscatL1bBufrFile
from ascat.read_native.eps_native import AscatL1bEpsFile
from ascat.utils import get_toi_subset, get_roi_subset
from ascat.file_handling import ChronFiles


class AscatL1bFile:

    """
    ASCAT Level 1b reader class.
    """

    def __init__(self, filename, file_format=None, mode='r'):
        """
        Initialize AscatL1File.

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
        self.mode = mode

        if file_format is None:
            file_format = get_file_format(self.filename)

        self.file_format = file_format

        if self.file_format in ['.nat', '.nat.gz']:
            self.fid = AscatL1bEpsFile(self.filename)
        elif self.file_format in ['.nc', '.nc.gz']:
            self.fid = AscatL1bNcFile(self.filename)
        elif self.file_format in ['.bfr', '.bfr.gz', '.buf', 'buf.gz']:
            self.fid = AscatL1bBufrFile(self.filename)
        elif self.file_format in ['.h5', '.h5.gz']:
            self.fid = AscatL1bHdf5File(self.filename)
        else:
            raise RuntimeError("ASCAT Level 1b file format unknown")

    def read(self, toi=None, roi=None, generic=True, to_xarray=False):
        """
        Read ASCAT Level 1b data.

        Parameters
        ----------
        toi : tuple of datetime, optional
            Filter data for given time of interest (default: None).
            e.g. (datetime(2020, 1, 1, 12), datetime(2020, 1, 2))
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


class AscatL1bBufrFileList(ChronFiles):

    def __init__(self, path, sat='A', res='SZR'):
        """
        Initialize.
        """
        sat_lut = {'A': 2, 'B': 1, 'C': 3}
        self.sat = sat_lut[sat]
        self.res = res
        fn_templ = 'M0{sat}-ASCA-ASC{res}1B0200-NA-9.1-{date}.000000000Z-*-*.bfr'

        super().__init__(path, AscatL1bFile, fn_templ)

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
        return datetime.strptime(os.path.basename(filename)[29:43],
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


class AscatL1bNcFileList(ChronFiles):

    def __init__(self, path, sat='A', res='SZR'):
        """
        Initialize.
        """
        self.sat = sat
        res_lut = {'SZR': '125', 'SZO': '250'}
        self.sampling = res_lut[res]
        fn_templ = 'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOP{sat}+ASCAT_C_EUMP_{date}_*_eps_o_{sampling}_l1.nc'

        super().__init__(path, AscatL1bFile, fn_templ)

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


class AscatL1bEpsFileList(ChronFiles):

    def __init__(self, path, sat='A', res='SZR'):
        """
        Initialize.
        """
        sat_lut = {'A': 2, 'B': 1, 'C': 3}
        self.sat = sat_lut[sat]

        self.res = res

        fn_templ = 'ASCA_{res}_1B_M0{sat}_{date}Z_*_*_*_*.nat'

        super().__init__(path, AscatL1bFile, fn_templ)

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
        if self.res == 'SZF':
            merged_data = defaultdict(list)
            for antenna in ['lf', 'lm', 'la', 'rf', 'rm', 'ra']:
                for d in data:
                    merged_data[antenna].append(d.pop(antenna))
                merged_data[antenna] = np.hstack(merged_data[antenna])
        else:
            merged_data = np.hstack(data)

        return merged_data


class AscatL1bHdf5FileList(ChronFiles):

    def __init__(self, path, sat='A'):
        """
        Initialize.
        """
        sat_lut = {'A': 2, 'B': 1, 'C': 3}
        self.sat = sat_lut[sat]

        fn_templ = 'ASCA_SZF_1B_M0{sat}_{date}Z_*_*_*_*.h5'

        super().__init__(path, AscatL1bFile, fn_templ)

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
                       'sat': self.sat}
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
        merged_data = defaultdict(list)

        for antenna in ['lf', 'lm', 'la', 'rf', 'rm', 'ra']:
            for d in data:
                merged_data[antenna].append(d.pop(antenna))
            merged_data[antenna] = np.hstack(merged_data[antenna])

        return merged_data
