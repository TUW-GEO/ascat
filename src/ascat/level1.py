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

import ascat.read_native.nc as nc
import ascat.read_native.hdf5 as hdf5
import ascat.read_native.bufr as bufr
import ascat.read_native.eps_native as eps_native

from ascat.utils import get_toi_subset, get_roi_subset


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
            self.fid = eps_native.AscatL1bEpsFile(self.filename)
        elif self.file_format in ['.nc', '.nc.gz']:
            self.fid = nc.AscatL1bNcFile(self.filename)
        elif self.file_format in ['.bfr', '.bfr.gz', '.buf', 'buf.gz']:
            self.fid = bufr.AscatL1bBufrFile(self.filename)
        elif self.file_format in ['.h5', '.h5.gz']:
            self.fid = hdf5.AscatL1bHdf5File(self.filename)
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

    def read_interval(self, interval, **kwargs):
        """
        Read interval.
        """
        return self.read(toi=interval, **kwargs)

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
