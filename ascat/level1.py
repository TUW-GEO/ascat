# Copyright (c) 2018, TU Wien, Department of Geodesy and Geoinformation
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
import lxml.etree as etree
from tempfile import NamedTemporaryFile
from gzip import GzipFile
from collections import OrderedDict

import numpy as np

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image
import ascat.lvl1_readers.read_eps_szx as read_eps_szx

"""
General Level 1 data readers for ASCAT data in all formats. Not specific to distributor.
"""

class AscatL1Image(ImageBase):
    """
    General Level 1 Image
    """
    def __init__(self, filename, mode='r', **kwargs):
        """
        Initialization of i/o object.
        """
        super(AscatL1Image, self).__init__(filename, mode=mode, **kwargs)

    def read(self, timestamp=None, **kwargs):
        file_format = get_file_format(self.filename)
        if file_format == ".nat":
            data = read_eps(self.filename)
            return data
        elif file_format == ".nc":
            data = read_netCDF(self.filename)
            return data
        elif file_format == ".bfr":
            data = read_bufr(self.filename)
            return data

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass

def read_eps_szf(filename):
    pass

def get_file_format(filename):
    if os.path.splitext(filename)[1] == '.gz':
        file_format = os.path.splitext(os.path.splitext(filename)[0])[1]
    else:
        file_format = os.path.splitext(filename)[1]
    return file_format


def read_netCDF_szx(filename):
    pass

def read_bufr_szx(filename):
    pass

def read_eps(filename):
    basename = os.path.basename(filename)
    if basename.startswith("ASCA_SZR") or basename.startswith(
            "ASCA_SZO"):
        data = read_eps_szx.read_eps_szx(filename)
        return Image()
    elif basename.startswith("ASCA_SZF"):
        data = read_eps_szf(filename)
        return Image()

def read_netCDF(filename):
    pass

def read_bufr(filename):
    pass

def test_level1():
    test = AscatL1Image('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M01_20160101000900Z_20160101015058Z_N_O_20160101005610Z.nat')
    test.read()

if __name__ == '__main__':
    test_level1()