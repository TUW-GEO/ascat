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

"""
General Level 1 data readers for ASCAT data in all formats. Not specific to distributor.
"""

import os

from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image
import ascat.lvl1_readers.read_eps_szx as read_eps_szx


class AscatL1Image(ImageBase):
    """
    General Level 1 Image
    """
    def __init__(self, filename, **kwargs):
        """
        Initialization of i/o object.
        """
        self.beams = ['f', 'm', 'a']
        super(AscatL1Image, self).__init__(filename, **kwargs)

    def read(self, timestamp=None, file_format=None, **kwargs):
        data = {}
        metadata = {}

        if file_format == ".nat":
            raw_data = read_eps(self.filename)

        elif file_format == ".nc":
            raw_data = read_netCDF(self.filename)

        elif file_format == ".bfr":
            raw_data = read_bufr(self.filename)

        fields = ['as_des_pass', 'swath_indicator', 'node_num',
                  'sat_track_azi', 'line_num', 'jd',
                  'spacecraft_id', 'abs_orbit_nr']
        for field in fields:
            data[field] = raw_data[field]

        fields = ['azi', 'inc', 'sig', 'kp', 'f_land',
                  'f_usable', 'f_kp', 'f_f', 'f_v', 'f_oa',
                  'f_sa', 'f_tel', 'f_ref', 'num_val']
        for field in fields:
            for i, beam in enumerate(self.beams):
                data[field + beam] = raw_data[field][:, i]

        fields = ['processor_major_version',
                  'processor_minor_version', 'format_major_version',
                  'format_minor_version']
        metadata = {}
        for field in fields:
            metadata[field] = raw_data[field]

        return Image(raw_data['lon'], raw_data['lat'], data, metadata,
                     timestamp, timekey='jd')

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
        return data
    elif basename.startswith("ASCA_SZF"):
        data = read_eps_szf(filename)
        return data


def read_netCDF(filename):
    pass


def read_bufr(filename):
    pass


def test_level1():
    test = AscatL1Image('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M01_20180403012100Z_20180403030558Z_N_O_20180403030402Z.nat')
    test.read()


if __name__ == '__main__':
    test_level1()