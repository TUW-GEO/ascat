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
General Level 2 data readers for ASCAT data in all formats. Not specific to distributor.
"""
import time
import os

from pygeobase.io_base import ImageBase
from pygeobase.io_base import MultiTemporalImageBase
from pygeobase.io_base import IntervalReadingMixin
from pygeobase.object_base import Image
import ascat.data_readers.read_eps as read_eps
import ascat.data_readers.read_bufr as read_bufr
import ascat.level2 as level2

class AscatL2Image(ImageBase):
    """
    General Level 2 Image
    """
    def __init__(self, *args, **kwargs):
        """
        Initialization of i/o object.
        """
        super(AscatL2Image, self).__init__(*args, **kwargs)

    def read(self, timestamp=None, file_format=None, **kwargs):

        if file_format == None:
            file_format = get_file_format(self.filename)

        if file_format == ".nat":
            longitude, latitude, data, metadata = read_eps.read_eps_l2(self.filename)
            img = Image(longitude, latitude, data, metadata,
                     timestamp, timekey='jd')

        elif file_format == ".nc":
            img = level2.AscatL2SsmNcFile(self.filename).read(timestamp)

        elif file_format == ".bfr" or file_format == ".buf":
            img = level2.AscatL2SsmBufrFile(self.filename).read(timestamp)

        else:
            raise RuntimeError("Format not found, please indicate the file format. [\".nat\", \".nc\", \".bfr\"]")

        return img

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def get_file_format(filename):
    if os.path.splitext(filename)[1] == '.gz':
        file_format = os.path.splitext(os.path.splitext(filename)[0])[1]
    else:
        file_format = os.path.splitext(filename)[1]
    return file_format


def test_level2():
    start_time = time.time()

    test = AscatL2Image('/home/mschmitz/Desktop/ascat_test_data/level2/eps_nat/ASCA_SMR_02_M01_20180408011800Z_20180408025958Z_N_O_20180408030052Z.nat')
    p = test.read()

    elapsed_time = time.time() - start_time
    print (elapsed_time)

if __name__ == '__main__':
    test_level2()