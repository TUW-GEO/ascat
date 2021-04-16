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
Test file search methods.
"""

import tempfile
from datetime import datetime

import numpy as np

from ascat.file_search import CsvFileRW


def test_csv():
    """
    Testing writing and reading CSV files.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    dtype = np.dtype([('date', 'datetime64[s]'), ('num', np.int32)])
    dates = np.datetime64('2000-01-01') + np.arange(3)
    file_length = np.array([3, 5, 2, 7, 1]) * 60

    csv = CsvFileRW(tmp_dir.name)

    j = 0
    k = 0
    for date in dates:
        arr = []
        for i in range(86400):
            dt = date + np.timedelta64(i, 's')
            arr.append((dt, i))
            k = k + 1
            if k == file_length[j]:
                arr = np.array(arr, dtype=dtype)
                timestamp = arr[0]['date'].astype(datetime)
                csv.write(arr, timestamp)
                k = 0
                arr = []
                j = j + 1
                if np.mod(j, len(file_length)) == 0:
                    j = 0

    # data = csv.read(datetime(2000, 1, 1))
    # print(data)

    # data = csv.read(datetime(2000, 1, 1, 3))
    # print(data)

    # first_list = csv.search_date(datetime(2000, 1, 1))
    # print(len(first_list), first_list)

    # sec_list = csv.search_period(datetime(2000, 1, 1), datetime(2000, 1, 2))
    # print(len(sec_list), sec_list)

    # if first_list == sec_list:
    #     print('Lists are exactly equal')
    # else:
    #     print('Lists are not equal')

    period = (datetime(2000, 1, 1, 0, 0, 4), datetime(2000, 1, 1, 0, 0, 10))
    data = csv.read_period(*period)
    print(data)

    tmp_dir.cleanup()
