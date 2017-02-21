# Copyright (c) 2017,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Readers for data downloaded from EUMETSAT data centre (UMARF)
'''

from ascat.level2 import AscatL2SsmBufr


class AscatAL2Ssm125(AscatL2SsmBufr):
    """
    ASCAT A Level2 Soil Moisture at 12.5 km Swath Grid BUFR files from EUMETSAT

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        If the data is stored in subpaths per year or month then specify the string
        that should be used in datetime.datetime.strftime to get the subpath for a file.
        Default: ''
    """

    def __init__(self, path, month_path_str=''):
        day_search_str = 'M02-ASCA-ASCSMR02-NA-5.0-%Y%m%d*.bfr'
        file_search_str = 'M02-ASCA-ASCSMR02-NA-5.0-{datetime}*.bfr'
        datetime_format = '%Y%m%d%H%M%S'
        filename_datetime_format = (25, 39, '%Y%m%d%H%M%S')
        super(AscatAL2Ssm125, self).__init__(path, month_path_str=month_path_str,
                                             day_search_str=day_search_str,
                                             file_search_str=file_search_str,
                                             datetime_format=datetime_format,
                                             filename_datetime_format=filename_datetime_format)


class AscatBL2Ssm125(AscatL2SsmBufr):
    """
    ASCAT B Level2 Soil Moisture at 12.5 km Swath Grid BUFR files from EUMETSAT

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        If the data is stored in subpaths per year or month then specify the string
        that should be used in datetime.datetime.strftime to get the subpath for a file.
        Default: ''
    """

    def __init__(self, path, month_path_str=''):
        day_search_str = 'M01-ASCA-ASCSMR02-NA-5.0-%Y%m%d*.bfr'
        file_search_str = 'M01-ASCA-ASCSMR02-NA-5.0-{datetime}*.bfr'
        datetime_format = '%Y%m%d%H%M%S'
        filename_datetime_format = (25, 39, '%Y%m%d%H%M%S')
        super(AscatBL2Ssm125, self).__init__(path, month_path_str=month_path_str,
                                             day_search_str=day_search_str,
                                             file_search_str=file_search_str,
                                             datetime_format=datetime_format,
                                             filename_datetime_format=filename_datetime_format)


class AscatAL2Ssm250(AscatL2SsmBufr):
    """
    ASCAT A Level2 Soil Moisture at 25.0 km Swath Grid BUFR files from EUMETSAT

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        If the data is stored in subpaths per year or month then specify the string
        that should be used in datetime.datetime.strftime to get the subpath for a file.
        Default: ''
    """

    def __init__(self, path, month_path_str=''):
        day_search_str = 'M02-ASCA-ASCSMO02-NA-5.0-%Y%m%d*.bfr'
        file_search_str = 'M02-ASCA-ASCSMO02-NA-5.0-{datetime}*.bfr'
        datetime_format = '%Y%m%d%H%M%S'
        filename_datetime_format = (25, 39, '%Y%m%d%H%M%S')
        super(AscatAL2Ssm250, self).__init__(path, month_path_str=month_path_str,
                                             day_search_str=day_search_str,
                                             file_search_str=file_search_str,
                                             datetime_format=datetime_format,
                                             filename_datetime_format=filename_datetime_format)


class AscatBL2Ssm250(AscatL2SsmBufr):
    """
    ASCAT B Level2 Soil Moisture at 25.0 km Swath Grid BUFR files from EUMETSAT

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        If the data is stored in subpaths per year or month then specify the string
        that should be used in datetime.datetime.strftime to get the subpath for a file.
        Default: ''
    """

    def __init__(self, path, month_path_str=''):
        day_search_str = 'M01-ASCA-ASCSMO02-NA-5.0-%Y%m%d*.bfr'
        file_search_str = 'M01-ASCA-ASCSMO02-NA-5.0-{datetime}*.bfr'
        datetime_format = '%Y%m%d%H%M%S'
        filename_datetime_format = (25, 39, '%Y%m%d%H%M%S')
        super(AscatBL2Ssm250, self).__init__(path, month_path_str=month_path_str,
                                             day_search_str=day_search_str,
                                             file_search_str=file_search_str,
                                             datetime_format=datetime_format,
                                             filename_datetime_format=filename_datetime_format)
