# Copyright (c) 2016,Vienna University of Technology,
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
Small untility functions that are needed but do not make sense in own package (yet).

Some are duplicates from pytesmo but we avoid the pytesmo dependency if we put
them also here.
'''

import numpy as np


def doy(month, day, year=None):
    """
    Calculation of day of year. If year is provided it will be tested for
    leap years.

    Parameters
    ----------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32, optional
        Year.

    Retruns
    -------
    doy : numpy.ndarray or int32
        Day of year.
    """
    daysPast = np.array([0, 31, 60, 91, 121, 152, 182, 213,
                         244, 274, 305, 335, 366])

    day_of_year = daysPast[month - 1] + day

    if year is not None:
        nonleap_years = np.invert(is_leap_year(year))
        day_of_year = day_of_year - nonleap_years + \
            np.logical_and(day_of_year < 60, nonleap_years)

    return day_of_year


def is_leap_year(year):
    """
    Check if year is a leap year.

    Parameters
    ----------
    year : numpy.ndarray or int32

    Returns
    -------
    leap_year : numpy.ndarray or boolean
        True if year is a leap year.
    """
    return np.logical_or(np.logical_and(year % 4 == 0, year % 100 != 0),
                         year % 400 == 0)
