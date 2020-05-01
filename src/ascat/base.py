# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
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
from datetime import datetime, timedelta

from pygeobase.io_base import MultiTemporalImageBase


class ASCAT_MultiTemporalImageBase(MultiTemporalImageBase):
    """
    Base class for the lvl1b and lvl2 MultiTemporal data readers
    """

    def __init__(self, *args, **kwargs):
        super(ASCAT_MultiTemporalImageBase, self).__init__(*args, **kwargs)

    def _get_orbit_start_date(self, filename):
        """
        Returns the datetime of the file.

        Parameters
        ----------
        filename : full name (including the path) of the file

        Returns
        -------
        dates : datetime object
            datetime from the filename
        """
        # if your data comes from the EUMETSAT EO Portal this function can
        if self.eo_portal is True:
            filename_base = os.path.basename(filename)
            fln_spl = filename_base.split('_')[4]
            fln_datetime = fln_spl[:-1]
            return datetime.strptime(fln_datetime, self.datetime_format)

        else:
            orbit_start_str = os.path.basename(filename)[
                self.filename_datetime_format[0]:
                self.filename_datetime_format[1]]
            return datetime.strptime(orbit_start_str,
                                     self.filename_datetime_format[2])

    def tstamps_for_daterange(self, startdate, enddate):
        """
        Returns the possible timestamps of the given daterange as a datetime
        array.

        Parameters
        ----------
        startdate : datetime.date or datetime.datetime
            start date
        enddate : datetime.date or datetime.datetime
            end date

        Returns
        -------
        dates : list
            list of datetimes
        """
        file_list = []
        delta_all = enddate - startdate
        timestamps = []

        for i in range(delta_all.days + 1):
            timestamp = startdate + timedelta(days=i)

            files = self._search_files(
                timestamp, custom_templ=self.day_search_str)

            file_list.extend(sorted(files))

        for filename in file_list:
            timestamps.append(self._get_orbit_start_date(filename))

        timestamps = [dt for dt in timestamps if (
            dt >= startdate and dt <= enddate)]
        return timestamps
