# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

from ascat.file_handling import Filenames
from ascat.utils import get_toi_subset, get_roi_subset
from ascat.utils import mask_dtype_nans


class AscatFile(Filenames):
    """
    Class reading ASCAT files.
    """
    def __init__(self, filename):
        """
        Initialize AscatFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        super().__init__(filename)

    def read(self, toi=None, roi=None, **kwargs):
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
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.
        """
        data, metadata = super().read(**kwargs)

        if toi:
            data = get_toi_subset(data, toi)

        if roi:
            data = get_roi_subset(data, roi)

        return data, metadata

    def read_period(self, dt_start, dt_end, **kwargs):
        """
        Read interval.

        Parameters
        ----------
        dt_start : datetime
            Start datetime.
        dt_end : datetime
            End datetime.

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.
        """
        return self.read(toi=(dt_start, dt_end), **kwargs)
