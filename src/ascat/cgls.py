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
CGLS SWI interface.
"""

import os
import glob
import numpy as np

import pynetcf.time_series as netcdf_dataset
import pygeogrids.netcdf as netcdf


class SWI_TS(netcdf_dataset.GriddedNcOrthoMultiTs):

    """
    SWI TS reader for timeseries data from CGLOPS

    Parameters
    ----------
    data_path : string
        path to the netCDF files
    parameters : list
        list of parameters to read from netCDF file
    dt : string, optional
        datetime in the filenames of the cells.
        If not given it is detected from the files in the data_path.
        Automatic detection only works if the files follow the CGLS
        naming convention.
    version : string, optional
        version number of the files
        If not given it is detected from the files in the data_path.
        Automatic detection only works if the files follow the CGLS
        naming convention.
    grid_fname : string, optional
        filename + path of the grid netCDF file,
        default is the standard grid file
        (c_gls_SWI-STATIC-DGG_201501010000_GLOBE_ASCAT_V3.0.1.nc)
        in the same folder as the data
    read_bulk : boolean, optional
        if set to true then a complete 5x5 degree cell will be read at once
        providing speedup if the complete data is needed.
    fname_template : string, optional
        Filename template. Has to have three slots for {dt}, {version} and a
        slot for the {cell} number that is available for further formatting.
        The has to be without the .nc ending since this is added during reading.
    cell_fn : string, optional
        cell number in the fname_template.
    """

    def __init__(self, data_path, parameters=['SWI_001', 'SWI_005', 'SWI_010',
                                              'SWI_015', 'SWI_020', 'SWI_040',
                                              'SWI_060', 'SWI_100', 'SSF'],
                 dt=None, version=None,
                 grid_fname=None, read_bulk=True,
                 fname_template='c_gls_SWI-TS_{dt}_C{cell}_ASCAT_V{version}',
                 cell_fn='{:04d}'):

        if grid_fname is None:
            grid_fname = os.path.join(
                data_path, 'c_gls_SWI-STATIC-DGG_201501010000_GLOBE_ASCAT_V3.0.1.nc')
        grid = netcdf.load_grid(grid_fname, location_var_name='location_id',
                                subset_flag='land_flag')

        # detect datetime and version if not given
        if dt is None or version is None:
            globstring = fname_template.format(dt="*",
                                               cell="*",
                                               version="*")
            found_files = glob.glob(os.path.join(data_path, globstring))
            if len(found_files) == 0:
                raise IOError("No data found in {}".format(data_path))
            fn = found_files[0]
            fn = os.path.splitext(os.path.basename(fn))[0]
            parts = fn.split('_')
        if dt is None:
            # this only works if the files follow the CGLS naming convention
            # for everything else dt should be given as a keyword
            dt = parts[3]
        if version is None:
            version = parts[-1][1:]

        scale_factors = {'SWI_001': 0.5,
                         'SWI_005': 0.5,
                         'SWI_010': 0.5,
                         'SWI_015': 0.5,
                         'SWI_020': 0.5,
                         'SWI_040': 0.5,
                         'SWI_060': 0.5,
                         'SWI_100': 0.5,
                         'QFLAG_001': 0.5,
                         'QFLAG_005': 0.5,
                         'QFLAG_010': 0.5,
                         'QFLAG_015': 0.5,
                         'QFLAG_020': 0.5,
                         'QFLAG_040': 0.5,
                         'QFLAG_060': 0.5,
                         'QFLAG_100': 0.5,
                         'SSF': 1}

        dtypes = {'SWI_001': np.uint8,
                  'SWI_005': np.uint8,
                  'SWI_010': np.uint8,
                  'SWI_015': np.uint8,
                  'SWI_020': np.uint8,
                  'SWI_040': np.uint8,
                  'SWI_060': np.uint8,
                  'SWI_100': np.uint8,
                  'QFLAG_001': np.uint8,
                  'QFLAG_005': np.uint8,
                  'QFLAG_010': np.uint8,
                  'QFLAG_015': np.uint8,
                  'QFLAG_020': np.uint8,
                  'QFLAG_040': np.uint8,
                  'QFLAG_060': np.uint8,
                  'QFLAG_100': np.uint8,
                  'SSF': np.uint8}

        super(SWI_TS, self).__init__(
            data_path, grid,
            fn_format=fname_template.format(dt=dt, version=version,
                                            cell=cell_fn),
            parameters=parameters, scale_factors=scale_factors,
            dtypes=dtypes, autoscale=False,
            automask=False, ioclass_kws={'read_bulk': read_bulk,
                                         'loc_ids_name': 'locations'})

    def _read_gp(self, gpi, period=None, mask_frozen=True):
        data = super(SWI_TS, self)._read_gp(gpi, period=period)

        if mask_frozen is True:
            unfrozen = data['SSF'].values <= 1
            data = data[unfrozen]

        for column in data:
            data.loc[data[column] > 100, column] = np.nan

        return data
