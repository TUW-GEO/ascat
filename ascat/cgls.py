# Copyright (c) 2017, Vienna University of Technology (TU Wien),
# Department of Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# All information contained herein is, and remains the property of Vienna
# University of Technology (TU Wien), Department of Geodesy and Geoinformation
# (GEO). The intellectual and technical concepts contained herein are
# proprietary to Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO). Dissemination of this information or
# reproduction of this material is forbidden unless prior written permission
# is obtained from Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).

'''
Module for reading CGLOPS SWI TS products
'''

import pynetcf.time_series as netcdf_dataset
import pygeogrids.netcdf as netcdf

import os
import numpy as np


class SWI_TS(netcdf_dataset.GriddedNcOrthoMultiTs):

    """
    SWI TS reader for timeseries data from CGLOPS

    Parameters
    ----------
    data_path: string
        path to the netCDF files
    parameters: list
        list of parameters to read from netCDF file
    dt: string, optional
        datetime in the filenames of the cells
    version: string, optional
        version number of the files
    grid_fname: string, optional
        filename + path of the grid netCDF file,
        default is the standard grid file (c_gls_SWI-STATIC-DGG_201501010000_GLOBE_ASCAT_V3.0.1.nc)
        in the same folder as the data
    read_bulk: boolean, optional
        if set to true then a complete 5x5 degree cell will be read at once
        providing speedup if the complete data is needed.
    fname_template: string, optional
        Filename template. Has to have two slots for {dt} and {version} and a
        slot for the cell number that is available for further formatting.
        Because of this the cell number location has to be written as '{{:04d}}'.
        The has to be without the .nc ending since this is added during reading.
    """

    def __init__(self, data_path, parameters=['SWI_001', 'SWI_005', 'SWI_010',
                                              'SWI_015', 'SWI_020', 'SWI_040',
                                              'SWI_060', 'SWI_100', 'SSF'],
                 dt='201612310000', version='3.0.1',
                 grid_fname=None, read_bulk=True,
                 fname_template='c_gls_SWI-TS_{dt}_C{{:04d}}_ASCAT_V{version}'):

        if grid_fname is None:
            grid_fname = os.path.join(
                data_path, 'c_gls_SWI-STATIC-DGG_201501010000_GLOBE_ASCAT_V3.0.1.nc')
        grid = netcdf.load_grid(grid_fname, location_var_name='location_id',
                                subset_flag='land_flag')

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
            fn_format=fname_template.format(dt=dt, version=version),
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
