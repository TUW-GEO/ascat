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
Tests for general level 1 readers.
'''

import unittest
import os
import numpy.testing as nptest
from ascat.level1 import AscatL1NcFile


def test_ascat_l1_netcdf_reading():

    data_path = os.path.join(
        os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L1_SZR_NC')
    fname_nc = os.path.join(
        data_path,
        'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170101012400_52940_eps_o_125_l1.nc')
    reader_nc = AscatL1NcFile(fname_nc)

    data_nc, meta, timestamp, lons_nc, lats_nc, time_var_nc = reader_nc.read()
    keys = ['siga', 'f_ff', 'f_usablea', 'f_usablef', 'sat_track_azi',
            'sigf', 'f_fm', 'sigm', 'f_usablem', 'f_saa', 'f_saf',
            'swath_indicator', 'f_tela', 'f_telf', 'f_sam', 'f_oam',
            'f_fa', 'f_vf', 'f_oaf', 'f_oaa', 'inca',
            'node_num', 'incf', 'num_valf', 'as_des_pass', 'incm',
            'f_landm', 'azia', 'azif', 'f_va', 'f_landf', 'f_landa', 'azim',
            'f_reff', 'f_telm', 'jd', 'num_valm', 'num_vala', 'kpa',
            'f_refm', 'f_kpm', 'kpf', 'f_kpa', 'f_vm', 'f_refa', 'kpm',
            'f_kpf', 'spacecraft_id', 'abs_orbit_nr', 'line_num']
    assert sorted(data_nc.keys()) == sorted(keys)
    meta_should = {'processor_major_version': u'10',
                   'format_minor_version': u'0',
                   'processor_minor_version': u'0',
                   'format_major_version': u'12'}
    assert meta_should == meta
    assert lons_nc.shape == (275520,)
    assert lats_nc.shape == (275520,)
    for key in keys:
        data_nc[key].shape == (275520,)
    nptest.assert_almost_equal(data_nc['jd'][0], 2457754.55833333)
    nptest.assert_almost_equal(data_nc['spacecraft_id'][2332], 2)


def test_ascat_l1_netcdf_reading_satellite_id_translation():

    data_path = os.path.join(
        os.path.dirname(__file__),  'test-data', 'sat', 'eumetsat', 'ASCAT_L1_SZR_NC')
    fname_nc = os.path.join(
        data_path,
        'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20170101012400_52940_eps_o_125_l1.nc')
    reader_nc = AscatL1NcFile(fname_nc,
                              satellite_id_translation=True)

    data_nc, meta, timestamp, lons_nc, lats_nc, time_var_nc = reader_nc.read()
    nptest.assert_almost_equal(data_nc['spacecraft_id'][2345], 3)
