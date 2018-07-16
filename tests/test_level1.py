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
Tests for general level 2 readers.
"""

import numpy.testing as nptest
import unittest

import ascat.level1 as level1

class Test_AscatL1Image(unittest.TestCase):

    def setUp(self):
        test_b = level1.AscatL1Image(
            '/home/mschmitz/Desktop/ascat_test_data/level1/bufr/M02-ASCA-ASCSZR1B0200-NA-9.1-20100609013900.000000000Z-20130824233100-1280350.bfr')
        self.reader_bufr = test_b.read()
        test_e = level1.AscatL1Image(
            '/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M02_20100609013900Z_20100609032058Z_R_O_20130824233100Z.nat.gz')
        self.reader_eps = test_e.read()
        test_n = level1.AscatL1Image(
            '/home/mschmitz/Desktop/ascat_test_data/level1/nc/W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA+ASCAT_C_EUMP_20100609013900_18872_eps_o_125_l1.nc')
        self.reader_nc = test_n.read()

        test_e_szf = level1.AscatL1Image(
            '/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M01_20180611041800Z_20180611055959Z_N_O_20180611050637Z.nat.gz')
        self.reader_eps_szf = test_e_szf.read()
        test_h_szf = level1.AscatL1Image(
            '/home/mschmitz/Desktop/ascat_test_data/level1/h5/ASCA_SZF_1B_M01_20180611041800Z_20180611055959Z_N_O_20180611050637Z.h5')
        self.reader_hdf5_szf = test_h_szf.read()

    def tearDown(self):
        self.reader_nc = None
        self.reader_bufr = None
        self.reader_eps = None
        self.reader_eps_szf = None
        self.reader_hdf5_szf = None

    def test_image_reading_szx(self):
        nptest.assert_allclose(self.reader_bufr.lat, self.reader_eps.lat,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_eps.lat, self.reader_nc.lat,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_nc.lat, self.reader_bufr.lat,
                               atol=1e-4)

        nptest.assert_allclose(self.reader_bufr.lon, self.reader_eps.lon,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_eps.lon, self.reader_nc.lon,
                               atol=1e-4)
        nptest.assert_allclose(self.reader_nc.lon, self.reader_bufr.lon,
                               atol=1e-4)

        matching = ['jd', 'sat_id', 'abs_line_nr', 'abs_orbit_nr', 'node_num',
                    'line_num', 'as_des_pass', 'swath', 'azif', 'azim', 'azia',
                    'incf', 'incm', 'inca', 'sigf', 'sigm', 'siga', 'kpf',
                    'kpm', 'kpa', 'kpf_quality', 'kpm_quality', 'kpa_quality',
                    'land_flagf', 'land_flagm', 'land_flaga', 'usable_flagf',
                    'usable_flagm', 'usable_flaga',
                    ]

        # lists with no data fields
        bufr_none = ['abs_line_nr', 'abs_orbit_nr', 'as_des_pass', 'azif', 'azim', 'azia', 'sigf', 'sigm', 'siga']

        # BUFR files contain less accurate data so we only compare to one 0.1
        # accuracy.
        for field in matching:
            if field not in bufr_none:
                nptest.assert_allclose(self.reader_bufr.data[field],
                                       self.reader_eps.data[field], atol=0.1)
                nptest.assert_allclose(self.reader_nc.data[field],
                                       self.reader_bufr.data[field], atol=0.1)

            nptest.assert_allclose(self.reader_eps.data[field],
                                   self.reader_nc.data[field], atol=0.1)

    def test_image_reading_szf(self):
        for szf_img in self.reader_eps_szf:
            nptest.assert_allclose(self.reader_eps_szf[szf_img].lat,
                                   self.reader_hdf5_szf[szf_img].lat,
                                   atol=1e-4)

            nptest.assert_allclose(self.reader_eps_szf[szf_img].lon,
                                   self.reader_hdf5_szf[szf_img].lon,
                                   atol=1e-4)

            matching = ['jd', 'sat_id', 'beam_number', 'abs_orbit_nr', 'node_num',
                        'line_num', 'as_des_pass', 'azi', 'inc', 'sig',
                        'land_frac', 'flagfield_rf1', 'flagfield_rf2',
                        'flagfield_pl', 'flagfield_gen1', 'flagfield_gen2',
                        'land_flag', 'usable_flag'
                        ]

            for field in matching:
                nptest.assert_allclose(self.reader_eps_szf[szf_img].data[field],
                                       self.reader_hdf5_szf[szf_img].data[field], atol=0.1)


