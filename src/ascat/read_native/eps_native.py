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
Readers for ASCAT Level 1b and Level 2 data in EPS Native format.
"""

import os
import fnmatch
from gzip import GzipFile
from collections import OrderedDict, defaultdict
from tempfile import NamedTemporaryFile

import numpy as np
import xarray as xr
import lxml.etree as etree
from cadati.jd_date import jd2dt

from ascat.utils import get_toi_subset, get_roi_subset

short_cds_time = np.dtype([('day', '>u2'), ('time', '>u4')])
long_cds_time = np.dtype([('day', '>u2'), ('ms', '>u4'), ('mms', '>u2')])

long_nan = np.iinfo(np.int32).min
ulong_nan = np.iinfo(np.uint32).max
int_nan = np.iinfo(np.int16).min
uint_nan = np.iinfo(np.uint16).max
byte_nan = np.iinfo(np.byte).min
int8_nan = np.iinfo(np.int8).max
uint8_nan = np.iinfo(np.uint8).max
float32_nan = -999999.

# 2000-01-01 00:00:00
julian_epoch = 2451544.5


class AscatL1bEpsSzfFile:

    """
    ASCAT Level 1b EPS Native reader class.
    """

    def __init__(self, filename):
        """
        Initialize AscatL1bEpsFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        self.filename = filename

    def read(self, toi=None, roi=None, generic=True, to_xarray=False):
        """
        Read ASCAT Level 1b data.

        Returns
        -------
        ds : xarray.Dataset
            ASCAT Level 1b data.
        """
        ds = read_eps_l1b(self.filename, generic, to_xarray,
                          full=False, unsafe=True, scale_mdr=False)

        if toi:
            ds = get_toi_subset(ds, toi)

        if roi:
            ds = get_roi_subset(ds, roi)

        return ds

    def read_period(self, dt_start, dt_end, **kwargs):
        """
        Read interval.
        """
        return self.read(toi=(dt_start, dt_end), **kwargs)

    def close(self):
        """
        Close file.
        """
        pass


class AscatL1bEpsFile:

    """
    ASCAT Level 1b EPS Native reader class.
    """

    def __init__(self, filename):
        """
        Initialize AscatL1bEpsFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        self.filename = filename

    def read(self, generic=False, to_xarray=False):
        """
        Read ASCAT Level 1b data.

        Returns
        -------
        ds : xarray.Dataset
            ASCAT Level 1b data.
        """
        return read_eps_l1b(self.filename, generic, to_xarray)

    def close(self):
        """
        Close file.
        """
        pass


class AscatL2EpsFile:

    """
    ASCAT Level 2 EPS Native reader class.
    """

    def __init__(self, filename):
        """
        Initialize AscatL2EpsFile.

        Parameters
        ----------
        filename : str
            Filename.
        """
        self.filename = filename

    def read(self, generic=False, to_xarray=False):
        """
        Read ASCAT Level 2 data.

        Returns
        -------
        ds : dict, xarray.Dataset
            ASCAT Level 1b data.
        """
        return read_eps_l2(self.filename, generic, to_xarray)

    def close(self):
        """
        Close file.
        """
        pass


class EPSProduct:

    """
    Class for reading EPS products.
    """

    def __init__(self, filename):
        """
        Initialize EPSProduct.

        Parameters
        ----------
        filename : str
            EPS Native Filename.
        """
        self.filename = filename
        self.fid = None
        self.mphr = None
        self.sphr = None
        self.aux = defaultdict(list)
        self.mdr = None
        self.scaled_mdr = None
        self.xml_file = None
        self.xml_doc = None
        self.mdr_template = None
        self.scaled_template = None
        self.sfactor = None

        self.grh_dtype = np.dtype([('record_class', 'u1'),
                                   ('instrument_group', 'u1'),
                                   ('record_subclass', 'u1'),
                                   ('record_subclass_version', 'u1'),
                                   ('record_size', '>u4'),
                                   ('record_start_time', short_cds_time),
                                   ('record_stop_time', short_cds_time)])

        self.ipr_dtype = np.dtype([('grh', self.grh_dtype),
                                   ('target_record_class', 'u1'),
                                   ('target_instrument_group', 'u1'),
                                   ('target_record_subclass', 'u1'),
                                   ('target_record_offset', '>u4')])

        self.pointer_dtype = np.dtype([('grh', self.grh_dtype),
                                       ('aux_data_pointer', 'u1', 100)])

        self.filesize = os.path.getsize(self.filename)

    def read_mphr(self):
        """
        Read only Main Product Header Record (MPHR).
        """
        with open(self.filename, 'rb') as fid:
            grh = np.fromfile(fid, dtype=self.grh_dtype, count=1)[0]
            if grh['record_class'] == 1:
                mphr = fid.read(grh['record_size'] - grh.itemsize)
                mphr = OrderedDict(item.replace(' ', '').split('=')
                                   for item in
                                   mphr.decode("utf-8").split('\n')[:-1])

        return mphr

    def read(self, full=True, unsafe=False, scale_mdr=True):
        """
        Read EPS file.

        Parameters
        ----------
        full : bool, optional
            Read full file content (True) or just Main Product Header
            Record (MPHR) and Main Data Record (MDR) (False). Default: True
        unsafe : bool, optional
            If True it is (unsafely) assumed that MDR are continuously
            stacked until the end of file. Makes reading a lot faster.
            Default: False
        scale_mdr : bool, optional
            Compute scaled MDR (True) or not (False). Default: True

        Returns
        -------
        mphr : dict self.sphr, self.aux, self.mdr, scaled_mdr
            Main Product Header Record (MPHR).
        sphr : dict
            Secondary Product Header Product (SPHR).
        aux : dict
            Auxiliary Header Products.
        mdr : numpy.ndarray
            Main Data Record (MDR)
        scaled_mdr : numpy.ndarray
            Scaled Main Data Record (MPHR) or None if not computed.
        """
        self.fid = open(self.filename, 'rb')

        abs_pos = 0
        grh = None
        prev_grh = None
        record_count = 0

        while True:
            # read generic record header of data block
            grh = np.fromfile(self.fid, dtype=self.grh_dtype, count=1)[0]

            if grh['record_class'] == 8 and unsafe:
                if np.mod((self.filesize - abs_pos),
                          self.mdr_template.itemsize) != 0:
                    # Unsafe reading fails, switching to safe reading
                    unsafe = False
                else:
                    num_mdr = (self.filesize -
                               abs_pos) // self.mdr_template.itemsize
                    self.fid.seek(abs_pos)
                    self.read_record_class(grh, num_mdr)
                    break

            if prev_grh is None:
                prev_grh = grh

            if ((prev_grh['record_class'] != grh['record_class']) or
                    (prev_grh['record_subclass'] != grh['record_subclass'])):

                # compute record start position of previous record
                start_pos = (abs_pos - prev_grh['record_size'] * record_count)
                self.fid.seek(start_pos)

                if full or (prev_grh['record_class'] == 8 or
                            prev_grh['record_class'] == 1):
                    # read previous record, because new one is coming
                    self.read_record_class(prev_grh, record_count)

                # reset record class count
                record_count = 1
            else:
                # same record class as before, increase count
                record_count += 1

            abs_pos += grh['record_size']

            # position after record
            self.fid.seek(abs_pos)

            # store grh
            prev_grh = grh

            # end of file?
            if abs_pos == self.filesize:

                # compute record start position of previous record class
                start_pos = (abs_pos - prev_grh['record_size'] * record_count)
                self.fid.seek(start_pos)

                # read final record class(es)
                self.read_record_class(prev_grh, record_count)

                break

        self.fid.close()

        if scale_mdr:
            self.scaled_mdr = self._scaling(self.mdr, self.scaled_template,
                                            self.mdr_sfactor)

        return self.mphr, self.sphr, self.aux, self.mdr, self.scaled_mdr

    def read_record_class(self, grh, record_count):
        """
        Read record class.

        Parameters
        ----------
        grh : numpy.ndarray
            Generic record header.
        record_count : int
            Number of records.
        """
        # mphr (Main Product Header Reader)
        if grh['record_class'] == 1:
            self.fid.seek(grh.itemsize, 1)
            self._read_mphr(grh)

            # find the xml file corresponding to the format version
            # and load template
            self.xml_file = self._get_eps_xml()
            self.xml_doc = etree.parse(self.xml_file)
            self.mdr_template, self.scaled_template, self.mdr_sfactor = \
                self._read_xml_mdr()

        # sphr (Secondary Product Header Record)
        elif grh['record_class'] == 2:
            self.fid.seek(grh.itemsize, 1)
            self._read_sphr(grh)

        # ipr (Internal Pointer Record)
        elif grh['record_class'] == 3:
            data = np.fromfile(self.fid, dtype=self.ipr_dtype,
                               count=record_count)
            self.aux['ipr'].append(data)

        # geadr (Global External Auxiliary Data Record)
        elif grh['record_class'] == 4:
            data = self._read_pointer(record_count)
            self.aux['geadr'].append(data)

        # veadr (Variable External Auxiliary Data Record)
        elif grh['record_class'] == 6:
            data = self._read_pointer(record_count)
            self.aux['veadr'].append(data)

        # viadr (Variable Internal Auxiliary Data Record)
        elif grh['record_class'] == 7:
            template, scaled_template, sfactor = self._read_xml_viadr(
                grh['record_subclass'])
            viadr_element = np.fromfile(self.fid, dtype=template,
                                        count=record_count)

            viadr_element_sc = self._scaling(viadr_element,
                                             scaled_template, sfactor)

            # store viadr_grid separately
            if grh['record_subclass'] == 8:
                self.aux['viadr_grid'].append(viadr_element)
                self.aux['viadr_grid_scaled'].append(viadr_element_sc)
            else:
                self.aux['viadr'].append(viadr_element)
                self.aux['viadr_scaled'].append(viadr_element_sc)

        # mdr (Measurement Data Record)
        elif grh['record_class'] == 8:
            if grh['instrument_group'] == 13:
                self.dummy_mdr = np.fromfile(
                    self.fid, dtype=self.mdr_template, count=record_count)
            else:
                self.mdr = np.fromfile(
                    self.fid, dtype=self.mdr_template, count=record_count)
                self.mdr_counter = record_count
        else:
            raise RuntimeError("Record class not found.")

    def _scaling(self, unscaled_mdr, scaled_template, sfactor):
        """
        Scale the MDR.

        Parameters
        ----------
        unscaled_mdr : numpy.ndarray
            Raw MDR.
        scaled_template : numpy.dtype
            Scaled MDR template.
        sfactor : dict
            Scale factors.

        Returns
        -------
        scaled_mdr : numpy.ndarray
            Scaled MDR.
        """
        scaled_mdr = np.empty(unscaled_mdr.shape, dtype=scaled_template)

        for key, value in sfactor.items():
            if value != 1:
                scaled_mdr[key] = unscaled_mdr[key] * 1./value
            else:
                scaled_mdr[key] = unscaled_mdr[key]

        return scaled_mdr

    def _read_mphr(self, grh):
        """
        Read Main Product Header (MPHR).
        """
        mphr = self.fid.read(grh['record_size'] - grh.itemsize)
        self.mphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in
                                mphr.decode("utf-8").split('\n')[:-1])

    def _read_sphr(self, grh):
        """
        Read Special Product Header (SPHR).
        """
        sphr = self.fid.read(grh['record_size'] - grh.itemsize)
        self.sphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in
                                sphr.decode("utf-8").split('\n')[:-1])

    def _read_pointer(self, count=1):
        """
        Read pointer record.
        """
        record = np.fromfile(self.fid, dtype=self.pointer_dtype, count=count)

        return record

    def _get_eps_xml(self):
        """
        Find the corresponding eps xml file.
        """
        format_path = os.path.join(os.path.dirname(__file__), 'formats')

        # loop through files where filename starts with 'eps_ascat'.
        for filename in fnmatch.filter(os.listdir(format_path), 'eps_ascat*'):
            doc = etree.parse(os.path.join(format_path, filename))
            file_extension = doc.xpath('//file-extensions')[0].getchildren()[0]

            format_version = doc.xpath('//format-version')
            for elem in format_version:
                major = elem.getchildren()[0]
                minor = elem.getchildren()[1]

                # return the xml file matching the metadata of the datafile.
                if major.text == self.mphr['FORMAT_MAJOR_VERSION'] and \
                        minor.text == self.mphr['FORMAT_MINOR_VERSION'] and \
                        self.mphr[
                            'PROCESSING_LEVEL'] in file_extension.text and \
                        self.mphr['PRODUCT_TYPE'] in file_extension.text:
                    return os.path.join(format_path, filename)

    def _read_xml_viadr(self, subclassid):
        """
        Read xml record of viadr class.
        """
        elements = self.xml_doc.xpath('//viadr')
        data = OrderedDict()
        length = []

        # find the element with the correct subclass
        for elem in elements:
            item_dict = dict(elem.items())
            subclass = int(item_dict['subclass'])
            if subclass == subclassid:
                break

        for child in elem.getchildren():

            if child.tag == 'delimiter':
                continue

            child_items = dict(child.items())
            name = child_items.pop('name')

            # check if the item is of type longtime
            longtime_flag = ('type' in child_items and
                             'longtime' in child_items['type'])

            # append the length if it isn't the special case of type longtime
            try:
                var_len = child_items.pop('length')
                if not longtime_flag:
                    length.append(np.int(var_len))
            except KeyError:
                pass

            data[name] = child_items

            if child.tag == 'array':
                for arr in child.iterdescendants():
                    arr_items = dict(arr.items())
                    if arr.tag == 'field':
                        data[name].update(arr_items)
                    else:
                        try:
                            var_len = arr_items.pop('length')
                            length.append(np.int(var_len))
                        except KeyError:
                            pass

            if length:
                data[name].update({'length': length})
            else:
                data[name].update({'length': 1})

            length = []

        conv = {'longtime': long_cds_time, 'time': short_cds_time,
                'boolean': 'u1', 'integer1': 'i1',
                'uinteger1': 'u1', 'integer': '>i4',
                'uinteger': '>u4', 'integer2': '>i2',
                'uinteger2': '>u2', 'integer4': '>i4',
                'uinteger4': '>u4', 'integer8': '>i8',
                'enumerated': 'u1', 'string': 'str', 'bitfield': 'u1'}

        scaling_factor = {}
        scaled_dtype = []
        dtype = []

        for key, value in data.items():

            if 'scaling-factor' in value:
                sf_dtype = np.float32
                sf_split = value['scaling-factor'].split('^')
                scaling_factor[key] = np.int(sf_split[0])**np.int(sf_split[1])
            else:
                sf_dtype = conv[value['type']]
                scaling_factor[key] = 1

            length = value['length']

            if length == 1:
                scaled_dtype.append((key, sf_dtype))
                dtype.append((key, conv[value['type']]))
            else:
                scaled_dtype.append((key, sf_dtype, length))
                dtype.append((key, conv[value['type']], length))

        return np.dtype(dtype), np.dtype(scaled_dtype), scaling_factor

    def _read_xml_mdr(self):
        """
        Read xml record of mdr class.
        """
        elements = self.xml_doc.xpath('//mdr')
        data = OrderedDict()
        length = []
        elem = elements[0]

        for child in elem.getchildren():

            if child.tag == 'delimiter':
                continue

            child_items = dict(child.items())
            name = child_items.pop('name')

            # check if the item is of type bitfield
            bitfield_flag = ('type' in child_items and
                             ('bitfield' in child_items['type'] or 'time' in
                              child_items['type']))

            # append the length if it isn't the special case of type
            # bitfield or time
            try:
                var_len = child_items.pop('length')
                if not bitfield_flag:
                    length.append(np.int(var_len))
            except KeyError:
                pass

            data[name] = child_items

            if child.tag == 'array':
                for arr in child.iterdescendants():
                    arr_items = dict(arr.items())

                    # check if the type is bitfield
                    bitfield_flag = ('type' in arr_items and
                                     'bitfield' in arr_items['type'])

                    if bitfield_flag:
                        data[name].update(arr_items)
                        break
                    else:
                        if arr.tag == 'field':
                            data[name].update(arr_items)
                        else:
                            try:
                                var_len = arr_items.pop('length')
                                length.append(np.int(var_len))
                            except KeyError:
                                pass

            if length:
                data[name].update({'length': length})
            else:
                data[name].update({'length': 1})

            length = []

        conv = {'longtime': long_cds_time, 'time': short_cds_time,
                'boolean': 'u1', 'integer1': 'i1',
                'uinteger1': 'u1', 'integer': '>i4',
                'uinteger': '>u4', 'integer2': '>i2',
                'uinteger2': '>u2', 'integer4': '>i4',
                'uinteger4': '>u4', 'integer8': '>i8',
                'enumerated': 'u1', 'string': 'str', 'bitfield': 'u1'}

        scaling_factor = {}
        scaled_dtype = []
        dtype = [('grh', self.grh_dtype)]

        for key, value in data.items():

            if 'scaling-factor' in value:
                sf_dtype = np.float32
                sf_split = value['scaling-factor'].split('^')
                scaling_factor[key] = np.int(sf_split[0])**np.int(sf_split[1])
            else:
                sf_dtype = conv[value['type']]
                scaling_factor[key] = 1

            length = value['length']

            if length == 1:
                scaled_dtype.append((key, sf_dtype))
                dtype.append((key, conv[value['type']]))
            else:
                scaled_dtype.append((key, sf_dtype, length))
                dtype.append((key, conv[value['type']], length))

        return np.dtype(dtype), np.dtype(scaled_dtype), scaling_factor


def conv_epsl1bszf_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Original dataset.
    metadata : dict
        Metadata.

    Returns
    -------
    data : dict of numpy.ndarray
        Converted dataset.
    """
    skip_fields = ['utc_localisation-days', 'utc_localisation-milliseconds',
                   'degraded_inst_mdr', 'degraded_proc_mdr', 'flagfield_rf1',
                   'flagfield_rf2', 'flagfield_pl', 'flagfield_gen1',
                   'flagfield_gen2']

    gen_fields_lut = {
        'inc_angle_full': ('inc', np.float32, (0, 90), float32_nan),
        'azi_angle_full': ('azi', np.float32, (0, 360), float32_nan),
        'sigma0_full': ('sig', np.float32, (-35, 35), float32_nan),
        'sat_track_azi': ('sat_track_azi', np.float32, (0, 360), float32_nan),
        'beam_number': ('beam_number', np.int8, (1, 6), int8_nan),
        'swath_indicator': ('swath_indicator', np.int8, (0, 1), int8_nan),
        'land_frac': ('land_frac', np.float32, (0, 1), float32_nan),
        'f_usable': ('f_usable', np.int8, (0, 2), int8_nan),
        # 'f_land': ('f_land', np.int8, (0, 1), int8_nan),
        'as_des_pass': ('as_des_pass', np.uint8, (0, 1), uint8_nan),
        'time': ('time', None, (np.datetime64('1900-01-01'),
                                np.datetime64('2100-01-01')),  0),
        'lon': ('lon', np.float32, (-180, 180), float32_nan),
        'lat': ('lat', np.float32, (-90, 90), float32_nan)}

    for var_name in skip_fields:
        data.pop(var_name, None)

    for var_name, (new_name, new_dtype, valid_range, nan_val) in gen_fields_lut.items():
        if new_dtype is None:
            data[new_name] = np.ma.array(data.pop(var_name))
            data[new_name].mask = ((data[new_name] < valid_range[0]) |
                                   (data[new_name] > valid_range[1]))
            data[new_name].set_fill_value(nan_val)
        else:
            data[new_name] = np.ma.array(data.pop(var_name).astype(new_dtype))
            data[new_name].mask = ((data[new_name] < valid_range[0]) |
                                   (data[new_name] > valid_range[1]))
            data[new_name].set_fill_value(nan_val)

    return data


def conv_epsl1bszx_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Original dataset.
    metadata : dict
        Metadata.

    Returns
    -------
    data : dict of numpy.ndarray
        Converted dataset.
    """
    gen_fields_lut = {'inc_angle_trip': ('inc', np.float32, uint_nan),
                      'azi_angle_trip': ('azi', np.float32, int_nan),
                      'sigma0_trip': ('sig', np.float32, long_nan),
                      'kp': ('kp', np.float32, uint_nan),
                      'f_kp': ('kp_quality', np.uint8, uint8_nan)}

    skip_fields = ['flagfield_rf1', 'f_f', 'f_v', 'f_oa', 'f_sa', 'f_tel']

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name, (new_name, new_dtype, nan_val) in gen_fields_lut.items():
        data[new_name] = data.pop(var_name).astype(new_dtype)
        if nan_val is not None:
            data[new_name][data[new_name] == nan_val] = float32_nan

    data['sat_id'] = np.repeat(metadata['sat_id'], data['time'].size)

    return data


def conv_epsl2szx_generic(data, metadata):
    """
    Rename and convert data types of dataset.

    Parameters
    ----------
    data : dict of numpy.ndarray
        Original dataset.
    metadata : dict
        Metadata.

    Returns
    -------
    data : dict of numpy.ndarray
        Converted dataset.
    """
    gen_fields_lut = {
        'inc_angle_trip': ('inc', np.float32, uint_nan),
        'azi_angle_trip': ('azi', np.float32, int_nan),
        'sigma0_trip': ('sig', np.float32, long_nan),
        'soil_moisture': ('sm', np.float32, uint_nan),
        'soil_moisture_error': ('sm_noise', np.float32, uint_nan),
        'mean_surf_soil_moisture': ('sm_mean', np.float32, uint_nan),
        'soil_moisture_sensetivity': ('sm_sens', np.float32, ulong_nan),
        'sigma40': ('sig40', np.float32, long_nan),
        'sigma40_error': ('sig40_noise', np.float32, long_nan),
        'slope40': ('slope40', np.float32, long_nan),
        'slope40_error': ('slope40_noise', np.float32, long_nan),
        'dry_backscatter': ('dry_sig40', np.float32, long_nan),
        'wet_backscatter': ('wet_sig40', np.float32, long_nan),
        'as_des_pass': ('as_des_pass', np.uint8, None),
        'aggregated_quality_flag': ('agg_flag', np.uint8, None),
        'processing_flags': ('proc_flag', np.uint8, None),
        'correction_flags': ('corr_flag', np.uint8, None),
        'snow_cover_probability': ('snow_prob', np.uint8, None),
        'frozen_soil_probability': ('frozen_prob', np.uint8, None),
        'innudation_or_wetland': ('wetland', np.uint8, None),
        'topographical_complexity': ('topo', np.uint8, None),
        'kp': ('kp', np.float32, uint_nan)}

    skip_fields = ['flagfield_rf1', 'f_f', 'f_v', 'f_oa', 'f_sa', 'f_tel']

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name, (new_name, new_dtype, nan_val) in gen_fields_lut.items():
        data[new_name] = data.pop(var_name).astype(new_dtype)
        if nan_val is not None:
            data[new_name][data[new_name] == nan_val] = float32_nan

    data['sat_id'] = np.repeat(metadata['sat_id'], data['time'].size)

    return data


def read_eps_l1b(filename, generic=False, to_xarray=False, full=True,
                 unsafe=False, scale_mdr=True):
    """
    Level 1b reader and data preparation.

    Parameters
    ----------
    filename : str
        ASCAT Level 1b file name in EPS Native format.
    generic : bool, optional
        'True' reading and converting into generic format or
        'False' reading original field names (default: False).
    to_xarray : bool, optional
        'True' return data as xarray.Dataset
        'False' return data as numpy.ndarray (default: False).

    Returns
    -------
    ds : xarray.Dataset, dict of xarray.Dataset
        ASCAT Level 1b data.
    """
    eps_file = read_eps(filename, full=full, unsafe=unsafe,
                        scale_mdr=scale_mdr)

    ptype = eps_file.mphr['PRODUCT_TYPE']
    fmv = int(eps_file.mphr['FORMAT_MAJOR_VERSION'])

    if ptype == 'SZF':

        if fmv == 12:
            data, metadata = read_szf_fmv_12(eps_file)
        else:
            raise RuntimeError("L1b SZF format version not supported.")

        rename_coords = {'longitude_full': 'lon', 'latitude_full': 'lat'}

        for k, v in rename_coords.items():
            data[v] = data.pop(k)

        if generic:
            data = conv_epsl1bszf_generic(data, metadata)

        # 1 Left Fore Antenna, 2 Left Mid Antenna 3 Left Aft Antenna
        # 4 Right Fore Antenna, 5 Right Mid Antenna, 6 Right Aft Antenna
        antennas = ['lf', 'lm', 'la', 'rf', 'rm', 'ra']
        ds = OrderedDict()

        for i, antenna in enumerate(antennas):

            subset = data['beam_number'] == i+1
            metadata['beam_number'] = i+1
            metadata['beam_name'] = antenna

            # convert spacecraft_id to internal sat_id
            sat_id = np.array([4, 3, 5])
            metadata['sat_id'] = sat_id[metadata['spacecraft_id']-1]

            # convert dict to xarray.Dataset or numpy.ndarray
            if to_xarray:
                sub_data = {}
                for var_name in data.keys():

                    if var_name == 'beam_number' and generic:
                        continue

                    if len(data[var_name].shape) == 1:
                        dim = ['obs']
                    elif len(data[var_name].shape) == 2:
                        dim = ['obs', 'echo']

                    sub_data[var_name] = (dim, data[var_name][subset])

                coords = {}
                coords_fields = ['lon', 'lat', 'time']

                for cf in coords_fields:
                    coords[cf] = sub_data.pop(cf)

                ds[antenna] = xr.Dataset(sub_data, coords=coords,
                                         attrs=metadata)
            else:
                # collect dtype info
                dtype = []
                fill_values = {}

                for var_name in data.keys():

                    if var_name == 'beam_number' and generic:
                        continue

                    if len(data[var_name][subset].shape) == 1:
                        dtype.append(
                            (var_name, data[var_name][subset].dtype.str))
                    elif len(data[var_name][subset].shape) > 1:
                        dtype.append((var_name, data[var_name][
                            subset].dtype.str, data[var_name][
                                subset].shape[1:]))

                    fill_values[var_name] = data[var_name].fill_value

                ds[antenna] = np.ma.empty(
                    data['time'][subset].size, dtype=np.dtype(dtype))

                for var_name, v in data.items():
                    if var_name == 'beam_number' and generic:
                        continue
                    ds[antenna][var_name] = v[subset]
                    ds[antenna][var_name].set_fill_value(fill_values[var_name])

    elif ptype in ['SZR', 'SZO']:

        if fmv == 11:
            data, metadata = read_szx_fmv_11(eps_file)
        elif fmv == 12:
            data, metadata = read_szx_fmv_12(eps_file)
        else:
            raise RuntimeError("SZX format version not supported.")

        data['time'] = jd2dt(data.pop('jd'))

        rename_coords = {'longitude': 'lon', 'latitude': 'lat'}

        for k, v in rename_coords.items():
            data[v] = data.pop(k)

        # convert spacecraft_id to internal sat_id
        sat_id = np.array([4, 3, 5])
        metadata['sat_id'] = sat_id[metadata['spacecraft_id']-1]

        # add/rename/remove fields according to generic format
        if generic:
            data = conv_epsl1bszx_generic(data, metadata)

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data.keys():
                if len(data[k].shape) == 1:
                    dim = ['obs']
                elif len(data[k].shape) == 2:
                    dim = ['obs', 'beam']

                data[k] = (dim, data[k])

            coords = {}
            coords_fields = ['lon', 'lat', 'time']
            for cf in coords_fields:
                coords[cf] = data.pop(cf)

            ds = xr.Dataset(data, coords=coords, attrs=metadata)
        else:
            # collect dtype info
            dtype = []
            for var_name in data.keys():
                if len(data[var_name].shape) == 1:
                    dtype.append((var_name, data[var_name].dtype.str))
                elif len(data[var_name].shape) > 1:
                    dtype.append((var_name, data[var_name].dtype.str,
                                  data[var_name].shape[1:]))

            ds = np.empty(data['time'].size, dtype=np.dtype(dtype))
            for k, v in data.items():

                ds[k] = v

    else:
        raise RuntimeError("Format not supported. Product type {:1}"
                           " Format major version: {:2}".format(ptype, fmv))

    return ds


def read_eps_l2(filename, generic=False, to_xarray=False):
    """
    Level 2 reader and data preparation.

    Parameters
    ----------
    filename : str
        ASCAT Level 1b file name in EPS Native format.
    generic : bool, optional
        'True' reading and converting into generic format or
        'False' reading original field names (default: False).
    to_xarray : bool, optional
        'True' return data as xarray.Dataset
        'False' return data as numpy.ndarray (default: False).

    Returns
    -------
    ds : xarray.Dataset, dict of xarray.Dataset
        ASCAT Level 1b data.
    """
    eps_file = read_eps(filename)
    ptype = eps_file.mphr['PRODUCT_TYPE']
    fmv = int(eps_file.mphr['FORMAT_MAJOR_VERSION'])

    if ptype in ['SMR', 'SMO']:

        if fmv == 12:
            data, metadata = read_smx_fmv_12(eps_file)
        else:
            raise RuntimeError("L2 SM format version not supported.")

        data['time'] = jd2dt(data.pop('jd'))

        rename_coords = {'longitude': 'lon', 'latitude': 'lat'}

        for k, v in rename_coords.items():
            data[v] = data.pop(k)

        # convert spacecraft_id to internal sat_id
        sat_id = np.array([4, 3, 5])
        metadata['sat_id'] = sat_id[metadata['spacecraft_id']-1]

        # add/rename/remove fields according to generic format
        if generic:
            data = conv_epsl2szx_generic(data, metadata)

        # convert dict to xarray.Dataset or numpy.ndarray
        if to_xarray:
            for k in data.keys():
                if len(data[k].shape) == 1:
                    dim = ['obs']
                elif len(data[k].shape) == 2:
                    dim = ['obs', 'beam']

                data[k] = (dim, data[k])

            coords = {}
            coords_fields = ['lon', 'lat', 'time']
            for cf in coords_fields:
                coords[cf] = data.pop(cf)

            ds = xr.Dataset(data, coords=coords, attrs=metadata)
        else:
            # collect dtype info
            dtype = []
            for var_name in data.keys():
                if len(data[var_name].shape) == 1:
                    dtype.append((var_name, data[var_name].dtype.str))
                elif len(data[var_name].shape) > 1:
                    dtype.append((var_name, data[var_name].dtype.str,
                                  data[var_name].shape[1:]))

            ds = np.empty(data['time'].size, dtype=np.dtype(dtype))
            for k, v in data.items():

                ds[k] = v
    else:
        raise ValueError("Format not supported. Product type {:1}"
                         " Format major version: {:2}".format(ptype, fmv))

    return ds


def read_eps(filename, mphr_only=False, full=True, unsafe=False,
             scale_mdr=True):
    """
    Read EPS file.

    Parameters
    ----------
    filename : str
        Filename

    Returns
    -------
    prod : EPSProduct
        EPS data.
    """
    zipped = False
    if os.path.splitext(filename)[1] == '.gz':
        zipped = True

    # for zipped files use an unzipped temporary copy
    if zipped:
        with NamedTemporaryFile(delete=False) as tmp_fid:
            with GzipFile(filename) as gz_fid:
                tmp_fid.write(gz_fid.read())
            filename = tmp_fid.name

    # create the eps object with the filename and read it
    prod = EPSProduct(filename)

    if mphr_only:
        mphr = prod.read_mphr()
        prod.mphr = mphr
    else:
        prod.read(full, unsafe, scale_mdr)

    # remove the temporary copy
    if zipped:
        os.remove(filename)

    return prod


def read_szx_fmv_11(eps_file):
    """
    Read SZO/SZR format version 11.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZO/SZR data.
    """
    raw_data = eps_file.scaled_mdr
    raw_unscaled = eps_file.mdr
    mphr = eps_file.mphr

    n_node_per_line = raw_data['LONGITUDE'].shape[1]
    n_lines = raw_data['LONGITUDE'].shape[0]
    n_records = raw_data['LONGITUDE'].size

    data = {}
    metadata = {}
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    ascat_time = shortcdstime2jd(raw_data['UTC_LINE_NODES'].flatten()['day'],
                                 raw_data['UTC_LINE_NODES'].flatten()['time'])
    data['jd'] = ascat_time[idx_nodes]

    metadata['spacecraft_id'] = np.int8(mphr['SPACECRAFT_ID'][-1])
    metadata['orbit_start'] = np.uint32(mphr['ORBIT_START'])

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']

    for f in fields:
        metadata[f] = np.int16(mphr[f.upper()])

    fields = ['sat_track_azi']
    for f in fields:
        data[f] = raw_data[f.upper()].flatten()[idx_nodes]

    fields = [('longitude', long_nan), ('latitude', long_nan),
              ('swath_indicator', byte_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].flatten()
        valid = raw_unscaled[f.upper()].flatten() != nan_val
        data[f][~valid] = nan_val

    fields = [('sigma0_trip', long_nan),
              ('inc_angle_trip', uint_nan),
              ('azi_angle_trip', int_nan),
              ('kp', uint_nan),
              ('f_kp', byte_nan),
              ('f_usable', byte_nan),
              ('f_f', uint_nan),
              ('f_v', uint_nan),
              ('f_oa', uint_nan),
              ('f_sa', uint_nan),
              ('f_tel', uint_nan),
              ('f_land', uint_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].reshape(n_records, 3)
        valid = raw_unscaled[f.upper()].reshape(n_records, 3) != nan_val
        data[f][~valid] = nan_val

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['longitude'] != long_nan,
                          data['longitude'] > 180)
    data['longitude'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi_angle_trip'] != int_nan) & (data['azi_angle_trip'] < 0)
    data['azi_angle_trip'][mask] += 360

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                               n_lines).astype(np.uint8)
    data['line_num'] = idx_nodes.astype(np.uint16)
    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data, metadata


def read_szx_fmv_12(eps_file):
    """
    Read SZO/SZR format version 12.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZO/SZR data.
    """
    raw_data = eps_file.scaled_mdr
    raw_unscaled = eps_file.mdr
    mphr = eps_file.mphr

    n_node_per_line = raw_data['LONGITUDE'].shape[1]
    n_lines = raw_data['LONGITUDE'].shape[0]
    n_records = raw_data['LONGITUDE'].size

    data = {}
    metadata = {}
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    ascat_time = shortcdstime2jd(raw_data['UTC_LINE_NODES'].flatten()['day'],
                                 raw_data['UTC_LINE_NODES'].flatten()['time'])
    data['jd'] = ascat_time[idx_nodes]

    metadata['spacecraft_id'] = np.int8(mphr['SPACECRAFT_ID'][-1])
    metadata['orbit_start'] = np.uint32(mphr['ORBIT_START'])

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']

    for f in fields:
        metadata[f] = np.int16(mphr[f.upper()])

    fields = ['degraded_inst_mdr', 'degraded_proc_mdr', 'sat_track_azi',
              'abs_line_number']

    for f in fields:
        data[f] = raw_data[f.upper()].flatten()[idx_nodes]

    fields = [('longitude', long_nan), ('latitude', long_nan),
              ('swath indicator', byte_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].flatten()
        valid = raw_unscaled[f.upper()].flatten() != nan_val
        data[f][~valid] = nan_val

    fields = [('sigma0_trip', long_nan),
              ('inc_angle_trip', uint_nan),
              ('azi_angle_trip', int_nan),
              ('kp', uint_nan),
              ('num_val_trip', ulong_nan),
              ('f_kp', byte_nan),
              ('f_usable', byte_nan),
              ('f_f', uint_nan),
              ('f_v', uint_nan),
              ('f_oa', uint_nan),
              ('f_sa', uint_nan),
              ('f_tel', uint_nan),
              ('f_ref', uint_nan),
              ('f_land', uint_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].reshape(n_records, 3)
        valid = raw_unscaled[f.upper()].reshape(n_records, 3) != nan_val
        data[f][~valid] = nan_val

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['longitude'] != long_nan,
                          data['longitude'] > 180)
    data['longitude'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi_angle_trip'] != int_nan) & (data['azi_angle_trip'] < 0)
    data['azi_angle_trip'][mask] += 360

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                               n_lines).astype(np.uint8)

    data['line_num'] = idx_nodes.astype(np.uint16)

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    data['swath_indicator'] = data.pop('swath indicator')

    return data, metadata


def read_szf_fmv_12(eps_file):
    """
    Read SZF format version 12.

    beam_num
    - 1 Left Fore Antenna
    - 2 Left Mid Antenna
    - 3 Left Aft Antenna
    - 4 Right Fore Antenna
    - 5 Right Mid Antenna
    - 6 Right Aft Antenna

    as_des_pass
    - 0 Ascending
    - 1 Descending

    swath_indicator
    - 0 Left
    - 1 Right

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZF data.
    """
    data = {}
    metadata = {}

    n_lines = eps_file.mdr_counter
    n_node_per_line = eps_file.mdr['LONGITUDE_FULL'].shape[1]
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    # extract metadata
    metadata['spacecraft_id'] = np.int8(eps_file.mphr['SPACECRAFT_ID'][-1])
    metadata['orbit_start'] = np.uint32(eps_file.mphr['ORBIT_START'])

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']
    for f in fields:
        metadata[f] = np.int16(eps_file.mphr[f.upper()])

    # extract time
    dt = np.datetime64('2000-01-01') + eps_file.mdr[
        'UTC_LOCALISATION']['day'].astype('timedelta64[D]') + eps_file.mdr[
            'UTC_LOCALISATION']['time'].astype('timedelta64[ms]')
    data['time'] = dt[idx_nodes]

    fields = ['degraded_inst_mdr', 'degraded_proc_mdr', 'sat_track_azi',
              'beam_number', 'flagfield_rf1', 'flagfield_rf2',
              'flagfield_pl', 'flagfield_gen1']

    # extract data
    for f in fields:
        if eps_file.mdr_sfactor[f.upper()] == 1:
            data[f] = eps_file.mdr[f.upper()].flatten()[idx_nodes]
        else:
            data[f] = (eps_file.mdr[f.upper()].flatten() *
                       1./eps_file.mdr_sfactor[f.upper()])[idx_nodes]

    data['swath_indicator'] = (data[
        'beam_number'].flatten() > 3).astype(np.uint8)
    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    fields = [('longitude_full', long_nan),
              ('latitude_full', long_nan),
              ('sigma0_full', long_nan),
              ('inc_angle_full', uint_nan),
              ('azi_angle_full', int_nan),
              ('land_frac', uint_nan),
              ('flagfield_gen2', byte_nan)]

    for f, nan_val in fields:
        data[f] = eps_file.mdr[f.upper()].flatten()
        invalid = eps_file.mdr[f.upper()].flatten() == nan_val

        if eps_file.mdr_sfactor[f.upper()] != 1:
            data[f] = data[f] * 1./eps_file.mdr_sfactor[f.upper()]

        data[f][invalid] = nan_val

    # modify longitudes from (0, 360) to (-180, 180)
    mask = np.logical_and(data['longitude_full'] != long_nan,
                          data['longitude_full'] > 180)
    data['longitude_full'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    idx = (data['azi_angle_full'] != int_nan) & (data['azi_angle_full'] < 0)
    data['azi_angle_full'][idx] += 360

    # set flags
    data['f_usable'] = set_flags(data)

    return data, metadata


def read_smx_fmv_12(eps_file):
    """
    Read SMO/SMR format version 12.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SMO/SMR data.
    """
    raw_data = eps_file.scaled_mdr
    raw_unscaled = eps_file.mdr

    n_node_per_line = raw_data['LONGITUDE'].shape[1]
    n_lines = raw_data['LONGITUDE'].shape[0]
    n_records = eps_file.mdr_counter * n_node_per_line
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(n_node_per_line)

    data = {}
    metadata = {}

    metadata['spacecraft_id'] = np.int8(eps_file.mphr['SPACECRAFT_ID'][-1])
    metadata['orbit_start'] = np.uint32(eps_file.mphr['ORBIT_START'])

    ascat_time = shortcdstime2jd(raw_data['UTC_LINE_NODES'].flatten()['day'],
                                 raw_data['UTC_LINE_NODES'].flatten()['time'])
    data['jd'] = ascat_time[idx_nodes]

    fields = [('sigma0_trip', long_nan),
              ('inc_angle_trip', uint_nan),
              ('azi_angle_trip', int_nan),
              ('kp', uint_nan),
              ('f_land', uint_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].reshape(n_records, 3)
        valid = raw_unscaled[f.upper()].reshape(n_records, 3) != nan_val
        data[f][~valid] = nan_val

    fields = ['sat_track_azi', 'abs_line_number']
    for f in fields:
        data[f] = raw_data[f.upper()].flatten()[idx_nodes]

    fields = [('longitude', long_nan, long_nan),
              ('latitude', long_nan, long_nan),
              ('swath_indicator', byte_nan, byte_nan),
              ('soil_moisture', uint_nan, uint_nan),
              ('soil_moisture_error', uint_nan, uint_nan),
              ('sigma40', long_nan, long_nan),
              ('sigma40_error', long_nan, long_nan),
              ('slope40', long_nan, long_nan),
              ('slope40_error', long_nan, long_nan),
              ('dry_backscatter', long_nan, long_nan),
              ('wet_backscatter', long_nan, long_nan),
              ('mean_surf_soil_moisture', uint_nan, uint_nan),
              ('soil_moisture_sensetivity', ulong_nan, float32_nan),
              ('correction_flags', uint8_nan, uint8_nan),
              ('processing_flags', uint8_nan, uint8_nan),
              ('aggregated_quality_flag', uint8_nan, uint8_nan),
              ('snow_cover_probability', uint8_nan, uint8_nan),
              ('frozen_soil_probability', uint8_nan, uint8_nan),
              ('innudation_or_wetland', uint8_nan, uint8_nan),
              ('topographical_complexity', uint8_nan, uint8_nan)]

    for f, nan_val, new_nan_val in fields:
        data[f] = raw_data[f.upper()].flatten()
        valid = raw_unscaled[f.upper()].flatten() != nan_val
        data[f][~valid] = new_nan_val

    # sat_track_azi (uint)
    data['as_des_pass'] = \
        np.array(raw_data['SAT_TRACK_AZI'].flatten()[idx_nodes] < 270)

    # modify longitudes from [0,360] to [-180,180]
    mask = np.logical_and(data['longitude'] != long_nan,
                          data['longitude'] > 180)
    data['longitude'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi_angle_trip'] != int_nan) & (data['azi_angle_trip'] < 0)
    data['azi_angle_trip'][mask] += 360

    fields = ['param_db_version', 'warp_nrt_version']
    for f in fields:
        data[f] = raw_data['PARAM_DB_VERSION'].flatten()[idx_nodes]

    metadata['spacecraft_id'] = int(eps_file.mphr['SPACECRAFT_ID'][2])

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1), n_lines)

    data['line_num'] = idx_nodes

    return data, metadata


def shortcdstime2jd(days, milliseconds):
    """
    Convert cds time to julian date.

    Parameters
    ----------
    days : int
        Days since 2000-01-01
    milliseconds : int
        Milliseconds.

    Returns
    -------
    jd : float
        Julian date.
    """
    offset = days + (milliseconds / 1000.) / (24. * 60. * 60.)
    return julian_epoch + offset


def set_flags(data):
    """
    Compute summary flag for each measurement with a value of 0, 1 or 2
    indicating nominal, slightly degraded or severely degraded data.

    The format of ASCAT products is defined by
    "EPS programme generic product format specification" (EPS.GGS.SPE.96167)
    and "ASCAT level 1 product format specification" (EPS.MIS.SPE.97233).

    bit name      category   description
    ------------------------------------

    flagfield_rf1
    0  fnoise     amber     noise missing, interpolated noise value used instead
    1  fpgp       amber     degraded power gain product
    2  vpgp       red       very degraded power gain product
    3  fhrx       amber     degraded filter shape
    4  vhrx       red       very degraded filter shape

    flagfield_rf2
    0  pgp_ool    red       power gain product is outside limits
    1  noise_ool  red       measured noise value is outside limits

    flagfield_pl
    0  forb       red       orbit height is outside limits
    1  fatt       red       no yaw steering
    2  fcfg       red       unexpected instrument configuration
    3  fman       red       satellite maneuver
    4  fosv       warning   osv file missing (fman may be incorrect)

    flagfield_gen1
    0  ftel       warning   telemetry missing (ftool may be incorrect)
    1  ftool      red       telemetry out of limits

    flagfield_gen2
    0  fsol   amber     possible interference from solar array
    1  fland  warning   lat/long position is over land
    2  fgeo   red       geolocation algorithm failed

    Each flag has belongs to a particular category which indicates the impact
    on data quality. Flags in the "amber" category indicate that the data is
    slightly degraded but still usable. Flags in the "red" category indicate
    that the data is severely degraded and should be discarded or
    used with caution.

    A simple algorithm for calculating a single summary flag with a value of
    0, 1 or 2 indicating nominal, slightly degraded or severely degraded is

    function calc_status( flags )
        status = 0
        if any amber flags are set then status = 1
        if any red flags are set then status = 2
    return status

    Parameters
    ----------
    data : numpy.ndarray
        SZF data.

    Returns
    -------
    f_usable : numpy.ndarray
        Flag indicating nominal (0), slightly degraded (1) or
        severely degraded(2).
    """
    flag_status_bit = {'flagfield_rf1': np.array([1, 1, 2, 1, 2, 0, 0, 0]),
                       'flagfield_rf2': np.array([2, 2, 0, 0, 0, 0, 0, 0]),
                       'flagfield_pl':  np.array([2, 2, 2, 2, 0, 0, 0, 0]),
                       'flagfield_gen1': np.array([0, 2, 0, 0, 0, 0, 0, 0]),
                       'flagfield_gen2': np.array([1, 0, 2, 0, 0, 0, 0, 0])}

    f_usable = np.zeros(data['flagfield_rf1'].size, dtype=np.uint8)

    for flagfield, bitmask in flag_status_bit.items():
        subset = np.nonzero(data[flagfield])[0]

        if subset.size > 0:
            unpacked_bits = np.fliplr(np.unpackbits(
                data[flagfield][subset]).reshape(-1, 8).astype(np.bool))

            flag = np.ma.array(
                np.tile(bitmask, unpacked_bits.shape[0]).reshape(-1, 8),
                mask=~unpacked_bits, fill_value=0)

            f_usable[subset] = np.max(np.vstack(
                (f_usable[subset], flag.filled().max(axis=1))), axis=0)

    return f_usable
