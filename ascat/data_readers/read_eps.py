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
Readers for lvl1b and lvl2 data in eps format.
"""

import os
import fnmatch
import lxml.etree as etree
from tempfile import NamedTemporaryFile
from gzip import GzipFile
from collections import OrderedDict
from pygeobase.io_base import ImageBase
from pygeobase.object_base import Image

import numpy as np
import datetime as dt
import matplotlib.dates as mpl_dates
import ascat.data_readers.templates as templates


short_cds_time = np.dtype([('day', np.uint16), ('time', np.uint32)])

long_cds_time = np.dtype([('day', np.uint16), ('ms', np.uint32),
                          ('mms', np.uint16)])

long_nan = -2 ** 31
ulong_nan = 2 ** 32 - 1
int_nan = -2 ** 15
uint_nan = 2 ** 16 - 1
byte_nan = -2 ** 7


class AscatL1bEPSImage(ImageBase):
    def __init__(self, *args, **kwargs):
        """
        Initialization of i/o object.
        """
        super(AscatL1bEPSImage, self).__init__(*args, **kwargs)

    def read(self, timestamp=None, file_format=None, **kwargs):
        longitude, latitude, data, metadata = read_eps_l1b(
            self.filename)

        return Image(longitude, latitude, data, metadata,
                     timestamp, timekey='jd')

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass

class AscatL2EPSImage(ImageBase):
    def __init__(self, *args, **kwargs):
        """
        Initialization of i/o object.
        """
        super(AscatL2EPSImage, self).__init__(*args, **kwargs)

    def read(self, timestamp=None, file_format=None, **kwargs):
        longitude, latitude, data, metadata = read_eps_l2(
            self.filename)

        return Image(longitude, latitude, data, metadata,
                     timestamp, timekey='jd')

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class EPSProduct(object):
    """
    Class for reading EPS products.
    """

    def __init__(self, filename):
        self.filename = filename
        self.fid = None
        self.grh = None
        self.mphr = None
        self.sphr = None
        self.ipr = None
        self.geadr = None
        self.giadr_archive = None
        self.veadr = None
        self.viadr = None
        self.viadr_scaled = None
        self.viadr_grid = None
        self.viadr_grid_scaled = None
        self.dummy_mdr = None
        self.mdr = None
        self.eor = 0
        self.bor = 0
        self.mdr_counter = 0
        self.filesize = 0
        self.xml_file = None
        self.xml_doc = None
        self.mdr_template = None
        self.scaled_mdr = None
        self.scaled_template = None
        self.sfactor = None

    def read_product(self):
        """
        Read complete file and create numpy arrays from raw byte string data.
        """
        # open file in read-binary mode
        self.fid = open(self.filename, 'rb')
        self.filesize = os.path.getsize(self.filename)
        self.eor = self.fid.tell()

        # loop as long as the current position hasn't reached the end of file
        while self.eor < self.filesize:

            # remember beginning of the record
            self.bor = self.fid.tell()

            # read grh of current record (Generic Record Header)
            self.grh = self._read_record(grh_record())
            record_size = self.grh[0]['record_size']
            record_class = self.grh[0]['record_class']
            record_subclass = self.grh[0]['record_subclass']

            # mphr (Main Product Header Reader)
            if record_class == 1:
                self._read_mphr()

                # find the xml file corresponding to the format version
                self.xml_file = self._get_eps_xml()
                self.xml_doc = etree.parse(self.xml_file)
                self.mdr_template, self.scaled_template, self.sfactor = self._read_xml_mdr()

            # sphr (Secondary Product Header Record)
            elif record_class == 2:
                self._read_sphr()

            # ipr (Internal Pointer Record)
            elif record_class == 3:
                ipr_element = self._read_record(ipr_record())
                if self.ipr is None:
                    self.ipr = {}
                    self.ipr['data'] = [ipr_element]
                    self.ipr['grh'] = [self.grh]
                else:
                    self.ipr['data'].append(ipr_element)
                    self.ipr['grh'].append(self.grh)

            # geadr (Global External Auxiliary Data Record)
            elif record_class == 4:
                geadr_element = self._read_pointer()
                if self.geadr is None:
                    self.geadr = {}
                    self.geadr['data'] = [geadr_element]
                    self.geadr['grh'] = [self.grh]
                else:
                    self.geadr['data'].append(geadr_element)
                    self.geadr['grh'].append(self.grh)

            # veadr (Variable External Auxiliary Data Record)
            elif record_class == 6:
                veadr_element = self._read_pointer()
                if self.veadr is None:
                    self.veadr = {}
                    self.veadr['data'] = [veadr_element]
                    self.veadr['grh'] = [self.grh]
                else:
                    self.veadr['data'].append(veadr_element)
                    self.veadr['grh'].append(self.grh)

            # viadr (Variable Internal Auxiliary Data Record)
            elif record_class == 7:
                template, scaled_template, sfactor = self._read_xml_viadr(
                    record_subclass)
                viadr_element = self._read_record(template)
                viadr_element_sc = self._scaling(viadr_element,
                                                 scaled_template, sfactor)

                # store viadr_grid separately
                if record_subclass == 8:
                    if self.viadr_grid is None:
                        self.viadr_grid = [viadr_element]
                        self.viadr_grid_scaled = [viadr_element_sc]
                    else:
                        self.viadr_grid.append(viadr_element)
                        self.viadr_grid_scaled.append(viadr_element_sc)
                else:
                    if self.viadr is None:
                        self.viadr = {}
                        self.viadr_scaled = {}
                        self.viadr['data'] = [viadr_element]
                        self.viadr['grh'] = [self.grh]
                        self.viadr_scaled['data'] = [viadr_element_sc]
                        self.viadr_scaled['grh'] = [self.grh]
                    else:
                        self.viadr['data'].append(viadr_element)
                        self.viadr['grh'].append(self.grh)
                        self.viadr_scaled['data'].append(viadr_element_sc)
                        self.viadr_scaled['grh'].append(self.grh)

            # mdr (Measurement Data Record)
            elif record_class == 8:
                if self.grh[0]['instrument_group'] == 13:
                    self.dummy_mdr = self._read_record(self.mdr_template)
                else:
                    mdr_element = self._read_record(self.mdr_template)
                    if self.mdr is None:
                        self.mdr = [mdr_element]
                    else:
                        self.mdr.append(mdr_element)
                    self.mdr_counter += 1

            else:
                raise RuntimeError("Record class not found.")

            # return pointer to the beginning of the record
            self.fid.seek(self.bor)
            self.fid.seek(record_size, 1)

            # determine number of bytes read
            # end of record
            self.eor = self.fid.tell()

        self.fid.close()

        self.mdr = np.hstack(self.mdr)
        self.scaled_mdr = self._scaling(self.mdr, self.scaled_template,
                                        self.sfactor)

    def _append_list(self, new_element):
        return

    def _scaling(self, unscaled_data, scaled_template, sfactor):
        '''
        Scale the data
        '''
        scaled_data = np.zeros_like(unscaled_data, dtype=scaled_template)

        for name, sf in zip(unscaled_data.dtype.names, sfactor):
            if sf != 1:
                scaled_data[name] = unscaled_data[name] / sf
            else:
                scaled_data[name] = unscaled_data[name]

        return scaled_data

    def _read_record(self, dtype, count=1):
        """
        Read record
        """
        record = np.fromfile(self.fid, dtype=dtype, count=count)
        return record.newbyteorder('B')

    def _read_mphr(self):
        """
        Read Main Product Header (MPHR).
        """
        mphr = self.fid.read(self.grh[0]['record_size'] - self.grh[0].itemsize)
        self.mphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in mphr.split('\n')[:-1])

    def _read_sphr(self):
        """
        Read Special Product Header (SPHR).
        """
        sphr = self.fid.read(self.grh[0]['record_size'] - self.grh[0].itemsize)
        self.sphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in sphr.split('\n')[:-1])

    def _read_pointer(self, count=1):
        """
        Read pointer record.
        """
        dtype = np.dtype([('aux_data_pointer', np.ubyte, 100)])
        record = np.fromfile(self.fid, dtype=dtype, count=count)
        return record.newbyteorder('B')

    def _get_eps_xml(self):
        '''
        Find the corresponding eps xml file.
        '''
        format_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                   'formats')

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
                if longtime_flag == False:
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
                'boolean': np.uint8, 'integer1': np.int8,
                'uinteger1': np.uint8, 'integer': np.int32,
                'uinteger': np.uint32, 'integer2': np.int16,
                'uinteger2': np.uint16, 'integer4': np.int32,
                'uinteger4': np.uint32, 'integer8': np.int64,
                'enumerated': np.uint8, 'string': 'str', 'bitfield': np.uint8}

        scaling_factor = []
        scaled_dtype = []
        dtype = []

        for key, value in data.items():

            if 'scaling-factor' in value:
                sf_dtype = np.float32
                sf = eval(value['scaling-factor'].replace('^', '**'))
            else:
                sf_dtype = conv[value['type']]
                sf = 1

            scaling_factor.append(sf)
            scaled_dtype.append((key, sf_dtype, value['length']))
            dtype.append((key, conv[value['type']], value['length']))

        return np.dtype(dtype), np.dtype(scaled_dtype), np.array(
            scaling_factor,
            dtype=np.float32)

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

            # append the length if it isn't the special case of type bitfield or time
            try:
                var_len = child_items.pop('length')
                if bitfield_flag == False:
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

                    if bitfield_flag == True:
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
                'boolean': np.uint8, 'integer1': np.int8,
                'uinteger1': np.uint8, 'integer': np.int32,
                'uinteger': np.uint32, 'integer2': np.int16,
                'uinteger2': np.uint16, 'integer4': np.int32,
                'uinteger4': np.uint32, 'integer8': np.int64,
                'enumerated': np.uint8, 'string': 'str', 'bitfield': np.uint8}

        scaling_factor = []
        scaled_dtype = []
        dtype = []

        for key, value in data.items():

            if 'scaling-factor' in value:
                sf_dtype = np.float32
                sf = eval(value['scaling-factor'].replace('^', '**'))
            else:
                sf_dtype = conv[value['type']]
                sf = 1

            scaling_factor.append(sf)
            scaled_dtype.append((key, sf_dtype, value['length']))
            dtype.append((key, conv[value['type']], value['length']))

        return np.dtype(dtype), np.dtype(scaled_dtype), np.array(
            scaling_factor,
            dtype=np.float32)


def grh_record():
    """
    Generic record header.
    """
    record_dtype = np.dtype([('record_class', np.ubyte),
                             ('instrument_group', np.ubyte),
                             ('record_subclass', np.ubyte),
                             ('record_subclass_version', np.ubyte),
                             ('record_size', np.uint32),
                             ('record_start_time', short_cds_time),
                             ('record_stop_time', short_cds_time)])

    return record_dtype


def ipr_record():
    """
    ipr template.
    """
    record_dtype = np.dtype([('target_record_class', np.ubyte),
                             ('target_instrument_group', np.ubyte),
                             ('target_record_subclass', np.ubyte),
                             ('target_record_offset', np.uint32)])
    return record_dtype


def read_eps_l1b(filename):
    """
    Use of correct lvl1b reader and data preparation.
    """
    data = {}
    metadata = {}
    eps_file = read_eps(filename)
    ptype = eps_file.mphr['PRODUCT_TYPE']
    fmv = int(eps_file.mphr['FORMAT_MAJOR_VERSION'])

    if ptype == 'SZF':
        if fmv == 12:
            raw_data, orbit_grid = read_szf_fmv_12(eps_file)
        else:
            raise RuntimeError("SZF format version not supported.")

        fields = ['as_des_pass', 'swath_indicator', 'beam_number',
                  'azi', 'inc', 'sig', 'f_usable', 'f_land', 'jd']
        for field in fields:
            data[field] = raw_data[field]

        fields = ['spacecraft_id', 'sat_track_azi',
                  'processor_major_version', 'processor_minor_version',
                  'format_major_version', 'format_minor_version',
                  'degraded_inst_mdr', 'degraded_proc_mdr',
                  'flagfield_rf1', 'flagfield_rf2', 'flagfield_pl',
                  'flagfield_gen1', 'flagfield_gen2', 'land_frac',
                  'f_usable', 'f_land']

        for field in fields:
            metadata[field] = raw_data[field]

    elif (ptype == 'SZR') or (ptype == 'SZO'):
        if fmv == 11:
            raw_data = read_szx_fmv_11(eps_file)
        elif fmv == 12:
            raw_data = read_szx_fmv_12(eps_file)
        else:
            raise RuntimeError("SZX format version not supported.")

        beams = ['f', 'm', 'a']

        fields = ['as_des_pass', 'swath_indicator', 'node_num',
                  'sat_track_azi', 'line_num', 'jd',
                  'spacecraft_id', 'abs_orbit_nr']
        for field in fields:
            data[field] = raw_data[field]

        fields = ['azi', 'inc', 'sig', 'kp', 'f_land',
                  'f_usable', 'f_kp', 'f_f', 'f_v', 'f_oa',
                  'f_sa', 'f_tel', 'f_ref', 'num_val']
        for field in fields:
            for i, beam in enumerate(beams):
                data[field + beam] = raw_data[field][:, i]

        fields = ['processor_major_version',
                  'processor_minor_version', 'format_major_version',
                  'format_minor_version']
        for field in fields:
            metadata[field] = raw_data[field]
    else:
        raise ValueError("Format not supported. Product type {:1}"
                         " Format major version: {:2}".format(ptype, fmv))

    return raw_data['lon'], raw_data['lat'], data, metadata


def read_eps_l2(filename):
    """
    Use of correct lvl2 reader and data preparation.
    """
    data = {}
    metadata = {}
    eps_file = read_eps(filename)
    ptype = eps_file.mphr['PRODUCT_TYPE']
    fmv = int(eps_file.mphr['FORMAT_MAJOR_VERSION'])

    if (ptype == 'SMR') or (ptype == 'SMO'):
        if fmv == 12:
            raw_data = read_smx_fmv_12(eps_file)
        else:
            raise RuntimeError("SMX format version not supported.")

        fields = ['jd', 'as_des_pass', 'swath_indicator', 'node_num', 'ssm',
                  'ssm_noise', 'norm_sigma', 'norm_sigma_noise', 'slope',
                  'slope_noise', 'dry_ref', 'wet_ref', 'mean_ssm',
                  'ssm_sens', 'correction_flag', 'processing_flag',
                  'aggregated_flag', 'snow', 'frozen', 'wetland', 'topo']
        for field in fields:
            data[field] = raw_data[field]

        fields = ['azi', 'inc', 'sig', 'kp', 'f_land']
        beams = ['f', 'm', 'a']
        for field in fields:
            for i, beam in enumerate(beams):
                data[field + beam] = raw_data[field][:, i]

        fields = ['spacecraft_id', 'warp_nrt_version',
                  'param_db_version', ]

        for field in fields:
            metadata[field] = raw_data[field]

    else:
        raise ValueError("Format not supported. Product type {:1}"
                         " Format major version: {:2}".format(ptype, fmv))

    return raw_data['lon'], raw_data['lat'], data, metadata


def read_eps(filename):
    """
    Read EPS file.
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
    prod.read_product()

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

    template = templates.template_SZX__002()
    n_node_per_line = raw_data['LONGITUDE'].shape[1]
    n_lines = raw_data['LONGITUDE'].shape[0]
    n_records = raw_data['LONGITUDE'].size
    data = np.repeat(template, n_records)
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    data['jd'] = mpl_dates.num2julian(shortcdstime2dtordinal(
        raw_data['UTC_LINE_NODES'].flatten()['day'],
        raw_data['UTC_LINE_NODES'].flatten()['time']))[idx_nodes]

    data['spacecraft_id'] = np.int8(mphr['SPACECRAFT_ID'][-1])
    data['abs_orbit_nr'] = np.uint32(mphr['ORBIT_START'])

    fields = [('processor_major_version', 'PROCESSOR_MAJOR_VERSION'),
              ('processor_minor_version', 'PROCESSOR_MINOR_VERSION'),
              ('format_major_version', 'FORMAT_MAJOR_VERSION'),
              ('format_minor_version', 'FORMAT_MINOR_VERSION')]
    for field in fields:
        data[field[0]] = np.int16(mphr[field[1]])

    fields = [('sat_track_azi', 'SAT_TRACK_AZI')]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()[idx_nodes]

    fields = [('lon', 'LONGITUDE', long_nan),
              ('lat', 'LATITUDE', long_nan),
              ('swath_indicator', 'SWATH_INDICATOR', byte_nan)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        # valid = data[field[0]] != field[2]
        valid = raw_unscaled[field[1]].flatten() != field[2]
        data[field[0]][valid == False] = field[2]

    fields = [('sig', 'SIGMA0_TRIP', long_nan),
              ('inc', 'INC_ANGLE_TRIP', uint_nan),
              ('azi', 'AZI_ANGLE_TRIP', int_nan),
              ('kp', 'KP', uint_nan),
              ('f_kp', 'F_KP', byte_nan),
              ('f_usable', 'F_USABLE', byte_nan),
              ('f_f', 'F_F', uint_nan),
              ('f_v', 'F_V', uint_nan),
              ('f_oa', 'F_OA', uint_nan),
              ('f_sa', 'F_SA', uint_nan),
              ('f_tel', 'F_TEL', uint_nan),
              ('f_land', 'F_LAND', uint_nan)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].reshape(n_records, 3)
        valid = raw_unscaled[field[1]].reshape(n_records, 3) != field[2]
        data[field[0]][valid == False] = field[2]

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                               n_lines)

    data['line_num'] = idx_nodes

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data


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

    template = templates.template_SZX__002()
    n_node_per_line = raw_data['LONGITUDE'].shape[1]
    n_lines = raw_data['LONGITUDE'].shape[0]
    n_records = raw_data['LONGITUDE'].size
    data = np.repeat(template, n_records)
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    data['jd'] = mpl_dates.num2julian(shortcdstime2dtordinal(
        raw_data['UTC_LINE_NODES'].flatten()['day'],
        raw_data['UTC_LINE_NODES'].flatten()['time']))[idx_nodes]

    data['spacecraft_id'] = np.int8(mphr['SPACECRAFT_ID'][-1])
    data['abs_orbit_nr'] = np.uint32(mphr['ORBIT_START'])

    fields = [('processor_major_version', 'PROCESSOR_MAJOR_VERSION'),
              ('processor_minor_version', 'PROCESSOR_MINOR_VERSION'),
              ('format_major_version', 'FORMAT_MAJOR_VERSION'),
              ('format_minor_version', 'FORMAT_MINOR_VERSION')]
    for field in fields:
        data[field[0]] = np.int16(mphr[field[1]])

    fields = [('degraded_inst_mdr', 'DEGRADED_INST_MDR'),
              ('degraded_proc_mdr', 'DEGRADED_PROC_MDR'),
              ('sat_track_azi', 'SAT_TRACK_AZI')]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()[idx_nodes]

    fields = [('lon', 'LONGITUDE', long_nan),
              ('lat', 'LATITUDE', long_nan),
              ('swath_indicator', 'SWATH INDICATOR', byte_nan)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        valid = raw_unscaled[field[1]].flatten() != field[2]
        data[field[0]][valid == False] = field[2]

    fields = [('sig', 'SIGMA0_TRIP', long_nan),
              ('inc', 'INC_ANGLE_TRIP', uint_nan),
              ('azi', 'AZI_ANGLE_TRIP', int_nan),
              ('kp', 'KP', uint_nan),
              ('num_val', 'NUM_VAL_TRIP', ulong_nan),
              ('f_kp', 'F_KP', byte_nan),
              ('f_usable', 'F_USABLE', byte_nan),
              ('f_f', 'F_F', uint_nan),
              ('f_v', 'F_V', uint_nan),
              ('f_oa', 'F_OA', uint_nan),
              ('f_sa', 'F_SA', uint_nan),
              ('f_tel', 'F_TEL', uint_nan),
              ('f_ref', 'F_REF', uint_nan),
              ('f_land', 'F_LAND', uint_nan)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].reshape(n_records, 3)
        # valid = data[field[0]] != field[2]
        valid = raw_unscaled[field[1]].reshape(n_records, 3) != field[2]
        data[field[0]][valid == False] = field[2]

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                               n_lines)

    data['line_num'] = idx_nodes

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data


def read_szf_fmv_12(eps_file):
    """
    Read SZF format version 12.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZF data.
    orbit_gri : numpy.ndarray
        6.25km orbit lat/lon grid.
    """
    raw_data = eps_file.scaled_mdr
    mphr = eps_file.mphr

    template = templates.template_SZF__001()
    n_node_per_line = raw_data['LONGITUDE_FULL'].shape[1]
    n_lines = eps_file.mdr_counter
    n_records = raw_data['LONGITUDE_FULL'].size
    data = np.repeat(template, n_records)
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    data['jd'] = mpl_dates.num2julian(shortcdstime2dtordinal(
        raw_data['UTC_LOCALISATION'].flatten()['day'],
        raw_data['UTC_LOCALISATION'].flatten()['time']))[idx_nodes]

    data['spacecraft_id'] = np.int8(mphr['SPACECRAFT_ID'][-1])

    fields = [('processor_major_version', 'PROCESSOR_MAJOR_VERSION'),
              ('processor_minor_version', 'PROCESSOR_MINOR_VERSION'),
              ('format_major_version', 'FORMAT_MAJOR_VERSION'),
              ('format_minor_version', 'FORMAT_MINOR_VERSION')]
    for field in fields:
        data[field[0]] = np.int16(mphr[field[1]])

    fields = [('degraded_inst_mdr', 'DEGRADED_INST_MDR'),
              ('degraded_proc_mdr', 'DEGRADED_PROC_MDR'),
              ('sat_track_azi', 'SAT_TRACK_AZI'),
              ('as_des_pass', 'AS_DES_PASS'),
              ('beam_number', 'BEAM_NUMBER'),
              ('flagfield_rf1', 'FLAGFIELD_RF1'),
              ('flagfield_rf2', 'FLAGFIELD_RF2'),
              ('flagfield_pl', 'FLAGFIELD_PL'),
              ('flagfield_gen1', 'FLAGFIELD_GEN1')]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()[idx_nodes]

    data['swath_indicator'] = np.int8(data['beam_number'].flatten() > 3)

    fields = [('lon', 'LONGITUDE_FULL', long_nan),
              ('lat', 'LATITUDE_FULL', long_nan),
              ('sig', 'SIGMA0_FULL', long_nan),
              ('inc', 'INC_ANGLE_FULL', uint_nan),
              ('azi', 'AZI_ANGLE_FULL', int_nan),
              ('land_frac', 'LAND_FRAC', uint_nan),
              ('flagfield_gen2', 'FLAGFIELD_GEN2', byte_nan)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid]

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    grid_nodes_per_line = 2 * 81

    viadr_grid = np.concatenate(eps_file.viadr_grid)
    orbit_grid = np.zeros(viadr_grid.size * grid_nodes_per_line,
                          dtype=np.dtype([('lon', np.float32),
                                          ('lat', np.float32),
                                          ('node_num', np.int16),
                                          ('line_num', np.int32)]))

    for pos_all in range(orbit_grid['lon'].size):
        line = pos_all / grid_nodes_per_line
        pos_small = pos_all % 81
        if (pos_all % grid_nodes_per_line <= 80):
            # left swath
            orbit_grid['lon'][pos_all] = viadr_grid[
                'LONGITUDE_LEFT'][line][80 - pos_small]
            orbit_grid['lat'][pos_all] = viadr_grid[
                'LATITUDE_LEFT'][line][80 - pos_small]
        else:
            # right swath
            orbit_grid['lon'][pos_all] = viadr_grid[
                'LONGITUDE_RIGHT'][line][pos_small]
            orbit_grid['lat'][pos_all] = viadr_grid[
                'LATITUDE_RIGHT'][line][pos_small]

    orbit_grid['node_num'] = np.tile((np.arange(grid_nodes_per_line) + 1),
                                     viadr_grid.size)

    lines = np.arange(0, viadr_grid.size * 2, 2)
    orbit_grid['line_num'] = np.repeat(lines, grid_nodes_per_line)

    fields = ['lon', 'lat']
    for field in fields:
        mask = orbit_grid[field] != long_nan
        orbit_grid[field] = orbit_grid[field] * 1e-6

    mask = (orbit_grid['lon'] != long_nan) & (orbit_grid['lon'] > 180)
    orbit_grid['lon'][mask] += -360.

    set_flags(data)

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data, orbit_grid


def set_flags(data):
    """
    Compute summary flag for each measurement with a value of 0, 1 or 2
    indicating nominal, slightly degraded or severely degraded data.

    Parameters
    ----------
    data : numpy.ndarray
        SZF data.
    """

    # category:status = 'red': 2, 'amber': 1, 'warning': 0
    flag_status_bit = {'flagfield_rf1': {'2': [2, 4],
                                         '1': [0, 1, 3]},

                       'flagfield_rf2': {'2': [0, 1]},

                       'flagfield_pl': {'2': [0, 1, 2, 3],
                                        '0': [4]},

                       'flagfield_gen1': {'2': [1],
                                          '0': [0]},

                       'flagfield_gen2': {'2': [2],
                                          '1': [0],
                                          '0': [1]}
                       }

    for flagfield in flag_status_bit.keys():
        # get flag data in binary format to get flags
        unpacked_bits = np.unpackbits(data[flagfield])

        # find indizes where a flag is set
        set_bits = np.where(unpacked_bits == 1)[0]
        if (set_bits.size != 0):
            pos_8 = 7 - (set_bits % 8)

            for category in sorted(flag_status_bit[flagfield].keys()):
                if (int(category) == 0) and (flagfield != 'flagfield_gen2'):
                    continue

                for bit2check in flag_status_bit[flagfield][category]:
                    pos = np.where(pos_8 == bit2check)[0]
                    data['f_usable'][set_bits[pos] / 8] = int(category)

                    # land points
                    if (flagfield == 'flagfield_gen2') and (bit2check == 1):
                        data['f_land'][set_bits[pos] / 8] = 1


def read_smx_fmv_12(eps_file):
    raw_data = eps_file.scaled_mdr
    raw_unscaled = eps_file.mdr

    template = templates.template_SMR__001()
    n_node_per_line = eps_file.mdr[0]['LONGITUDE'].size
    n_records = eps_file.mdr_counter * n_node_per_line
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(n_node_per_line)

    data = np.repeat(template, n_records)

    ascat_time = mpl_dates.num2julian(
        shortcdstime2dtordinal(raw_data['UTC_LINE_NODES'].flatten()['day'],
                               raw_data['UTC_LINE_NODES'].flatten()['time']))
    data['jd'] = ascat_time[idx_nodes]

    fields = [('sig', 'SIGMA0_TRIP', long_nan),
              ('inc', 'INC_ANGLE_TRIP', uint_nan),
              ('azi', 'AZI_ANGLE_TRIP', int_nan),
              ('kp', 'KP', uint_nan),
              ('f_land', 'F_LAND', uint_nan)]

    for field in fields:
        data[field[0]] = raw_data[field[1]].reshape(n_records, 3)
        valid = raw_data[field[1]].reshape(n_records, 3) != field[2]
        data[field[0]][valid == False] = field[2]

    fields = [('lon', 'LONGITUDE', long_nan),
              ('lat', 'LATITUDE', long_nan),
              ('ssm', 'SOIL_MOISTURE', uint_nan),
              ('ssm_noise', 'SOIL_MOISTURE_ERROR', uint_nan),
              ('norm_sigma', 'SIGMA40', long_nan),
              ('norm_sigma_noise', 'SIGMA40_ERROR', long_nan),
              ('slope', 'SLOPE40', long_nan),
              ('slope_noise', 'SLOPE40_ERROR', long_nan),
              ('dry_ref', 'DRY_BACKSCATTER', long_nan),
              ('wet_ref', 'WET_BACKSCATTER', long_nan),
              ('mean_ssm', 'MEAN_SURF_SOIL_MOISTURE', uint_nan),
              ('ssm_sens', 'SOIL_MOISTURE_SENSETIVITY', ulong_nan),
              ('correction_flag', 'CORRECTION_FLAGS', None),
              ('processing_flag', 'PROCESSING_FLAGS', None),
              ('aggregated_flag', 'AGGREGATED_QUALITY_FLAG', None),
              ('snow', 'SNOW_COVER_PROBABILITY', None),
              ('frozen', 'FROZEN_SOIL_PROBABILITY', None),
              ('wetland', 'INNUDATION_OR_WETLAND', None),
              ('topo', 'TOPOGRAPHICAL_COMPLEXITY', None)]

    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        if field[2] is not None:
            valid = raw_unscaled[field[1]].flatten() != field[2]
            data[field[0]][valid == False] = field[2]

    # sat_track_azi (uint)
    data['as_des_pass'] = \
        np.array(raw_data['SAT_TRACK_AZI'].flatten()[idx_nodes] < 27000)

    # modify longitudes from [0,360] to [-180,180]
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    data['param_db_version'] = \
        raw_data['PARAM_DB_VERSION'].flatten()[idx_nodes]
    data['warp_nrt_version'] = \
        raw_data['WARP_NRT_VERSION'].flatten()[idx_nodes]

    data['spacecraft_id'] = int(eps_file.mphr['SPACECRAFT_ID'][2])

    lswath = raw_data['SWATH_INDICATOR'].flatten() == 0
    rswath = raw_data['SWATH_INDICATOR'].flatten() == 1

    if raw_data.dtype.fields.has_key('node_num') is False:
        if (n_node_per_line == 82):
            leftSw = np.arange(20, -21, -1)
            rightSw = np.arange(-20, 21, 1)

        if (n_node_per_line == 42):
            leftSw = np.arange(10, -11, -1)
            rightSw = np.arange(-10, 11, 1)

        lineNum = np.concatenate((leftSw, rightSw), axis=0).flatten()
        nodes = np.repeat(np.array(lineNum, ndmin=2),
                          raw_data['ABS_LINE_NUMBER'].size, axis=0)

    if (n_node_per_line == 82):
        if raw_data.dtype.fields.has_key('node_num'):
            data['node_num'][lswath] = 21 + raw_data['node_num'].flat[lswath]
            data['node_num'][rswath] = 62 + raw_data['node_num'].flat[rswath]
        else:
            data['node_num'][lswath] = 21 + nodes.flat[lswath]
            data['node_num'][rswath] = 62 + nodes.flat[rswath]
    if (n_node_per_line == 42):
        if raw_data.dtype.fields.has_key('node_num'):
            data['node_num'][lswath] = 11 + raw_data['node_num'].flat[lswath]
            data['node_num'][rswath] = 32 + raw_data['node_num'].flat[rswath]
        else:
            data['node_num'][lswath] = 11 + nodes.flat[lswath]
            data['node_num'][rswath] = 32 + nodes.flat[rswath]

    return data


def shortcdstime2dtordinal(days, milliseconds):
    """
    Converting shortcdstime to datetime ordinal.

    Parameters
    ----------
    days : int
        Days.
    milliseconds : int
        Milliseconds

    Returns
    -------
    date : datetime.datetime
        Ordinal datetime.
    """
    epoch = dt.datetime.strptime('2000-01-01 00:00:00',
                                 '%Y-%m-%d %H:%M:%S').toordinal()
    offset = days + (milliseconds / 1000.) / (24. * 60. * 60.)

    return epoch + offset


def test_eps():
    """
    Test read EPS file.
    """
    data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M01_20180403012100Z_20180403030558Z_N_O_20180403030402Z.nat')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M01_20160101000900Z_20160101015058Z_N_O_20160101005610Z.nat.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZO_1B_M02_20070101010300Z_20070101024756Z_R_O_20140127103410Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZO_1B_M02_20140331235400Z_20140401013856Z_R_O_20140528192253Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M02_20070101010300Z_20070101024759Z_R_O_20140127103401Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M02_20140331235400Z_20140401013900Z_R_O_20140528192238Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M02_20071212071500Z_20071212085659Z_R_O_20081225063118Z.nat.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M02_20121212071500Z_20121212085659Z_N_O_20121212080501Z.nat')


if __name__ == '__main__':
    test_eps()
