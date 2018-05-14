import os
import fnmatch
import lxml.etree as etree
from tempfile import NamedTemporaryFile
from gzip import GzipFile
from collections import OrderedDict

import numpy as np
import pygenio.genio as genio
import datetime as dt
import matplotlib.dates as mpl_dates

short_cds_time = np.dtype([('day', np.uint16), ('time', np.uint32)])

long_cds_time = np.dtype([('day', np.uint16), ('ms', np.uint32),
                          ('mms', np.uint16)])

long_nan = -2 ** 31
ulong_nan = 2 ** 32 - 1
int_nan = -2 ** 15
uint_nan = 2 ** 16 - 1
byte_nan = -2 ** 7

def template_SZF__001():
    """
    Re-sampled backscatter template 001.

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
    """
    metadata = {'temp_name': 'SZF__001'}

    struct = np.dtype([('jd', np.double),
                       ('spacecraft_id', np.int8),
                       ('processor_major_version', np.int16),
                       ('processor_minor_version', np.int16),
                       ('format_major_version', np.int16),
                       ('format_minor_version', np.int16),
                       ('degraded_inst_mdr', np.int8),
                       ('degraded_proc_mdr', np.int8),
                       ('sat_track_azi', np.float32),
                       ('as_des_pass', np.int8),
                       ('swath_indicator', np.int8),
                       ('azi', np.float32),
                       ('inc', np.float32),
                       ('sig', np.float32),
                       ('lat', np.float32),
                       ('lon', np.float32),
                       ('beam_number', np.int8),
                       ('land_frac', np.float32),
                       ('flagfield_rf1', np.uint8),
                       ('flagfield_rf2', np.uint8),
                       ('flagfield_pl', np.uint8),
                       ('flagfield_gen1', np.uint8),
                       ('flagfield_gen2', np.uint8),
                       ('f_usable', np.uint8),
                       ('f_land', np.uint8)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset


def template_SZX__002():
    """
    Re-sampled backscatter template 002.
    """
    metadata = {'temp_name': 'SZX__002'}

    struct = np.dtype([('jd', np.double),
                       ('spacecraft_id', np.int8),
                       ('abs_orbit_nr', np.uint32),
                       ('processor_major_version', np.int16),
                       ('processor_minor_version', np.int16),
                       ('format_major_version', np.int16),
                       ('format_minor_version', np.int16),
                       ('degraded_inst_mdr', np.int8),
                       ('degraded_proc_mdr', np.int8),
                       ('sat_track_azi', np.float32),
                       ('as_des_pass', np.int8),
                       ('swath_indicator', np.int8),
                       ('azi', np.float32, 3),
                       ('inc', np.float32, 3),
                       ('sig', np.float32, 3),
                       ('lat', np.float32),
                       ('lon', np.float32),
                       ('kp', np.float32, 3),
                       ('node_num', np.int16),
                       ('line_num', np.int32),
                       ('num_val', np.uint32, 3),
                       ('f_kp', np.int8, 3),
                       ('f_usable', np.int8, 3),
                       ('f_f', np.uint16, 3),
                       ('f_v', np.uint16, 3),
                       ('f_oa', np.uint16, 3),
                       ('f_sa', np.uint16, 3),
                       ('f_tel', np.uint16, 3),
                       ('f_ref', np.uint16, 3),
                       ('f_land', np.float32, 3)], metadata=metadata)

    dataset = np.zeros(1, dtype=struct)

    return dataset

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
        # record_class_dict = {1: 'MPHR', 2: 'SPHR', 3: 'IPR', 4: 'GEADR',
        #                      5: 'GIADR', 6: 'VEADR', 7: 'VIADR', 8: 'MDR'}

        mdr_template = None

        self.fid = open(self.filename, 'rb')
        self.filesize = os.path.getsize(self.filename)
        self.eor = self.fid.tell()

        while self.eor < self.filesize:

            # remember beginning of the record
            self.bor = self.fid.tell()

            # read grh of current record
            self.grh = self._read_record(grh_record())[0]
            record_size = self.grh['record_size']
            record_class = self.grh['record_class']
            record_subclass = self.grh['record_subclass']

            # mphr
            if record_class == 1:
                self._read_mphr()
                self.xml_file = self._get_eps_xml()
                self.xml_doc = etree.parse(self.xml_file)
                self.mdr_template, self.scaled_template, self.sfactor = self._read_xml_mdr()

            # sphr
            elif record_class == 2:
                self._read_sphr()

            # geadr
            elif record_class == 3:
                self._read_geadr()

            elif record_class == 4:
                self._read_geadr()

            # veadr
            elif record_class == 6:
                self._read_veadr()

            # viadr
            elif record_class == 7:
                template, scaled_template, sfactor = self._read_xml_viadr(record_subclass)
                viadr_element = self._read_record(template)
                viadr_element_sc = self._scaling(viadr_element, scaled_template, sfactor)
                if record_subclass == 8:
                    if self.viadr_grid is None:
                        self.viadr_grid = [viadr_element]
                        self.viadr_grid_scaled = [viadr_element_sc]
                    else:
                        self.viadr_grid.append(viadr_element)
                        self.viadr_grid_scaled.append(viadr_element_sc)
                else:
                    if self.viadr is None:
                        self.viadr = [viadr_element]
                        self.viadr_scaled = [viadr_element_sc]
                    else:
                        self.viadr.append(viadr_element)
                        self.viadr_scaled.append(viadr_element_sc)

            # mdr
            elif record_class == 8:
                if self.grh['instrument_group'] == 13:
                    pass
                else:
                    mdr_element = self._read_record(self.mdr_template)
                    if self.mdr is None:
                        self.mdr = [mdr_element]
                    else:
                        self.mdr.append(mdr_element)
                    self.mdr_counter += 1

            # return pointer to the beginning of the record
            self.fid.seek(self.bor)
            self.fid.seek(record_size, 1)


            # determine number of bytes read
            # end of record
            self.eor = self.fid.tell()

        self.fid.close()

        self.mdr = np.hstack(self.mdr)
        self.scaled_mdr = self._scaling(self.mdr, self.scaled_template, self.sfactor)

    def _scaling(self, unscaled_data, scaled_template, sfactor):
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
        mphr = self.fid.read(self.grh['record_size'] - self.grh.itemsize)
        self.mphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in mphr.split('\n')[:-1])

    def _read_sphr(self):
        """
        Read Special Product Header (SPHR).
        """
        sphr = self.fid.read(self.grh['record_size'] - self.grh.itemsize)
        self.sphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in sphr.split('\n')[:-1])

    def _read_geadr(self):
        pass

    def _read_veadr(self):
        pass

    def _get_eps_xml(self):
        '''
        Find the corresponding eps xml file.

        :param: mphr_dict
        :return: filename
        '''
        format_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                   'formats')

        for filename in fnmatch.filter(os.listdir(format_path), 'eps_ascat*'):
            doc = etree.parse(os.path.join(format_path, filename))
            file_extension = doc.xpath('//file-extensions')[0].getchildren()[0]

            format_version = doc.xpath('//format-version')
            for elem in format_version:
                major = elem.getchildren()[0]
                minor = elem.getchildren()[1]
                if major.text == self.mphr['FORMAT_MAJOR_VERSION'] and \
                        minor.text == self.mphr['FORMAT_MINOR_VERSION'] and \
                        self.mphr[
                            'PROCESSING_LEVEL'] in file_extension.text and \
                        self.mphr['PRODUCT_TYPE'] in file_extension.text:
                    return os.path.join(format_path, filename)


    def _read_xml_viadr(self, subclassid=99):
        """
        Read xml record.
        """
        if subclassid == 4:
            subclass = 0
        elif subclassid == 6:
            subclass = 1
        elif subclassid == 8:
            subclass = 2
        else:
            raise RuntimeError("VIADR subclass not supported.")

        elements = self.xml_doc.xpath('//viadr')
        data = OrderedDict()
        length = []
        elem = elements[subclass]

        for child in elem.getchildren():

            if child.tag == 'delimiter':
                continue

            child_items = dict(child.items())
            name = child_items.pop('name')

            longtime_flag = ('type' in child_items and
                             'longtime' in child_items['type'])

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
                'uinteger1': np.uint8,
                'integer': np.int32, 'uinteger': np.uint32,
                'integer2': np.int16,
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

        return np.dtype(dtype), np.dtype(scaled_dtype), np.array(scaling_factor,
                                                                 dtype=np.float32)

    def _read_xml_mdr(self):
        """
        Read xml record.
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

            bitfield_flag = ('type' in child_items and
                             'bitfield' in child_items['type'])

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

                    try:
                        if arr_items['type']:
                            bitfield_flag = True;
                    except KeyError:
                        pass

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
                'boolean': np.uint8, 'integer1': np.int8, 'uinteger1': np.uint8,
                'integer': np.int32, 'uinteger': np.uint32, 'integer2': np.int16,
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

        return np.dtype(dtype), np.dtype(scaled_dtype), np.array(scaling_factor,
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

def read_eps_l1b(filename):
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
        if fmv == 12:
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

    return raw_data, data, metadata

def read_eps(filename):
    """
    Read EPS file.
    """

    zipped = False
    if os.path.splitext(filename)[1] == '.gz':
        zipped = True

    if zipped:
        with NamedTemporaryFile(delete=False) as tmp_fid:
            with GzipFile(filename) as gz_fid:
                tmp_fid.write(gz_fid.read())
            filename = tmp_fid.name

    prod = EPSProduct(filename)
    prod.read_product()

    if zipped:
        os.remove(filename)

    return prod


def read_szx_fmv_11(eps_file):
    """
            Read SZO/SZR format version 12.

            Parameters
            ----------
            filename: filename of the eps product

            Returns
            -------
            data : numpy.ndarray
                SZO/SZR data.
            """
    raw_data = eps_file.scaled_mdr
    mphr = eps_file.mphr

    template = template_SZX__002()
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
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid]

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
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid]

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
        filename: filename of the eps product

        Returns
        -------
        data : numpy.ndarray
            SZO/SZR data.
        """
    raw_data = eps_file.scaled_mdr
    mphr = eps_file.mphr

    template = template_SZX__002()
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
              ('degraded_proc_mdr','DEGRADED_PROC_MDR'),
              ('sat_track_azi','SAT_TRACK_AZI')]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()[idx_nodes]

    fields = [('lon', 'LONGITUDE', long_nan),
              ('lat', 'LATITUDE', long_nan),
              ('swath_indicator', 'SWATH INDICATOR', byte_nan)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid]

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
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid]

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
        filename: filename of the eps product

        Returns
        -------
        data : numpy.ndarray
            SZF data.
        """
    raw_data = eps_file.scaled_mdr
    mphr = eps_file.mphr

    template = template_SZF__001()
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
              ('degraded_proc_mdr','DEGRADED_PROC_MDR'),
              ('sat_track_azi','SAT_TRACK_AZI'),
              ('as_des_pass','AS_DES_PASS'),
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
        unpacked_bits = np.unpackbits(data[flagfield])

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
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M01_20180403012100Z_20180403030558Z_N_O_20180403030402Z.nat')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M01_20160101000900Z_20160101015058Z_N_O_20160101005610Z.nat.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZO_1B_M02_20070101010300Z_20070101024756Z_R_O_20140127103410Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZO_1B_M02_20140331235400Z_20140401013856Z_R_O_20140528192253Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M02_20070101010300Z_20070101024759Z_R_O_20140127103401Z.gz')
    data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M02_20140331235400Z_20140401013900Z_R_O_20140528192238Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M02_20070906003300Z_20070906021459Z_R_O_20081223163950Z.nat.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZF_1B_M02_20100101013000Z_20100101031159Z_R_O_20130824055501Z.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M02_20071212071500Z_20071212085659Z_R_O_20081225063118Z.nat.gz')
    # data = read_eps_l1b('/home/mschmitz/Desktop/ascat_test_data/level1/eps_nat/ASCA_SZR_1B_M02_20121212071500Z_20121212085659Z_N_O_20121212080501Z.nat')


if __name__ == '__main__':
    test_eps()
