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
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import numpy as np
import xarray as xr
import lxml.etree as etree
from cadati.jd_date import jd2dt

short_cds_time = np.dtype([('day', np.uint16), ('time', np.uint32)])

long_cds_time = np.dtype([('day', np.uint16), ('ms', np.uint32),
                          ('mms', np.uint16)])

long_nan = np.iinfo(np.int32).min
ulong_nan = np.iinfo(np.uint32).max
int_nan = np.iinfo(np.int16).min
uint_nan = np.iinfo(np.uint16).max
byte_nan = np.iinfo(np.byte).min
int8_nan = np.iinfo(np.int8).max
uint8_nan = np.iinfo(np.uint8).max
float32_nan = -999999.

# 1.1.2000 00:00:00 as jd
julian_epoch = 2451544.5


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
                self.mdr_template, self.scaled_template, self.sfactor = \
                    self._read_xml_mdr()

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

    def _scaling(self, unscaled_data, scaled_template, sfactor):
        """
        Scale the data
        """
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
                                for item in
                                mphr.decode("utf-8").split('\n')[:-1])

    def _read_sphr(self):
        """
        Read Special Product Header (SPHR).
        """
        sphr = self.fid.read(self.grh[0]['record_size'] - self.grh[0].itemsize)
        self.sphr = OrderedDict(item.replace(' ', '').split('=')
                                for item in
                                sphr.decode("utf-8").split('\n')[:-1])

    def _read_pointer(self, count=1):
        """
        Read pointer record.
        """
        dtype = np.dtype([('aux_data_pointer', np.ubyte, 100)])
        record = np.fromfile(self.fid, dtype=dtype, count=count)
        return record.newbyteorder('B')

    def _get_eps_xml(self):
        """
        Find the corresponding eps xml file.
        """
        format_path = os.path.join(os.path.dirname(__file__), '..',
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
                sf = float(eval(value['scaling-factor'].replace('^', '**')))
            else:
                sf_dtype = conv[value['type']]
                sf = 1.

            if not isinstance(value['length'], list):
                length = [value['length']]
            else:
                length = value['length']

            scaling_factor.append(sf)
            scaled_dtype.append((key, sf_dtype, length))
            dtype.append((key, conv[value['type']], length))

        return np.dtype(dtype), np.dtype(scaled_dtype), np.array(scaling_factor)

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
                sf = float(eval(value['scaling-factor'].replace('^', '**')))
            else:
                sf_dtype = conv[value['type']]
                sf = 1.

            scaling_factor.append(sf)

            if not isinstance(value['length'], list):
                length = [value['length']]
            else:
                length = value['length']

            scaled_dtype.append((key, sf_dtype, length))
            dtype.append((key, conv[value['type']], length))

        return np.dtype(dtype), np.dtype(scaled_dtype), np.array(scaling_factor)


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

    gen_fields_lut = {'inc_angle_full': ('inc', np.float32),
                      'azi_angle_full': ('azi', np.float32),
                      'sigma0_full': ('sig', np.float32)}

    for var_name in skip_fields:
        if var_name in data:
            data.pop(var_name)

    for var_name in data.keys():
        if var_name in gen_fields_lut:
            new_name = gen_fields_lut[var_name][0]
            new_dtype = gen_fields_lut[var_name][1]
            data[new_name] = data.pop(var_name).astype(new_dtype)

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
                      'sigma0_trip': ('sig', np.float32, -2147483600.),
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


def read_eps_l1b(filename, generic=False, to_xarray=False):
    """
    Use of correct Level 1b reader and data preparation.

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

    if ptype == 'SZF':

        if fmv == 12:
            data, metadata, orbit_grid = read_szf_fmv_12(eps_file)
        else:
            raise RuntimeError("SZF format version not supported.")

        data['time'] = jd2dt(data.pop('jd'))

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
                for var_name in data.keys():

                    if len(data[var_name][subset].shape) == 1:
                        dtype.append(
                            (var_name, data[var_name][subset].dtype.str))
                    elif len(data[var_name][subset].shape) > 1:
                        dtype.append((var_name, data[var_name][
                            subset].dtype.str, data[var_name][
                                subset].shape[1:]))

                ds[antenna] = np.empty(
                    data['time'][subset].size, dtype=np.dtype(dtype))

                for var_name, v in data.items():
                    if var_name == 'beam_number' and generic:
                        continue
                    ds[antenna][var_name] = v[subset]

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


def read_eps_l2(filename):
    """
    Use of correct Level 2 reader and data preparation.

    Parameters
    ----------
    filename : str
        ASCAT Level 2 file name in EPS Native format.

    Returns
    -------
    ds : xarray.Dataset, dict of xarray.Dataset
        ASCAT Level 2 data.
    """
    data = {}
    eps_file = read_eps(filename)
    ptype = eps_file.mphr['PRODUCT_TYPE']
    fmv = int(eps_file.mphr['FORMAT_MAJOR_VERSION'])

    if ptype in ['SMR', 'SMO']:
        if fmv == 12:
            raw_data, metadata = read_smx_fmv_12(eps_file)
        else:
            raise RuntimeError("SMX format version not supported.")

        coords = {"lon": (['obs'], data.pop('longitude')),
                  "lat": (['obs'], data.pop('latitude')),
                  "time": (['obs'], jd2dt(data.pop('jd')))}

        data_var = {}
        for k, v in raw_data.items():
            if len(v.shape) == 1:
                dim = ['obs']
            elif len(v.shape) == 2:
                dim = ['obs', 'beam']
            else:
                raise RuntimeError('Unknown dimension')

            data_var[k.lower()] = (dim, v)

        sat_id = np.array([4, 3, 5])
        metadata['sat_id'] = sat_id[metadata['spacecraft_id']-1]

        ds = xr.Dataset(data_var, coords=coords, attrs=metadata)

    else:
        raise ValueError("Format not supported. Product type {:1}"
                         " Format major version: {:2}".format(ptype, fmv))

    return ds


def read_eps(filename):
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
    orbit_gri : numpy.ndarray
        6.25km orbit lat/lon grid.
    """
    raw_data = eps_file.scaled_mdr
    mphr = eps_file.mphr

    n_node_per_line = raw_data['LONGITUDE_FULL'].shape[1]
    n_lines = eps_file.mdr_counter

    data = {}
    metadata = {}
    idx_nodes = np.arange(n_lines).repeat(n_node_per_line)

    ascat_time = shortcdstime2jd(raw_data['UTC_LOCALISATION'].flatten()['day'],
                                 raw_data['UTC_LOCALISATION'].flatten()[
                                     'time'])
    data['jd'] = ascat_time[idx_nodes]

    metadata['spacecraft_id'] = np.int8(mphr['SPACECRAFT_ID'][-1])
    metadata['orbit_start'] = np.uint32(eps_file.mphr['ORBIT_START'])

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']
    for f in fields:
        metadata[f] = np.int16(mphr[f.upper()])

    fields = ['degraded_inst_mdr', 'degraded_proc_mdr', 'sat_track_azi',
              'beam_number', 'flagfield_rf1', 'flagfield_rf2',
              'flagfield_pl', 'flagfield_gen1']
    for f in fields:
        data[f] = raw_data[f.upper()].flatten()[idx_nodes]

    data['swath_indicator'] = np.int8(data['beam_number'].flatten() > 3)

    fields = [('longitude_full', long_nan),
              ('latitude_full', long_nan),
              ('sigma0_full', long_nan),
              ('inc_angle_full', uint_nan),
              ('azi_angle_full', int_nan),
              ('land_frac', uint_nan),
              ('flagfield_gen2', byte_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].flatten()
        valid = eps_file.mdr[f.upper()].flatten() != nan_val
        data[f][~valid] = nan_val

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['longitude_full'] != long_nan,
                          data['longitude_full'] > 180)
    data['longitude_full'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi_angle_full'] != int_nan) & (data['azi_angle_full'] < 0)
    data['azi_angle_full'][mask] += 360

    grid_nodes_per_line = 2 * 81

    viadr_grid = np.concatenate(eps_file.viadr_grid)
    orbit_grid = np.zeros(viadr_grid.size * grid_nodes_per_line,
                          dtype=np.dtype([('lon', np.float32),
                                          ('lat', np.float32),
                                          ('node_num', np.uint8),
                                          ('line_num', np.uint32)]))

    for pos_all in range(orbit_grid['lon'].size):
        line = pos_all // grid_nodes_per_line
        pos_small = pos_all % 81
        if pos_all % grid_nodes_per_line <= 80:
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
        orbit_grid[field] = orbit_grid[field] * 1e-6

    mask = (orbit_grid['lon'] != long_nan) & (orbit_grid['lon'] > 180)
    orbit_grid['lon'][mask] += -360.

    set_flags(data)

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data, metadata, orbit_grid


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

    n_node_per_line = raw_data['longitude'].shape[1]
    n_lines = raw_data['longitude'].shape[0]
    n_records = eps_file.mdr_counter * n_node_per_line
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(n_node_per_line)

    data = {}
    metadata = {}

    metadata['spacecraft_id'] = np.int8(eps_file.mphr['spacecraft_id'][-1])
    metadata['orbit_start'] = np.uint32(eps_file.mphr['orbit_start'])

    ascat_time = shortcdstime2jd(raw_data['utc_line_nodes'].flatten()['day'],
                                 raw_data['utc_line_nodes'].flatten()['time'])
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

    fields = [('longitude', long_nan),
              ('latitude', long_nan),
              ('soil_moisture', uint_nan),
              ('swath_indicator', byte_nan),
              ('soil_moisture_error', uint_nan),
              ('sigma40', long_nan),
              ('sigma40_error', long_nan),
              ('slope40', long_nan),
              ('slope40_error', long_nan),
              ('dry_backscatter', long_nan),
              ('wet_backscatter', long_nan),
              ('mean_surf_soil_moisture', uint_nan),
              ('soil_moisture_sensetivity', ulong_nan),
              ('correction_flags', uint8_nan),
              ('processing_flags', uint8_nan),
              ('aggregated_quality_flag', uint8_nan),
              ('snow_cover_probability', uint8_nan),
              ('frozen_soil_probability', uint8_nan),
              ('innudation_or_wetland', uint8_nan),
              ('topographical_complexity', uint8_nan)]

    for f, nan_val in fields:
        data[f] = raw_data[f.upper()].flatten()
        valid = raw_unscaled[f.upper()].flatten() != nan_val
        data[f][~valid] = nan_val

    # sat_track_azi (uint)
    data['as_des_pass'] = \
        np.array(raw_data['sat_track_azi'].flatten()[idx_nodes] < 270)

    # modify longitudes from [0,360] to [-180,180]
    mask = np.logical_and(data['longitude'] != long_nan,
                          data['longitude'] > 180)
    data['longitude'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi_angle_trip'] != int_nan) & (data['azi_angle_trip'] < 0)
    data['azi_angle_trip'][mask] += 360

    fields = ['param_db_version', 'warp_nrt_version']
    for f in fields:
        data[f] = raw_data['param_db_version'].flatten()[idx_nodes]

    metadata['spacecraft_id'] = int(eps_file.mphr['spacecraft_id'][2])

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

    Parameters
    ----------
    data : numpy.ndarray
        SZF data.

    Returns
    -------
    data : numpy.ndarray
        SZF data with updated flags.
    """
    # category:status = 'red': 2, 'amber': 1, 'warning': 0
    flag_status_bit = {'flagfield_rf1': {'2': [2, 4], '1': [0, 1, 3]},
                       'flagfield_rf2': {'2': [0, 1]},
                       'flagfield_pl': {'2': [0, 1, 2, 3], '0': [4]},
                       'flagfield_gen1': {'2': [1], '0': [0]},
                       'flagfield_gen2': {'2': [2], '1': [0], '0': [1]}}

    for flagfield in flag_status_bit.keys():

        # get flag data in binary format to get flags
        unpacked_bits = np.unpackbits(data[flagfield])

        # find indizes where a flag is set
        set_bits = np.where(unpacked_bits == 1)[0]
        if set_bits.size != 0:
            pos_8 = 7 - (set_bits % 8)

            for category in sorted(flag_status_bit[flagfield].keys()):
                if (int(category) == 0) and (flagfield != 'flagfield_gen2'):
                    continue

                for bit2check in flag_status_bit[flagfield][category]:
                    pos = np.where(pos_8 == bit2check)[0]
                    data['f_usable'] = np.zeros(data['flagfield_gen2'].size,
                                                dtype=np.int8)
                    data['f_usable'][set_bits[pos] // 8] = int(category)

                    # land points
                    if (flagfield == 'flagfield_gen2') and (bit2check == 1):
                        data['f_land'] = np.zeros(data['flagfield_gen2'].size,
                                                  dtype=np.int8)
                        data['f_land'][set_bits[pos] // 8] = 1

    return data
