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
File search methods.
"""

import os
import abc
import glob
import warnings
import tempfile
from datetime import timedelta
from datetime import datetime

import numpy as np

from ascat.level1 import AscatL1bFile


class FilenameTemplate:

    """
    FilenameTemplate class.
    """

    def __init__(self, root_path, fn_templ, sf_templ=None):
        """
        Initialize root path, filename pattern and optional subfolder pattern.

        Parameters
        ----------
        root_path : str
            Root path.
        fn_templ : str
            Filename (glob) pattern.
            e.g. '{date}_*.{suffix}'
        sf_templ : dict, optional
            Subfolder pattern defined as dictionary (default: None).
            Keys represent unique meta names of subfolders and values define
            real folder names and/or (glob) pattern.
            e.g. {'variable': '*', 'tile': 'EN012*'}
        """
        self.root_path = root_path
        self.fn_templ = fn_templ
        self.sf_templ = sf_templ

    @property
    def template(self):
        """
        Name property.
        """
        if self.sf_templ is None:
            filename = os.path.join(self.root_path, self.fn_templ)
        else:
            filename = os.path.join(
                self.root_path, *list(self.sf_templ.values()), self.fn_templ)

        return filename

    def build_filename(self, fn_fmt, sf_fmt=None):
        """
        Create filename from format dictionary.

        Parameters
        ----------
        fn_fmt : dict
            Filename format applied on filename pattern (fn_pattern).
            e.g. fn_pattern = '{date}*.{suffix}'
            with fn_format_dict = {'date': '20000101', 'suffix': 'nc'}
            returns '20000101*.nc'
        fmt : dict of dicts
            Format dictionary for subfolders. Each subfolder contains
            a dictionary defining the format of the folder name.
            e.g. sf_pattern = {'years': {year}, 'months': {month}}
            with format_dict = {'years': {'year': '2000'},
                                'months': {'month': '02'}}
            returns ['2000', '02']

        Returns
        -------
        filename : str
            Filename with format_dict applied.
        """
        fn = self.build_basename(fn_fmt)

        if sf_fmt is None:
            filename = os.path.join(self.root_path, fn)
        else:
            sf = self.build_subfolder(sf_fmt)
            filename = os.path.join(self.root_path, *sf, fn)

        return filename

    def build_basename(self, fmt):
        """
        Create file basename from format dictionary.

        Parameters
        ----------
        fmt : dict
            Filename format applied on filename pattern (fn_pattern).
            e.g. fn_pattern = '{date}*.{suffix}'
            with fmt = {'date': '20000101', 'suffix': 'nc'}
            returns '20000101*.nc'

        Returns
        -------
        filename : str
            Filename with format_dict applied.
        """
        return self.fn_templ.format(**fmt)

    def build_subfolder(self, fmt):
        """
        Create subfolder path from format dictionary.

        Parameters
        ----------
        fmt : dict of dicts
            Format dictionary for subfolders. Each subfolder contains
            a dictionary defining the format of the folder name.
            e.g. sf_pattern = {'years': {year}, 'months': {month}}
            with format_dict = {'years': {'year': '2000'},
                                'months': {'month': '02'}}
            returns ['2000', '02']

        Returns
        -------
        subfolder : list of str
            Subfolder with format_dict applied.
        """
        subfolder = []

        if self.sf_templ is not None:
            for name, v in self.sf_templ.items():
                if fmt is not None:
                    if name in fmt:
                        subfolder.append(self.sf_templ[name].format(
                            **fmt[name]))
                    else:
                        subfolder.append(self.sf_templ[name])
                else:
                    subfolder.append(self.sf_templ[name])

        return subfolder


class FileSearch:

    """
    FileSearch class.
    """

    def __init__(self, root_path, fn_pattern, sf_pattern=None):
        """
        Initialize root path, filename pattern and optional subfolder pattern.

        Parameters
        ----------
        root_path : str
            Root path.
        fn_pattern : str
            Filename (glob) pattern.
            e.g. '{date}_*.{suffix}'
        sf_pattern : dict, optional
            Subfolder pattern defined as dictionary (default: None).
            Keys represent unique meta names of subfolders and values define
            real folder names and/or (glob) pattern.
            e.g. {'variable': '*', 'tile': 'EN012*'}
        """
        self.file_templ = FilenameTemplate(root_path, fn_pattern, sf_pattern)

    def search(self, fn_fmt, sf_fmt=None, recursive=False):
        """
        Search filesystem for given pattern returning list.

        Parameters
        ----------
        fn_fmt : dict
            Filename format dictionary.
        sf_fmt : dict of dicts, optional
            Format dictionary for subfolders (default: None).
        recursive : bool, optional
            If recursive is true, the pattern "**" will match any files and
            zero or more directories, subdirectories and symbolic links to
            directories (default: False).

        Returns
        -------
        filenames : list of str
            Return a possibly-empty list of path/file names that match.
        """
        return glob.glob(self.file_templ.build_filename(fn_fmt, sf_fmt),
                         recursive=recursive)

    def isearch(self, fn_fmt, sf_fmt=None, recursive=False):
        """
        Search filesystem for given pattern returning iterator.

        Parameters
        ----------
        fn_fmt : dict
            Filename format dictionary.
        sf_fmt : dict of dicts, optional
            Format dictionary for subfolders (default: None).
        recursive : bool, optional
            If recursive is true, the pattern "**" will match any files and
            zero or more directories, subdirectories and symbolic links to
            directories (default: False).

        Returns
        -------
        filenames : iterator
            Iterator which yields the same values as search() without
            actually storing them all simultaneously.
        """
        return glob.iglob(self.file_templ.build_filename(fn_fmt, sf_fmt),
                          recursive=recursive)

    def create_search_func(self, func, recursive=False):
        """
        Create custom search function returning it.

        Parameters
        ----------
        func : function
            Search function with its own args/kwargs returning
            a filename format dictionary and subfolder format dictionary
            depending on the passed arguments.
        recursive : bool, optional
            If recursive is true, the pattern "**" will match any files and
            zero or more directories, subdirectories and symbolic links to
            directories (default: False).

        Returns
        -------
        custom_search : function
            Custom search function returning a possibly-empty list
            of path/file names that match.
        """
        def custom_search(*args, **kwargs):
            fn_fmt, sf_fmt = func(*args, **kwargs)
            return self.search(fn_fmt, sf_fmt, recursive=recursive)

        return custom_search

    def create_isearch_func(self, func, recursive=False):
        """
        Create custom search function returning it.

        Parameters
        ----------
        func : function
            Search function with its own args/kwargs returning
            a filename format dictionary and subfolder format dictionary
            depending on the passed arguments.
        recursive : bool, optional
            If recursive is true, the pattern "**" will match any files and
            zero or more directories, subdirectories and symbolic links to
            directories (default: False).

        Returns
        -------
        custom_search : function
            Custom search function returning an iterator of path/file names
            that match.
        """
        def custom_search(*args, **kwargs):
            fn_fmt, sf_fmt = func(*args, **kwargs)
            return self.isearch(fn_fmt, sf_fmt, recursive=recursive)

        return custom_search


class MultiFileHandler(metaclass=abc.ABCMeta):

    """
    MultiFileHandler class.
    """

    def __init__(self, root_path, cls, fn_templ, sf_templ=None, mode='r',
                 cls_kwargs=None, err=False):
        """
        Initialize MultiFileHandler.

        Parameters
        ----------
        root_path : str
            Root path.
        cls : class
            Class reading/writing files.
        fn_templ : str
            Filename template (e.g. '{date}_ascat.nc').
        sf_templ : dict, optional
            Subfolder template defined as dictionary (default: None).
        cls_kwargs : dict, optional
            Class keyword arguments (default: None).
        err : bool, optional
            Set true if error should be re-raised instead of
            reporting a warning.
            Default: False
        """
        self.root_path = root_path
        self.cls = cls
        self.ft = FilenameTemplate(root_path, fn_templ, sf_templ)
        self.mode = mode
        self.fid = None
        self.err = err

        if cls_kwargs is None:
            self.cls_kwargs = {}
        else:
            self.cls_kwargs = cls_kwargs

    def __enter__(self):
        """
        Context manager initialization.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object. The file will be
        closed. The parameters describe the exception that caused the
        context to be exited.
        """
        self._close()

    def _open(self, filename):
        """
        Test whether IO class instance works fine.

        Parameters
        ----------
        filename : str
            Filename.
        """
        self._close()

        if self.mode == 'r' and not os.path.isfile(filename):
            msg = 'File not found: {}'.format(filename)
            if self.err:
                raise IOError(msg)
            else:
                warnings.warn(msg)

        try:
            self.fid = self.cls(filename, **self.cls_kwargs)
        except IOError:
            self.fid = None
            if self.err:
                raise
            else:
                warnings.warn("IOError: {}".format(filename))

    def _close(self):
        """
        Close file.
        """
        if self.fid is not None and hasattr(self.fid, 'close'):
            self.fid.close()
            self.fid = None

    @abc.abstractmethod
    def _fmt(*args, **kwargs):
        """
        Filename format and subfolder format used to create distinct filenames.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        return

    def build_filename(self, *args, **kwargs):
        """
        Build filename based on timestamp and filename/folder format
        definitions.

        Returns
        -------
        filename : str
            Filename.
        """
        fn_fmt, sf_fmt = self._fmt(*args, **kwargs)
        filename = self.ft.build_filename(fn_fmt, sf_fmt)

        return filename

    def read(self, *fmt_args, fmt_kwargs=None, cls_kwargs=None):
        """
        Read data.

        Parameters
        ----------
        fmt_args : tuple
            Format arguments.
        fmt_kwargs : dict, optional
            Format keywords (Default: None).
        cls_kwargs : dict, optional
            Class keywords (Default: None).

        Returns
        -------
        data : dict, numpy.ndarray
            Data stored in file.
        """
        if self.mode != 'r':
            raise RuntimeError('Opening mode not read.')

        if fmt_kwargs is None:
            fmt_kwargs = {}

        if cls_kwargs is None:
            cls_kwargs = {}

        filename = self.build_filename(*fmt_args, **fmt_kwargs)
        self._open(filename)
        data = self.fid.read(**cls_kwargs)

        return data

    def write(self, data, *fmt_args, fmt_kwargs=None, cls_kwargs=None):
        """
        Write data.

        Parameters
        ----------
        data : dict, numpy.ndarray
            Data to write.
        fmt_args : tuple
            Format arguments.
        fmt_kwargs : dict, optional
            Format keywords (Default: None).
        cls_kwargs : dict, optional
            Class keywords (Default: None).
        """
        if self.mode != 'w':
            raise RuntimeError('Opening mode not write.')

        if fmt_kwargs is None:
            fmt_kwargs = {}

        if cls_kwargs is None:
            cls_kwargs = {}

        filename = self.build_filename(*fmt_args, **fmt_kwargs)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        self._open(filename)
        self.fid.write(data, **cls_kwargs)


class MultiFileSearch:

    """
    MultiFileSearch class.
    """

    def __init__(self, root_path, cls, fn_templ, fn_search_pattern,
                 sf_templ=None, sf_search_pattern=None, mode='r',
                 cls_kwargs=None, err=False):
        """
        Initialize MultiFileSearch.

        Parameters
        ----------
        root_path : str
            Root path.
        cls : class
            Class reading/writing files.
        fn_templ : str
            Filename template (e.g. 'prefix_{date}_postfix.nc').
        fn_search_pattern : str
            Filename pattern (e.g. 'prefix_{date}*_postfix.nc').
        sf_templ : dict, optional
            Subfolder template defined as dictionary (default: None).
        sf_search_pattern : dict, optional
            Subfolder pattern defined as dictionary (default: None).
        cls_kwargs : dict, optional
            Class keyword arguments (default: None).
        err : bool, optional
            Set true if error should be re-raised instead of
            reporting a warning.
            Default: False
        """
        super().__init__(root_path, cls, fn_templ, sf_templ,
                         mode, cls_kwargs, err)

        self.fs = FileSearch(root_path, fn_search_pattern, sf_search_pattern)

    @abc.abstractmethod
    def _search_fmt(*args, **kwargs):
        """
        Filename format and subfolder format used for searching.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        return

    def search_files(self, *args, **kwargs):
        """
        Search files for given path and custom filename/folder pattern.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        fn_fmt, sf_fmt = self._search_fmt(*args, **kwargs)
        filenames = self.fs.search(fn_fmt, sf_fmt)

        return sorted(filenames)


# class ChronFileCollection(FileCollection):

#     def search_period(self, dt_start, dt_end, delta):
#         """
#         Search files for time period.

#         Parameters
#         ----------
#         start : datetime
#             Start datetime.
#         end : datetime
#             End datetime.
#         delta : timedelta
#             Delta.

#         Returns
#         -------
#         filenames : list of str
#             Filenames.
#         """
#         filenames = []

#         for dt_cur in np.arange(dt_start, dt_end, delta).astype(datetime):
#             file_list = self.search_files(dt_cur)
#             if file_list:
#                 # remove duplicates
#                 # filenames.extend(f for f in sorted(file_list, reverse=True)
#                 #                  if f not in filenames)
#                 filenames.extend(file_list)

#         return sorted(filenames)

    # def read_period(self, period, buf_len=timedelta(hours=3), **kwargs):
    #     """
    #     Read data for given interval.

    #     Parameters
    #     ----------
    #     period : tuple of datetime
    #         Start and end time of period.

    #     Returns
    #     -------
    #     data : dict, numpy.ndarray
    #         Data stored in file.
    #     """
    #     if self.mode not in ['r']:
    #         raise IOError("File mode not read")

    #     start_date, end_date = period
    #     delta = timedelta(days=1)

    #     # buffer
    #     cur_date = start_date - buf_len
    #     interval_search_pattern = "{date}*"

    #     fn = FnSearch(self.path, interval_search_pattern, self.sf_pattern)

    #     cdata = []
    #     while cur_date.date() <= end_date.date():

    #         # fn_format = {'date': '{}'.format(cur_date.strftime('%Y%m%d'))}
    #         fn_format, sf_format = self._fn_sf_format(cur_date)

    #         files = fn.search(fn_format, sf_format)
    #         import pdb
    #         pdb.set_trace()

    #         for f in sorted(files):
    #             if self._open(f):
    #                 data = self.fid.read_interval(interval, **kwargs)
    #                 if data is not None:
    #                     cdata.append(data)

    #         cur_date += delta

    #     return cdata


class CsvFile:

    """
    Read and write CSV file.
    """

    def __init__(self, filename, mode='r'):
        """
        Initialize CsvFile.

        Parameters
        ----------
        filename : str
            Filename
        mode : str, optional
            File opening mode.
        """
        self.filename = filename
        self.mode = mode

    def header2dtype(self, header):
        """
        Convert header string to dtype info.

        Parameters
        ----------
        header : str
            Header string with dtype info.

        Returns
        -------
        dtype : numpy.dtype
            Data type.
        """
        dtype_list = []
        for substr in header.split('(')[2:]:
            d = []
            for substr2 in substr.split(','):
                if substr2.endswith(')'):
                    substr2 = substr2[:-1]
                if substr2.endswith('\n'):
                    substr2 = substr2[:-4]
                substr2 = substr2.strip()
                substr2 = substr2.strip("'")
                if substr2 == '':
                    continue
                d.append(substr2)
            dtype_list.append(tuple(d))

        return np.dtype(dtype_list)

    def read(self):
        """
        Read data from CSV file.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.
        """
        with open(self.filename) as fid:
            header = fid.readline()
            dtype = self.header2dtype(header)
            data = np.loadtxt(fid, dtype)

        return data

    def read_interval(self, interval):
        """
        Read subset data from CSV file for given interval.

        Parameters
        ----------
        interval : (datetime, datetime)
            Time interval to extract data.

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        with open(self.filename) as fid:
            header = fid.readline()
            dtype = self.header2dtype(header)
            data = np.loadtxt(fid, dtype)

        subset = ((data['date'] >= np.datetime64(interval[0])) &
                  (data['date'] <= np.datetime64(interval[1])))

        if np.sum(subset) > 0:
            data = data[subset]
        else:
            data = None

        return data

    def write(self, data):
        """
        Write data to CSV file.

        Parameters
        ----------
        data : numpy.ndarray
            Data.
        """
        header = data.dtype.__repr__()
        np.savetxt(self.filename, data, fmt='%s', header=header)


class CsvFiles(MultiFileHandler):

    """
    CSV Files.
    """

    def __init__(self, root_path, mode='r'):
        """
        Initialize CsvMultiFileHandler.

        Parameters
        ----------
        root_path : str
            Root path.
        """
        fn_templ = 'prefix_{date}_postfix.csv'
        sf_templ = {'Y': '{year}', 'M': '{month}'}

        super().__init__(root_path, CsvFile, fn_templ, sf_templ=sf_templ,
                         mode=mode)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_fmt = {'date': timestamp.strftime('%Y%m%d_%H%M%S')}
        sf_fmt = {'Y': {'year': timestamp.strftime('%Y')},
                  'M': {'month': timestamp.strftime('%m')}}

        return fn_fmt, sf_fmt


class CsvFilesSearch():

    def __init__(self):
        fn_search_pattern = 'prefix_{date}*_postfix.csv'
        sf_search_pattern = {'Y': '{year}', 'M': '{month}'}

    def _search_fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_fmt = {'date': timestamp.strftime('%Y%m%d')}
        sf_fmt = {'Y': {'year': timestamp.strftime('%Y')},
                  'M': {'month': timestamp.strftime('%m')}}

        return fn_fmt, sf_fmt


def test_csv():
    """
    Testing writing and reading CSV files.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    dtype = np.dtype([('date', 'datetime64[s]'), ('num', np.int32)])
    dates = np.datetime64('2000-01-01') + np.arange(3)
    file_length = np.array([3, 5, 2, 7, 1]) * 60

    csv = CsvFiles(tmp_dir.name, mode='w')

    j = 0
    k = 0
    for date in dates:
        arr = []
        for i in range(86400):
            dt = date + np.timedelta64(i, 's')
            arr.append((dt, i))
            k = k + 1
            if k == file_length[j]:
                arr = np.array(arr, dtype=dtype)
                timestamp = arr[0]['date'].astype(datetime)
                csv.write(arr, timestamp)
                k = 0
                arr = []
                j = j + 1
                if np.mod(j, len(file_length)) == 0:
                    j = 0

    csv = CsvFiles(tmp_dir.name)

    data = csv.read(datetime(2000, 1, 1))
    print(data)

    data = csv.read(datetime(2000, 1, 1, 3))
    print(data)

    # files = csv_files.search_files(datetime(2000, 1, 1))
    # print(files)

    # period = (datetime(2000, 1, 1), datetime(2000, 1, 1, 10, 0, 0))
    # files = csv_files.search_period(*period)
    # print(files)

    tmp_dir.cleanup()


class AscatL1bNc(MultiFileHandler):

    def __init__(self, path, sat='A'):
        """
        Initialize CsvMultiFileHandler.
        """
        self.sat = sat
        fn_templ = 'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOP{sat}+ASCAT_C_EUMP_{date}_52940_eps_o_125_l1.nc'
        fn_search_pattern = 'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOP{sat}+ASCAT_C_EUMP_{date}*_*_eps_o_125_l1.nc'

        super().__init__(path, AscatL1bFile, fn_templ, fn_search_pattern)

    def _fmt(self, timestamp):
        """
        Definition of filename and subfolder format.

        Parameters
        ----------
        timestamp : datetime
            Time stamp.

        Returns
        -------
        fn_fmt : dict
            Filename format.
        sf_fmt : dict
            Subfolder format.
        """
        fn_fmt = {'date': timestamp.strftime('%Y%m%d%H%M%S'), 'sat': self.sat}
        sf_fmt = None

        return fn_fmt, sf_fmt

    def _search_fmt(self, timestamp):
        """
        """
        fn_search_fmt = {'date': timestamp.strftime('%Y%m%d'), 'sat': self.sat}
        sf_search_fmt = None

        return fn_search_fmt, sf_search_fmt


def main():
    path = '/home/shahn/swdvlp/ascat/tests/ascat_test_data/eumetsat/ASCAT_L1_SZR_NC/'
    dt = datetime(2017, 1, 1, 1, 24, 0)
    obj = AscatL1bNc(path)

    # data = obj.read(dt)
    # print(data)

    period = (datetime(2017, 1, 1, 1, 24, 0), datetime(2017, 1, 1, 2, 0, 0))
    data = obj.search_period(*period)
    print(data)


if __name__ == '__main__':
    # main()
    test_csv()
