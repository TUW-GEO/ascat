# Copyright (c) 2023, TU Wien, Department of Geodesy and Geoinformation
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

import abc
import glob
import re
import warnings
from pathlib import Path
from datetime import timedelta
from datetime import datetime

import numpy as np


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
            e.g. "{date}_*.{suffix}"
        sf_templ : dict, optional
            Subfolder pattern defined as dictionary (default: None).
            Keys represent unique meta names of subfolders and values define
            real folder names and/or (glob) pattern.
            e.g. {"variable": "*", "tile": "EN012*"}
        """
        self.root_path = Path(root_path)
        self.fn_templ = fn_templ
        self.sf_templ = sf_templ

    @property
    def template(self):
        """
        Name property.
        """
        if self.sf_templ is None:
            filename = self.root_path / self.fn_templ
        else:
            filename = self.root_path.joinpath(*list(self.sf_templ.values()),
                                               self.fn_templ)

        return filename

    def build_filename(self, fn_fmt, sf_fmt=None):
        """
        Create filename from format dictionary.

        Parameters
        ----------
        fn_fmt : dict
            Filename format applied on filename pattern (fn_pattern).
            e.g. fn_pattern = "{date}*.{suffix}"
            with fn_format_dict = {"date": "20000101", "suffix": "nc"}
            returns "20000101*.nc"
        sf_fmt : dict of dicts
            Format dictionary for subfolders. Each subfolder contains
            a dictionary defining the format of the folder name.
            e.g. sf_templ = {"years": {year}, "months": {month}}
            with sf_format = {"years": {"year": "2000"},
                                "months": {"month": "02"}}
            returns ["2000", "02"]

        Returns
        -------
        filename : str
            Filename with format_dict applied.
        """
        fn = self.build_basename(fn_fmt)

        if sf_fmt is None:
            filename = self.root_path / fn
        else:
            sf = self.build_subfolder(sf_fmt)
            filename = self.root_path.joinpath(*sf, fn)

        return str(filename)

    def build_basename(self, fmt):
        """
        Create file basename from format dictionary.

        Parameters
        ----------
        fmt : dict
            Filename format applied on filename pattern (fn_pattern).
            e.g. fn_pattern = "{date}*.{suffix}"
            with fmt = {"date": "20000101", "suffix": "nc"}
            returns "20000101*.nc"

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
            e.g. sf_pattern = {"years": {year}, "months": {month}}
            with format_dict = {"years": {"year": "2000"},
                                "months": {"month": "02"}}
            returns ["2000", "02"]

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
                        subfolder.append(
                            self.sf_templ[name].format(**fmt[name]))
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
            e.g. "{date}_*.{suffix}"
        sf_pattern : dict, optional
            Subfolder pattern defined as dictionary (default: None).
            Keys represent unique meta names of subfolders and values define
            real folder names and/or (glob) pattern.
            e.g. {"variable": "*", "tile": "EN012*"}
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

    def __init__(
        self,
        root_path,
        cls,
        fn_templ,
        sf_templ=None,
        cls_kwargs=None,
        err=False,
        cache_size=0,
    ):
        """
        Initialize MultiFileHandler class.

        Parameters
        ----------
        root_path : str
            Root path.
        cls : class
            Class reading/writing files.
        fn_templ : str
            Filename template (e.g. "{date}_ascat.nc").
        sf_templ : dict, optional
            Subfolder template defined as dictionary (default: None).
        cls_kwargs : dict, optional
            Class keyword arguments (default: None).
        err : bool, optional
            Set true if a file error should be re-raised instead of
            reporting a warning.
            Default: False
        cache_size : int, optional
            Number of files to keep in memory (default=0).
        """
        self.root_path = root_path
        self.cls = cls
        self.ft = FilenameTemplate(root_path, fn_templ, sf_templ)
        self.fid = None
        self.err = err

        self.cache_size = cache_size
        if cache_size > 0:
            self.cache = {}

        if cls_kwargs is None:
            self.cls_kwargs = {}
        else:
            self.cls_kwargs = cls_kwargs

        self.fs = FileSearch(self.root_path, self.ft.fn_templ,
                             self.ft.sf_templ)

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
        Open file, i.e. create io class instance.

        Parameters
        ----------
        filename : str
            Filename.
        """
        self._close()

        try:
            self.fid = self.cls(filename, **self.cls_kwargs)
        except IOError:
            self.fid = None
            if self.err:
                raise
            else:
                warnings.warn(f"IOError: {filename}")

    def _close(self):
        """
        Try closing file.
        """
        if self.fid is not None and hasattr(self.fid, "close"):
            self.fid.close()
            self.fid = None

    @abc.abstractmethod
    def _fmt(*args, **kwargs):
        """
        Filename format and subfolder format used to read/write
        individual files.

        Returns
        -------
        fn_read_fmt : dict
            Filename format.
        sf_read_fmt : dict
            Subfolder format.
        fn_write_fmt : dict
            Filename format.
        sf_write_fmt : dict
            Subfolder format.
        """
        return

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
        if fmt_kwargs is None:
            fmt_kwargs = {}

        if cls_kwargs is None:
            cls_kwargs = {}

        fn_read_fmt, sf_read_fmt, _, _ = self._fmt(*fmt_args, **fmt_kwargs)
        search_filename = self.ft.build_filename(fn_read_fmt, sf_read_fmt)
        filename = glob.glob(search_filename)

        data = None
        if len(filename) == 0:
            msg = f"File not found: {search_filename}"
            if self.err:
                raise IOError(msg)
            else:
                warnings.warn(msg)
        elif len(filename) > 1:
            msg = "Multiple files found"
            if self.err:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)
        else:
            data = self.read_file(filename[0], cls_kwargs)

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
        if fmt_kwargs is None:
            fmt_kwargs = {}

        if cls_kwargs is None:
            cls_kwargs = {}

        _, _, fn_write_fmt, sf_write_fmt = self._fmt(*fmt_args, **fmt_kwargs)
        filename = self.ft.build_filename(fn_write_fmt, sf_write_fmt)
        self.write_file(data, filename, cls_kwargs=cls_kwargs)

    def read_file(self, filename, cls_kwargs=None):
        """
        Read data for given filename.

        Parameters
        ----------
        filename : str
            Filename.
        """
        if self.cache_size > 0 and filename in self.cache:
            return self.cache[filename]

        self._open(filename)
        data = self.fid.read(**cls_kwargs)

        if self.cache_size > 0:
            if len(self.cache) == self.cache_size:
                del self.cache[next(iter(self.cache))]
            self.cache[filename] = data

        return data

    def write_file(self, data, filename, cls_kwargs=None):
        """
        Write data for given filename.

        Parameters
        ----------
        filename : str
            Filename.
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        self._open(filename)
        self.fid.write(data, **cls_kwargs)

    def search(
        self,
        fn_search_pattern,
        sf_search_pattern=None,
        custom_fn_templ=None,
        custom_sf_templ=None,
    ):
        """
        Search files for given root path and filename/folder pattern.

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        fn_templ = custom_fn_templ if custom_fn_templ else self.ft.fn_templ
        sf_templ = custom_sf_templ if custom_sf_templ else self.ft.sf_templ

        if custom_fn_templ or custom_sf_templ:
            self.fs = FileSearch(self.root_path, fn_templ, sf_templ)

        filenames = self.fs.search(fn_search_pattern, sf_search_pattern)

        return sorted(filenames, reverse=True)


def braces_to_re_groups(string):
    """
    Convert braces to character patterns defining regular expression groups.
    If any group name is repeated in the template string, a backreference
    is used for subsequent appearances.

    Parameters
    ----------
    string : str
        String with braces.

    Returns
    -------
    string : str
        String with regular expression groups.

    Examples
    --------
    >>> braces_to_re_groups("{year}-{month}-{day}")
    "(?P<year>.+)-(?P<month>.+)-(?P<day>.+)"
    >>> braces_to_re_groups("{year}-{month}-{day}_{year}-{month}-{day2}")
    "(?P<year>.+)-(?P<month>.+)-(?P<day>.+)_(?P=year)-(?P=month)-(?P<day2>.+)"
    """

    pattern = re.compile(r"{(.+?)}")
    seen = set()
    parts = pattern.split(string)

    for i in range(1, len(parts), 2):
        content = parts[i]

        if content in seen:
            parts[i] = f"(?P={content})"
        else:
            parts[i] = f"(?P<{content}>.+)"
            seen.add(content)

    return "".join(parts)


def datetime_wildcardify(
    dt,
    fmt,
    year=True,
    month=True,
    day=True,
    hour=True,
    minute=True,
    second=True,
    microsecond=True,
):
    """
    Convert datetime object to string following given format, but with
    wildcards for the user-defined timesteps.

    Parameters
    ----------
    dt : datetime.datetime
        Datetime object.
    fmt : str
        Format string.
    year : bool, optional
        Format the year as a wildcard (Default: True).
    month : bool, optional
        Format the month as a wildcard (Default: True).
    day : bool, optional
        Format the day as a wildcard (Default: True).
    hour : bool, optional
        Format the hour as a wildcard (Default: True).
    minute : bool, optional
        Format the minute as a wildcard (Default: True).
    second : bool, optional
        Format the second as a wildcard (Default: True).
    microsecond : bool, optional
        Format the microsecond as a wildcard (Default: True).

    Returns
    -------
    string : str
        String with wildcards.
    """
    if year:
        fmt = fmt.replace("%Y", "*").replace("%y", "*")
    if month:
        fmt = fmt.replace("%m", "*").replace("%b", "*").replace("%B", "*")
    if day:
        fmt = fmt.replace("%d", "*").replace("%j", "*")
    if hour:
        fmt = fmt.replace("%H", "*").replace("%I", "*").replace("%p", "*")
    if minute:
        fmt = fmt.replace("%M", "*")
    if second:
        fmt = fmt.replace("%S", "*")
    if microsecond:
        fmt = fmt.replace("%f", "*")

    return dt.strftime(fmt)


class ChronFiles(MultiFileHandler):
    """
    Managing chronological files with a date field in the filename.
    """

    def __init__(
        self,
        root_path,
        cls,
        fn_templ,
        sf_templ,
        cls_kwargs=None,
        err=True,
        fn_read_fmt=None,
        sf_read_fmt=None,
        fn_write_fmt=None,
        sf_write_fmt=None,
        cache_size=0,
    ):
        """
        Initialize ChronFiles class.

        Parameters
        ----------
        root_path : str
            Root path.
        cls : class
            Class reading/writing files.
        fn_templ : str
            Filename template (e.g. "{date}_ascat.nc").
        sf_templ : dict, optional
            Subfolder template defined as dictionary (default: None).
        cls_kwargs : dict, optional
            Class keyword arguments (default: None).
        err : bool, optional
            Set true if a file error should be re-raised instead of
            reporting a warning.
            Default: False
        fn_read_fmt : str or function, optional
            Filename format for read operation.
        sf_read_fmt : str or function, optional
            Subfolder format for read operation.
        fn_write_fmt : str or function, optional
            Filename format for write operation.
        sf_write_fmt : str or function, optional
            Subfolder format for write operation.
        cache_size : int, optional
            Number of files to keep in memory (default=0).
        """
        super().__init__(root_path, cls, fn_templ, sf_templ, cls_kwargs, err,
                         cache_size)

        self.fn_read_fmt = fn_read_fmt
        self.sf_read_fmt = sf_read_fmt
        self.fn_write_fmt = fn_write_fmt
        self.sf_write_fmt = sf_write_fmt

    def _fmt(self, *fmt_args, **fmt_kwargs):
        """
        Format filenames/filepaths.
        """
        if callable(self.fn_read_fmt):
            fn_read_fmt = self.fn_read_fmt(*fmt_args, **fmt_kwargs)
        else:
            fn_read_fmt = self.fn_read_fmt

        if callable(self.sf_read_fmt):
            sf_read_fmt = self.sf_read_fmt(*fmt_args, **fmt_kwargs)
        else:
            sf_read_fmt = self.sf_read_fmt

        if callable(self.fn_write_fmt):
            fn_write_fmt = self.fn_write_fmt(*fmt_args, **fmt_kwargs)
        else:
            fn_write_fmt = self.fn_write_fmt

        if callable(self.sf_write_fmt):
            sf_write_fmt = self.sf_write_fmt(*fmt_args, **fmt_kwargs)
        else:
            sf_write_fmt = self.sf_write_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt

    def _parse_date(self, filename, date_str, date_field):
        """
        Parse datetime from filename.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        timestamp : datetime
            File timestamp.
        """
        filename = Path(filename).name

        # add escape '\'
        templ = self.ft.fn_templ.replace("+", r"\+")

        # Replace braces surrounding date_fields in template with characters
        # to # define a named group in a regular expression
        pattern = braces_to_re_groups(templ)

        # replace each '*' character with a (?P<placeholder_i>.+) group
        for i in range(pattern.count("*")):
            pattern = pattern.replace("*", f"(?P<placeholder_{i}>.+)", 1)

        match = re.match(pattern, filename)

        # Then extract the date from the filename using the named group
        date = match.group(date_field)

        return datetime.strptime(date, date_str)

    def _merge_data(self, data):
        """
        Merge datasets after reading period. Needs to be overwritten
        by child class, otherwise data is returned as is.

        Parameters
        ----------
        data : list
            Data.

        Returns
        -------
        data : list
            Merged data.
        """
        return self.fid.merge(data)

    def search_date(self,
                    timestamp,
                    search_date_str="%Y%m%d*",
                    date_str="%Y%m%d",
                    date_field="date",
                    return_date=False):
        """
        Search files for given date.

        Parameters
        ----------
        timestamp : datetime
            Search date.
        search_str : str, optional
            Search date string used during file search (default: %Y%m%d*).
        date_str : str, optional
            Date field (default: %Y%m%d).
        date_field : str, optional
            Date field name (default: "date")
        return_date : bool, optional
            Return date parsed from filename (default: False).

        Returns
        -------
        filenames : list of str
            Filenames.
        dates : list of datetime
            Parsed date of filename (only returned if return_date=True).
        """
        fn_read_fmt, sf_read_fmt, _, _ = self._fmt(timestamp)
        fn_read_fmt[date_field] = timestamp.strftime(search_date_str)

        fs = FileSearch(self.root_path, self.ft.fn_templ, self.ft.sf_templ)
        filenames = sorted(fs.search(fn_read_fmt, sf_read_fmt))

        if return_date:
            dates = []
            for filename in filenames:
                dates.append(self._parse_date(filename, date_str, date_field))
            return filenames, dates
        else:
            return filenames

    def search_period(
            self,
            dt_start,
            dt_end,
            dt_delta=timedelta(days=1),
            search_date_str="%Y%m%d*",
            date_str="%Y%m%d",
            date_field="date",
    ):
        """
        Search files for time period.

        Parameters
        ----------
        dt_start : datetime
            Start datetime.
        dt_end : datetime
            End datetime.
        dt_delta : timedelta, optional
            Time delta used to jump through search date.
        search_str : str, optional
            Search date string used during file search (default: %Y%m%d*).
        date_str : str, optional
            Date field (default: %Y%m%d).
        date_field : str, optional
            Date field name (default: "date").

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        filenames = []

        for dt_cur in np.arange(dt_start, dt_end + dt_delta,
                                dt_delta).astype(datetime):
            files, dates = self.search_date(dt_cur,
                                            search_date_str,
                                            date_str,
                                            date_field,
                                            return_date=True)
            for f, dt in zip(files, dates):
                if f not in filenames and dt >= dt_start and dt < dt_end + dt_delta:
                    filenames.append(f)

        return filenames

    def read_period(
        self,
        dt_start,
        dt_end,
        dt_delta=timedelta(days=1),
        dt_buffer=timedelta(days=1),
        search_date_str="%Y%m%d*",
        date_str="%Y%m%d",
        date_field="date",
        **kwargs,
    ):
        """
        Read data for given interval.

        Parameters
        ----------
        dt_start : datetime
            Start datetime.
        dt_end : datetime
            End datetime.
        dt_delta : timedelta, optional
            Time delta used to jump through search date.
        dt_buffer : timedelta, optional
            Search buffer used to find files which could possibly contain
            data but would be left out because of dt_start.
        search_str : str, optional
            Search date string used during file search (default: %Y%m%d*).
        date_str : str, optional
            Date field (default: %Y%m%d).
        date_field : str, optional
            Date field name (default: "date").

        Returns
        -------
        data : dict, numpy.ndarray
            Data stored in file.
        """
        filenames = self.search_period(dt_start - dt_buffer, dt_end, dt_delta,
                                       search_date_str, date_str, date_field)

        data = []

        for filename in filenames:
            self._open(filename)
            d = self.fid.read_period(dt_start, dt_end, **kwargs)
            if d is not None:
                data.append(d)

        if data:
            data = self._merge_data(data)

        return data


class Csv:
    """
    Read and write single CSV file.
    """

    def __init__(self, filename, mode="r"):
        """
        Initialize Csv

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
        for substr in header.split("(")[2:]:
            d = []
            for substr2 in substr.split(","):
                if substr2.endswith(")"):
                    substr2 = substr2[:-1]
                if substr2.endswith("\n"):
                    substr2 = substr2[:-4]
                substr2 = substr2.strip()
                substr2 = substr2.strip("'")
                if substr2 == "":
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

    def read_period(self, dt_start, dt_end):
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
        data = self.read()

        subset = (data["date"] >= np.datetime64(dt_start)) & (
            data["date"] <= np.datetime64(dt_end))

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
        np.savetxt(self.filename, data, fmt="%s", header=header)

    @staticmethod
    def merge(data):
        """
        Merge data.

        Parameters
        ----------
        data : list of numpy.ndarray
            List of data.

        Returns
        -------
        data : numpy.ndarray
            Merged data.
        """
        return np.hstack(data)


class CsvFiles(ChronFiles):
    """
    Write CSV files.
    """

    def __init__(self, root_path):
        """
        Initialize CvsFileRW.

        Parameters
        ----------
        root_path : str
            Root path.
        """
        fn_templ = "prefix_{date}_{now}_postfix.csv"
        sf_templ = {"Y": "{year}", "M": "{month}"}

        super().__init__(root_path, Csv, fn_templ, sf_templ=sf_templ)

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
        fn_read_fmt = {"date": timestamp.strftime("%Y%m%d_%H%M%S"), "now": "*"}

        sf_read_fmt = {
            "Y": {
                "year": timestamp.strftime("%Y")
            },
            "M": {
                "month": timestamp.strftime("%m")
            },
        }

        fn_write_fmt = {
            "date": timestamp.strftime("%Y%m%d_%H%M%S"),
            "now": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        sf_write_fmt = sf_read_fmt

        return fn_read_fmt, sf_read_fmt, fn_write_fmt, sf_write_fmt
