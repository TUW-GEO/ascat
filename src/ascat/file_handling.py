# Copyright (c) 2025, TU Wien
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
from collections import defaultdict

import numpy as np

from tqdm import tqdm
from tqdm.dask import TqdmCallback

from dask.delayed import delayed
from dask.base import compute

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
        return glob.glob(
            self.file_templ.build_filename(fn_fmt, sf_fmt),
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
        return glob.iglob(
            self.file_templ.build_filename(fn_fmt, sf_fmt),
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
        self._open(filename)

        data = None
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

    def _parse_date(self, filename, date_field, date_field_fmt):
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
        # escape special characters in the template string
        escaped_template = re.escape(self.ft.fn_templ)

        # replace escaped curly braces with capturing groups
        pattern = re.sub(r'\\{(.*?)\\}', r'(?P<\1>.*?)', escaped_template)

        match = re.match(pattern, Path(filename).name)
        date_substring = match.group(date_field)
        date = datetime.strptime(date_substring, date_field_fmt)

        return date

    def _merge_data(self, data, **kwargs):
        """
        Merge datasets after reading period. Needs to be overwritten
        by child class, otherwise data is returned as is.

        Parameters
        ----------
        data : list
            Data.
        **kwargs : dict
            Additional keyword arguments to the fid's merge method.

        Returns
        -------
        data : list
            Merged data.
        """
        return self.fid.merge(data, **kwargs)

    def search_date(self,
                    timestamp,
                    search_date_fmt="%Y%m%d*",
                    date_field="date",
                    date_field_fmt="%Y%m%d",
                    return_date=False,
                    **fmt_kwargs):
        """
        Search files for given date.

        Parameters
        ----------
        timestamp : datetime
            Search date.
        search_date_fmt : str, optional
            Search date string format used during file search (default: %Y%m%d*).
        date_field : str, optional
            Date field name (default: "date")
        date_field_format : str, optional
            Date field string format (default: %Y%m%d).
        return_date : bool, optional
            Return date parsed from filename (default: False).

        Returns
        -------
        filenames : list of str
            Filenames.
        dates : list of datetime
            Parsed date of filename (only returned if return_date=True).
        """
        fn_read_fmt, sf_read_fmt, _, _ = self._fmt(timestamp, **fmt_kwargs)
        fn_read_fmt[date_field] = timestamp.strftime(search_date_fmt)

        fs = FileSearch(self.root_path, self.ft.fn_templ, self.ft.sf_templ)
        def key_func(x): return self._parse_date(x, date_field, date_field_fmt)
        filenames = sorted(fs.search(fn_read_fmt, sf_read_fmt), key=key_func)

        if return_date:
            dates = []
            for filename in filenames:
                dates.append(
                    self._parse_date(filename, date_field, date_field_fmt))
            return filenames, dates
        else:
            return filenames

    def search_period(
        self,
        dt_start,
        dt_end,
        dt_delta=timedelta(days=1),
        search_date_fmt="%Y%m%d*",
        date_field="date",
        date_field_fmt="%Y%m%d",
        end_inclusive=True,
        **fmt_kwargs
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
        search_fmt : str, optional
            Search date string format used during file search (default: %Y%m%d*).
        date_field : str, optional
            Date field name (default: "date").
        date_field_fmt : str, optional
            Date field string format (default: %Y%m%d).
        end_inclusive : bool, optional
            Include files from a dt_delta length period beyond dt_end if True
            (default: False).

        Returns
        -------
        filenames : list of str
            Filenames.
        """
        filenames = []

        dt_end = dt_end + dt_delta if end_inclusive else dt_end

        for dt_cur in np.arange(dt_start, dt_end, dt_delta).astype(datetime):
            files, dates = self.search_date(
                dt_cur,
                search_date_fmt=search_date_fmt,
                date_field=date_field,
                date_field_fmt=date_field_fmt,
                return_date=True,
                **fmt_kwargs,
            )
            for f, dt in zip(files, dates):
                if f not in filenames and dt >= dt_start and dt < dt_end:
                    filenames.append(f)

        return filenames

    def read_period(
        self,
        dt_start,
        dt_end,
        dt_delta=timedelta(days=1),
        dt_buffer=timedelta(days=1),
        search_date_fmt="%Y%m%d*",
        date_field="date",
        date_field_fmt="%Y%m%d",
        end_inclusive=True,
        fmt_kwargs={},
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
        search_date_fmt : str, optional
            Search date string format used during file search (default: %Y%m%d*).
        date_field : str, optional
            Date field name (default: "date").
        date_field_fmt : str, optional
            Date field string format (default: %Y%m%d).

        Returns
        -------
        data : dict, numpy.ndarray
            Data stored in file.
        """
        filenames = self.search_period(dt_start - dt_buffer, dt_end, dt_delta,
                                       search_date_fmt, date_field,
                                       date_field_fmt, end_inclusive, **fmt_kwargs)

        data = []

        for filename in filenames:
            self._open(filename)
            d = self.fid.read_period(dt_start, dt_end, **kwargs)
            if d is not None:
                data.append(d)

        if data:
            data = self._merge_data(data)

        return data

class Filenames:
    """
    A class to handle operations on multiple filenames.

    This class provides methods for reading from, writing to, and merging data from multiple files.
    """

    def __init__(self, filenames):
        """
        Initialize Filenames.

        Parameters
        ----------
        filenames : str, Path, or list
            File path(s) to be handled.
        """
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]
        elif not isinstance(filenames, list):
            raise ValueError("filenames must be a string or list of strings.")

        self.filenames = [Path(f) for f in filenames]
        self.cache = {}

    def _read(self, filename, **kwargs):
        """
        Read data from a single file.

        This method should be implemented by subclasses.

        Parameters
        ----------
        filename : Path
            The file to read from.
        **kwargs : dict
            Additional keyword arguments for reading.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _merge(self, data, **kwargs):
        """
        Merge multiple data objects.

        This method should be implemented by subclasses.

        Parameters
        ----------
        data : list
            List of data objects to merge.
        **kwargs : dict
            Additional keyword arguments for merging.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _write(self, data, filename, **kwargs):
        """
        Write data to a single file.

        This method should be implemented by subclasses.

        Parameters
        ----------
        data : object
            The data to write.
        filename : Path
            The file to write to.
        **kwargs : dict
            Additional keyword arguments for writing.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def reprocess(self,
                  out_dir,
                  func,
                  parallel=False,
                  print_progress=False,
                  read_kwargs=None,
                  **write_kwargs):
        """
        Reprocess data from all files through `func`, writing the results to `out_dir`.
        Assumes that if any files have the same name, they should be merged.

        Parameters
        ----------
        out_dir : Path
            Directory to write the output files. This will be prepended to the filenames.
        func : function
            The function to apply to the data before writing out.
        parallel : bool, optional
            Whether to process the data in parallel (default: False).
        **kwargs : dict
            Additional keyword arguments for writing.
        """
        read_kwargs = read_kwargs or {}
        if parallel:
            read_ = delayed(self._read)
            getattr_ = delayed(getattr)
            func_ = delayed(func)
            merge_ = delayed(self._merge)
        else:
            read_ = self._read
            getattr_ = getattr
            func_ = func
            merge_ = self._merge

        filenames = self.filenames

        name_to_paths = defaultdict(list)
        for path in filenames:
            name_to_paths[path.name].append(path)

        out_filenames = [out_dir / name for name in name_to_paths]
        out_path_groups = list(name_to_paths.values())

        if print_progress:
            out_path_groups = tqdm(out_path_groups)
            out_path_groups.set_description("Opening files...")

        data = [merge_([func_(read_(f, **read_kwargs)) for f in paths])
                for paths in out_path_groups]

        self.filenames = out_filenames

        self.write(data, parallel=parallel, print_progress=print_progress, **write_kwargs)

    def write(self, data, parallel=False, print_progress=False, **kwargs):
        """
        Write data to file.

        If there's only one filename in `self.filenames`, write provided data to that file.
        If there is more than one filename, write each element of the provided data list
        to the corresponding filename.

        Parameters
        ----------
        data :  list of objects
            The data to write. Should be a list with the same length as self.filenames,
            where each element is the data to be written to the corresponding filename.
        """
        if len(self.filenames) == 1 and not isinstance(data, list):
            data = [data]


        if len(data) == len(self.filenames):
            if parallel:
                write_ = delayed(self._write)
                writers = [write_(d, f, **kwargs) for d, f in zip(data, self.filenames)]
                if print_progress:
                    with TqdmCallback(desc="Writing cells to disk...", total=len(writers)):
                        compute(writers, scheduler="processes")
                else:
                    compute(writers, scheduler="processes")

            else:
                if print_progress:
                    data = tqdm(data)
                    data.set_description("Writing cells to disk...")
                for d, f in zip(data, self.filenames):
                    self._write(d, f, **kwargs)
        else:
            # Special case when the data object meant to be written to a single filename is a list
            raise ValueError("Number of data objects must match number of filenames.")

        return

    def read(self, parallel=False, closer_attr=None, **kwargs):
        """
        Read all data from files.

        Returns
        -------
        object
            Merged data from all files.
        """
        if parallel:
            read_ = delayed(self._read)
            getattr_ = delayed(getattr)
        else:
            read_ = self._read
            getattr_ = getattr

        data = [read_(f, **kwargs) for f in self.filenames]
        if closer_attr is not None:
            closers = [getattr_(d, closer_attr) for d in data if d is not None]

        if parallel:
            data = compute(data, scheduler="processes")[0]
            if closer_attr is not None:
                closers = compute(closers)[0]

        data = self.merge(data)

        if closer_attr is not None:
            return data, closers

        return data

    def iter_read(self, print_progress=False, **kwargs):
        """
        Iterate over all files and yield data.

        Yields
        ------
        object
            Data read from each file.
        """
        if print_progress:
            filenames = tqdm(self.filenames)
        else:
            filenames = self.filenames

        size = 0
        for filename in filenames:
            if print_progress:
                filenames.set_description(f"Opening {Path(filename).name}, total {size} bytes...")
            data = self._read(filename, **kwargs)
            size += self._nbytes(data)
            yield data

    def iter_read_nbytes(self, max_nbytes, print_progress=False, **kwargs):
        """
        Iterate over all files and yield data until the specified number of bytes is reached.
        If `_read` returns dask objects, they are computed (in parallel) before merging the data.
        """
        size = 0
        data_list = []
        for data in self.iter_read(print_progress, **kwargs):
            data_size = self._nbytes(data)
            size += data_size
            if size > max_nbytes and size > data_size:
                if print_progress:
                    print(f"Opened {size} bytes, reading and merging data...")

                    with TqdmCallback(desc="Reading data..."):
                        out_data = compute(*[d for d in data_list],
                                           scheduler="processes")
                    print("Merging data...")
                    out_data = self.merge(out_data)
                else:
                    out_data = self.merge(compute(*[d for d in data_list],
                                                scheduler="processes"))
                yield out_data
                size = data_size
                data_list = [data]
            else:
                data_list.append(data)
        if data_list:
            if print_progress:
                print("All source files opened, reading and merging remaining data...")
            yield self.merge(compute(*[d for d in data_list], scheduler="processes"))

    @staticmethod
    def _nbytes(data):
        """
        Returns size of data object in bytes.
        """
        raise NotImplementedError

    def merge(self, data):
        """
        Merge data from multiple data objects.

        Parameters
        ----------
        data : list
            List of data objects.

        Returns
        -------
        object
            Merged data, or None if the input list is empty.
        """
        if len(data) > 1:
            data = self._merge(data)
        elif len(data) == 1:
            data = data[0]
        else:
            data = None

        return data

    def close(self):
        """
        Close file(s).

        This method can be overridden in subclasses if necessary.
        """
        pass

    @staticmethod
    def _multi_file_closer(closers):
        for closer in closers:
            closer()
        return


class CsvFile(Filenames):
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
        self.mode = mode
        super().__init__(filename)

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

    def _read(self, filename):
        """
        Read data from CSV file.

        Parameters
        ----------
        filename : str
            Filename.
        """
        with open(filename) as fid:
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

    def _write(self, data, filename):
        """
        Write data to CSV file.

        Parameters
        ----------
        data : numpy.ndarray
            Data.
        """
        header = data.dtype.__repr__()
        np.savetxt(filename, data, fmt="%s", header=header)

    @staticmethod
    def _merge(data):
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

        super().__init__(root_path, CsvFile, fn_templ, sf_templ=sf_templ)

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
