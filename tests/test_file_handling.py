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
Test file handler.
"""

import unittest
import random
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta

import numpy as np
from numpy.testing import assert_array_equal

from ascat.file_handling import FilenameTemplate
from ascat.file_handling import FileSearch
from ascat.file_handling import MultiFileHandler
from ascat.file_handling import ChronFiles
from ascat.file_handling import Csv

class TestFilenameTemplate(unittest.TestCase):
    """
    Tests for FilenameTemplate class.
    """
    def setUp(self):
        root_path = tmpdir/"file_handler_test_data"
        fn_pattern = "{date}_*.{suffix}"
        sf_pattern = {"variables": "{variable}", "tiles": "{tile}"}
        self.template = FilenameTemplate(root_path, fn_pattern, sf_pattern)

    def test_template_property(self):
        """
        Test the template property.
        """

        self.assertEqual(str(self.template.template), str(tmpdir/"file_handler_test_data/{variable}/{tile}/{date}_*.{suffix}"))

    def test_build_filename(self):
        """
        Test the build_filename method.
        """

        fn_fmt = {"date": "20220101", "suffix": "csv"}
        sf_fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN01234"}}
        self.assertEqual(self.template.build_filename(fn_fmt, sf_fmt),
                         str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_*.csv"))

    def test_build_basename(self):
        """
        Test the build_basename method.
        """
        fmt = {"date": "20220101", "suffix": "csv"}
        self.assertEqual(self.template.build_basename(fmt), "20220101_*.csv")

    def test_build_subfolder(self):
        """
        Test the build_subfolder method.
        """
        fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN01234"}}
        self.assertEqual(self.template.build_subfolder(fmt), ["temperature", "EN01234"])


class TestFileSearch(unittest.TestCase):
    """
    Tests for FileSearch class.
    """
    def setUp(self):
        root_path = tmpdir/"file_handler_test_data"
        fn_pattern = "{date}_*.{suffix}"
        sf_pattern = {"variables": "{variable}", "tiles": "{tile}"}
        self.filesearch = FileSearch(root_path, fn_pattern, sf_pattern)

        # test these with wildcards as well
    def test_search(self):
        fn_fmt = {"date": "20220101", "suffix": "csv"}
        sf_fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN01234"}}
        recursive = False
        search_result = self.filesearch.search(fn_fmt, sf_fmt, recursive)
        expected_result = [str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv")]
        self.assertEqual(set(search_result), set(expected_result))

    def test_search_wc(self):
        fn_fmt = {"date": "202201*", "suffix": "csv"}
        sf_fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN012*"}}
        recursive = False
        search_result = self.filesearch.search(fn_fmt, sf_fmt, recursive)
        expected_result = [str(item) for item in
                           [tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01234/20220102_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01234/20220103_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01212/20220101_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01212/20220102_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01212/20220103_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01256/20220101_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01256/20220102_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01256/20220103_ascat.csv"]]
        self.assertEqual(set(search_result), set(expected_result))

    def test_isearch(self):
        fn_fmt = {"date": "20220101", "suffix": "csv"}
        sf_fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN01234"}}
        recursive = False
        search_result = self.filesearch.isearch(fn_fmt, sf_fmt, recursive)
        expected_result = iter([str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv")])
        self.assertEqual(set(search_result), set(expected_result))

    def test_isearch_wc(self):
        fn_fmt = {"date": "202201*", "suffix": "csv"}
        sf_fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN012*"}}
        recursive = False
        search_result = self.filesearch.isearch(fn_fmt, sf_fmt, recursive)
        expected_result = iter([str(item) for item in
                                [tmpdir/"file_handler_test_data/temperature/EN01212/20220101_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01256/20220101_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01212/20220102_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01234/20220102_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01256/20220102_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01212/20220103_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01234/20220103_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01256/20220103_ascat.csv"]])
        self.assertEqual(set(search_result), set(expected_result))

    def test_create_search_func(self):
        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {"variables": {"variable": arg2}, "tiles": {"tile": "EN01234"}}
            return fn_fmt, sf_fmt

        recursive = False
        custom_search_func = self.filesearch.create_search_func(custom_func, recursive)
        search_result = custom_search_func("20220101", "temperature")
        expected_result = [str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv")]
        self.assertEqual(set(search_result), set(expected_result))

    def test_create_search_func_wc(self):
        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {"variables": {"variable": arg2}, "tiles": {"tile": "EN012*"}}
            return fn_fmt, sf_fmt

        recursive = False
        custom_search_func = self.filesearch.create_search_func(custom_func, recursive)
        search_result = custom_search_func("202201*", "temperature")
        expected_result = [str(item) for item in
                           [tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01234/20220102_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01234/20220103_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01212/20220101_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01212/20220102_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01212/20220103_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01256/20220101_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01256/20220102_ascat.csv",
                            tmpdir/"file_handler_test_data/temperature/EN01256/20220103_ascat.csv"]]
        self.assertEqual(set(search_result), set(expected_result))

    def test_create_isearch_func(self):
        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {"variables": {"variable": arg2}, "tiles": {"tile": "EN01234"}}
            return fn_fmt, sf_fmt

        recursive = False
        custom_isearch_func = self.filesearch.create_isearch_func(custom_func, recursive)
        search_result = custom_isearch_func("20220101", "temperature")
        expected_result = iter([str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv")])
        self.assertEqual(set(search_result), set(expected_result))

    def test_create_isearch_func_wc(self):
        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {"variables": {"variable": arg2}, "tiles": {"tile": "EN012*"}}
            return fn_fmt, sf_fmt

        recursive = False
        custom_isearch_func = self.filesearch.create_isearch_func(custom_func, recursive)
        search_result = custom_isearch_func("202201*", "temperature")
        expected_result = iter([str(item) for item in
                                [tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01234/20220102_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01234/20220103_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01212/20220101_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01212/20220102_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01212/20220103_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01256/20220101_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01256/20220102_ascat.csv",
                                 tmpdir/"file_handler_test_data/temperature/EN01256/20220103_ascat.csv"]])
        self.assertEqual(set(search_result), set(expected_result))


class TestChronFiles(unittest.TestCase):
    """
    Tests for ChronFiles class, as well as MultiFileHandler
    """
    def setUp(self):
        root_path = str(tmpdir/"file_handler_test_data")
        cls = Csv
        fn_templ = "{date}_ascat.csv"
        sf_templ = {"variables": "{variable}", "tiles": "{tile}"}
        cls_kwargs = {"mode": "r"}
        err = True
        fn_read_fmt = lambda timestamp: {"date": timestamp.strftime("%Y%m%d")}
        sf_read_fmt = {"variables": {"variable": "temperature"}, "tiles": {"tile": "EN01234"}}
        self.chron_files = ChronFiles(root_path, cls, fn_templ, sf_templ, cls_kwargs, err,
                                              fn_read_fmt, sf_read_fmt)

    def test_search_date(self):
        timestamp = datetime(2022, 1, 1)
        filenames = self.chron_files.search_date(timestamp, date_str="%Y%m%d",
                                                 date_fields=["date"])
        expected_filenames = [str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv")]
        self.assertEqual(filenames, expected_filenames)

    def test_search_period(self):
        dt_start = datetime(2022, 1, 1)
        dt_end = datetime(2022, 1, 3)
        filenames = self.chron_files.search_period(dt_start, dt_end, dt_delta=timedelta(days=1))
        expected_filenames = [str(tmpdir/"file_handler_test_data/temperature/EN01234/20220101_ascat.csv"),
                              str(tmpdir/"file_handler_test_data/temperature/EN01234/20220102_ascat.csv"),
                              str(tmpdir/"file_handler_test_data/temperature/EN01234/20220103_ascat.csv")]
        self.assertTrue(filenames)
        self.assertEqual(filenames, expected_filenames)

    def test_read_period(self):
        dt_start = datetime(2022, 1, 1, hour=12, minute=30)
        dt_end = datetime(2022, 1, 2, hour = 12, minute=30)
        data = self.chron_files.read_period(dt_start, dt_end, dt_delta=timedelta(days=1))
        minimum_date = data["date"].min()
        maximum_date = data["date"].max()
        self.assertTrue(minimum_date >= dt_start)
        self.assertTrue(maximum_date <= dt_end)


class TestChronFiles_multidate(unittest.TestCase):
    """
    Tests for ChronFiles class with two-date filenames, as well as MultiFileHandler
    """
    def setUp(self):
        root_path = (tmpdir/"file_handler_test_data")
        cls = Csv
        fn_templ = "ascat_{dt1}-{dt2}.csv"
        sf_templ = {"variables": "{variable}", "tiles": "{tile}"}
        cls_kwargs = {"mode": "r"}
        err = True
        fn_read_fmt = lambda timestamp1, timestamp2: {"dt1": timestamp1.strftime("%Y%m%d_%H%M%S"), "dt2": timestamp2.strftime("%Y%m%d_%H%M%S")}
        sf_read_fmt = {"variables": {"variable": "precipitation"}, "tiles": {"tile": "EN01234"}}
        self.chron_files = ChronFiles(root_path, cls, fn_templ, sf_templ, cls_kwargs, err,
                                                fn_read_fmt, sf_read_fmt)

    def test_search_date(self):
        timestamp = datetime(2022, 1, 1)
        filenames = self.chron_files.search_date(timestamp, date_str="%Y%m%d_%H%M%S",
                                                 date_fields=["dt1", "dt2"])
        unique_dates = set()
        for fname in filenames:
            fname_datetime = self.chron_files._parse_date(fname, "%Y%m%d_%H%M%S", ["dt1"])[0]
            fname_date = datetime(fname_datetime.year, fname_datetime.month, fname_datetime.day)
            unique_dates.add(fname_date)

        expected_dates = {datetime(2022, 1, 1)}
        self.assertTrue(filenames)
        self.assertEqual(unique_dates, expected_dates)

    def test_search_period(self):
        dt_start = datetime(2022, 1, 1)
        dt_end = datetime(2022, 1, 3)
        filenames = self.chron_files.search_period(dt_start, dt_end, dt_delta=timedelta(days=1),
                                                   date_str="%Y%m%d_%H%M%S", date_fields=["dt1", "dt2"])
        unique_dates = set()
        for fname in filenames:
            fname_datetime = self.chron_files._parse_date(fname, "%Y%m%d_%H%M%S", ["dt1"])[0]
            fname_date = datetime(fname_datetime.year, fname_datetime.month, fname_datetime.day)
            unique_dates.add(fname_date)

        expected_dates = {datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)}
        self.assertTrue(filenames)
        self.assertEqual(unique_dates, expected_dates)

    def test_merge(self):
        timestamp1 = datetime(2022, 1, 1, hour=0, minute=0)
        timestamp2 = datetime(2022, 1, 1, hour=12, minute=0)
        timestamp3 = datetime(2022, 1, 2, hour=0, minute=0)

        data1 = self.chron_files.read_period(timestamp1, timestamp2, dt_delta=timedelta(hours=1),
                                             date_str="%Y%m%d_%H%M%S", date_fields=["dt1", "dt2"])
        data2 = self.chron_files.read_period(timestamp2, timestamp3, dt_delta=timedelta(hours=1),
                                                date_str="%Y%m%d_%H%M%S", date_fields=["dt1", "dt2"])
        data3 = self.chron_files.read_period(timestamp1, timestamp3, dt_delta=timedelta(hours=1),
                                                date_str="%Y%m%d_%H%M%S", date_fields=["dt1", "dt2"])
        data_merged = np.unique(self.chron_files._merge_data([data1, data2]))

        assert_array_equal(data3, data_merged)


    def test_read_period(self):
        dt_start = datetime(2022, 1, 1, hour=12, minute=30)
        dt_end = datetime(2022, 1, 3, hour = 12, minute=30)
        data = self.chron_files.read_period(dt_start, dt_end, dt_delta=timedelta(days=1),
                                            date_str="%Y%m%d_%H%M%S", date_fields=["dt1", "dt2"])
        minimum_date = data["date"].min().astype(datetime)
        maximum_date = data["date"].max().astype(datetime)
        self.assertTrue(minimum_date >= dt_start)
        self.assertTrue(minimum_date.strftime("%Y%m%d") == dt_start.strftime("%Y%m%d"))
        self.assertTrue(maximum_date <= dt_end)
        self.assertTrue(maximum_date.strftime("%Y%m%d") == dt_end.strftime("%Y%m%d"))

    def test_read_period_caching(self):
        self.chron_files.cache_size = 10
        self.chron_files.cache = {}
        dt_start = datetime(2022, 1, 1, hour=12, minute=30)
        dt_end = datetime(2022, 1, 3, hour = 12, minute=30)
        filenames = self.chron_files.search_period(dt_start, dt_end, dt_delta=timedelta(days=1),
                                                    date_str="%Y%m%d_%H%M%S", date_fields=["dt1", "dt2"])

        start_time = time.time()
        data = self.chron_files.read_file(filenames[0], cls_kwargs={})
        end_time = time.time()
        first_read_time = end_time - start_time
        minimum_date = data["date"].min()
        maximum_date = data["date"].max()
        self.assertTrue(minimum_date >= dt_start)
        self.assertTrue(maximum_date <= dt_end)

        # read again, should be cached
        start_time = time.time()
        data = self.chron_files.read_file(filenames[0], cls_kwargs={})
        end_time = time.time()
        second_read_time = end_time - start_time

        minimum_date = data["date"].min()
        maximum_date = data["date"].max()
        self.assertTrue(minimum_date >= dt_start)
        self.assertTrue(maximum_date <= dt_end)
        self.assertTrue(second_read_time < first_read_time)


if __name__ == '__main__':
    # generate test data
    with TemporaryDirectory() as temp_directory:
        tmpdir = Path(temp_directory)
        for num in ["01", "02", "03"]:
            for tile in ["EN01212", "EN01234", "EN01256"]:
                (tmpdir/f"file_handler_test_data/temperature/{tile}").mkdir(parents=True, exist_ok=True)
                file = Csv(tmpdir/f"file_handler_test_data/temperature/{tile}/202201{num}_ascat.csv", mode="w")
                file_dates = [datetime.strptime(f"202201{num}", "%Y%m%d") + timedelta(seconds=(3600*i + seconds))\
                            for i, seconds in enumerate(range(3500, 3700))]
                file_temps = random.choices(range(20,35), k=15)
                dtype = np.dtype([("date", "datetime64[s]"), ("temperature", "float32")])
                write_data = np.array(list(zip(file_dates, file_temps)), dtype=dtype)
                file.write(write_data)

        for num in ["01", "02", "03"]:
            for tile in ["EN01212", "EN01234", "EN01256"]:
                prev_time = datetime.strptime(f"202201{num}0000", "%Y%m%d%H%M")
                next_day = prev_time + timedelta(days=1)
                while prev_time < next_day:
                    (tmpdir/f"file_handler_test_data/precipitation/{tile}").mkdir(parents=True, exist_ok=True)
                    file_dates = [prev_time + timedelta(seconds=i)\
                                for i in range(random.randrange(120,180))\
                                if (random.randrange(10) < 7)]
                    for i, date in enumerate(file_dates):
                        if date >= next_day:
                            prev_time = date
                            del(file_dates[i+1:])
                    if len(file_dates) < 2:
                        continue

                    d1, d2 = file_dates[0].strftime("%Y%m%d"), file_dates[-2].strftime("%Y%m%d")
                    t1, t2 = file_dates[0].strftime("%H%M%S"), file_dates[-2].strftime("%H%M%S")
                    filename = tmpdir/f"file_handler_test_data/precipitation/{tile}/ascat_{d1}_{t1}-{d2}_{t2}.csv"
                    file = Csv(filename, mode="w")
                    file_precips = random.choices(range(0,5), k=len(file_dates)-1)
                    dtype = np.dtype([("date", "datetime64[s]"), ("precipitation", "float32")])
                    write_data = np.array(list(zip(file_dates[:-1], file_precips)), dtype=dtype)
                    file.write(write_data)
                    prev_time = file_dates[-1]

        unittest.main(exit=False)
