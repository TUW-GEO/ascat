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
Test file handler.
"""

import shutil
import unittest
import random
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from ascat.file_handling import FilenameTemplate
from ascat.file_handling import FileSearch
from ascat.file_handling import ChronFiles
from ascat.file_handling import CsvFile
from ascat.file_handling import CsvFiles


def generate_test_data():
    """
    Generate fake test data.
    """
    tmpdir = Path(tempfile.mkdtemp())

    for num in ["01", "02", "03"]:
        for tile in ["EN01212", "EN01234", "EN01256"]:
            folder = tmpdir / "temperature" / tile
            folder.mkdir(parents=True, exist_ok=True)

            filename = tmpdir / "temperature" / tile / f"202201{num}_ascat.csv"

            csv_file = CsvFile(filename, mode="w")
            file_dates = [
                datetime.strptime(f"202201{num}", "%Y%m%d") +
                timedelta(seconds=(3600 * i + seconds))
                for i, seconds in enumerate(range(3500, 3700))
            ]

            file_temps = random.choices(range(20, 35), k=15)
            dtype = np.dtype([("date", "datetime64[s]"),
                              ("temperature", "float32")])
            write_data = np.array(
                list(zip(file_dates, file_temps)), dtype=dtype)
            csv_file.write(write_data)

    for num in ["01", "02", "03"]:
        for tile in ["EN01212", "EN01234", "EN01256"]:
            prev_time = datetime.strptime(f"202201{num}0000", "%Y%m%d%H%M")
            next_day = prev_time + timedelta(days=1)

            folder = tmpdir / "precipitation" / tile
            folder.mkdir(parents=True, exist_ok=True)

            while prev_time < next_day:
                file_dates = [
                    prev_time + timedelta(seconds=i)
                    for i in range(random.randrange(120, 180))
                    if (random.randrange(10) < 7)
                ]
                for i, date in enumerate(file_dates):
                    if date >= next_day:
                        prev_time = date
                        del (file_dates[i + 1:])

                if len(file_dates) < 2:
                    continue

                d1, d2 = file_dates[0].strftime(
                    "%Y%m%d"), file_dates[-2].strftime("%Y%m%d")
                t1, t2 = file_dates[0].strftime(
                    "%H%M%S"), file_dates[-2].strftime("%H%M%S")
                filename = folder / Path(f"ascat_{d1}_{t1}-{d2}_{t2}.csv")

                csv_file = CsvFile(filename, mode="w")
                file_precips = random.choices(
                    range(0, 5), k=len(file_dates) - 1)
                dtype = np.dtype([("date", "datetime64[s]"),
                                  ("precipitation", "float32")])
                write_data = np.array(
                    list(zip(file_dates[:-1], file_precips)), dtype=dtype)
                csv_file.write(write_data)
                prev_time = file_dates[-1]

    return tmpdir


class CustomTestCase(unittest.TestCase):
    """
    Custom test case generating and deleting test data.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = generate_test_data()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)


class TestFilenameTemplate(unittest.TestCase):
    """
    Tests for FilenameTemplate class.
    """

    def setUp(self):
        """
        Setup test.
        """
        self.tmpdir = Path(tempfile.mkdtemp())
        fn_pattern = "{date}_*.{suffix}"
        sf_pattern = {"variables": "{variable}", "tiles": "{tile}"}
        self.template = FilenameTemplate(self.tmpdir, fn_pattern, sf_pattern)

    def test_template_property(self):
        """
        Test the template property.
        """
        self.assertEqual(
            str(self.template.template),
            str(self.tmpdir / "{variable}" / "{tile}" / "{date}_*.{suffix}"))

    def test_build_filename(self):
        """
        Test the build_filename method.
        """
        fn_fmt = {"date": "20220101", "suffix": "csv"}
        sf_fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN01234"
            }
        }
        self.assertEqual(
            self.template.build_filename(fn_fmt, sf_fmt),
            str(self.tmpdir / "temperature" / "EN01234" / "20220101_*.csv"))

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
        fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN01234"
            }
        }
        self.assertEqual(
            self.template.build_subfolder(fmt), ["temperature", "EN01234"])


class TestFileSearch(CustomTestCase):
    """
    Tests for FileSearch class.
    """

    def setUp(self):
        """
        Setup test.
        """
        fn_pattern = "{date}_*.{suffix}"
        sf_pattern = {"variables": "{variable}", "tiles": "{tile}"}
        self.filesearch = FileSearch(self.tmpdir, fn_pattern, sf_pattern)

    def test_search(self):
        """
        Test search.
        """
        fn_fmt = {"date": "20220101", "suffix": "csv"}
        sf_fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN01234"
            }
        }
        recursive = False
        search_result = self.filesearch.search(fn_fmt, sf_fmt, recursive)
        expected_result = [
            str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv")
        ]
        self.assertEqual(search_result, expected_result)

    def test_search_wc(self):
        """
        Test search with wildcard.
        """
        fn_fmt = {"date": "202201*", "suffix": "csv"}
        sf_fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN012*"
            }
        }
        recursive = False
        search_result = self.filesearch.search(fn_fmt, sf_fmt, recursive)
        expected_result = [
            str(item) for item in [
                self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv",
                self.tmpdir / "temperature" / "EN01234" / "20220102_ascat.csv",
                self.tmpdir / "temperature" / "EN01234" / "20220103_ascat.csv",
                self.tmpdir / "temperature" / "EN01212" / "20220101_ascat.csv",
                self.tmpdir / "temperature" / "EN01212" / "20220102_ascat.csv",
                self.tmpdir / "temperature" / "EN01212" / "20220103_ascat.csv",
                self.tmpdir / "temperature" / "EN01256" / "20220101_ascat.csv",
                self.tmpdir / "temperature" / "EN01256" / "20220102_ascat.csv",
                self.tmpdir / "temperature" / "EN01256" / "20220103_ascat.csv"
            ]
        ]
        self.assertEqual(sorted(search_result), sorted(expected_result))

    def test_isearch(self):
        """
        Test isearch.
        """
        fn_fmt = {"date": "20220101", "suffix": "csv"}
        sf_fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN01234"
            }
        }
        recursive = False
        search_result = self.filesearch.isearch(fn_fmt, sf_fmt, recursive)
        expected_result = iter(
            [str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv")])

        self.assertEqual(list(search_result), list(expected_result))

    def test_isearch_wc(self):
        """
        Test isearch with wildcard.
        """
        fn_fmt = {"date": "202201*", "suffix": "csv"}
        sf_fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN012*"
            }
        }
        recursive = False
        search_result = self.filesearch.isearch(fn_fmt, sf_fmt, recursive)
        expected_result = iter([
            str(item) for item in [
                self.tmpdir / "temperature" /"EN01212"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220103_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220103_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220103_ascat.csv"
            ]
        ])
        self.assertEqual(sorted(search_result), sorted(expected_result))

    def test_create_search_func(self):
        """
        Test custom search function.
        """

        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {
                "variables": {
                    "variable": arg2
                },
                "tiles": {
                    "tile": "EN01234"
                }
            }
            return fn_fmt, sf_fmt

        recursive = False
        custom_search_func = self.filesearch.create_search_func(
            custom_func, recursive)
        search_result = custom_search_func("20220101", "temperature")
        expected_result = [
            str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv")
        ]
        self.assertEqual(search_result, expected_result)

    def test_create_search_func_wc(self):
        """
        Test custom search function with wildcard.
        """

        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {
                "variables": {
                    "variable": arg2
                },
                "tiles": {
                    "tile": "EN012*"
                }
            }
            return fn_fmt, sf_fmt

        recursive = False
        custom_search_func = self.filesearch.create_search_func(
            custom_func, recursive)
        search_result = custom_search_func("202201*", "temperature")
        expected_result = [
            str(item) for item in [
                self.tmpdir / "temperature" /"EN01234"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220103_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220103_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220103_ascat.csv"
            ]
        ]
        self.assertEqual(sorted(search_result), sorted(expected_result))

    def test_create_isearch_func(self):
        """
        Test custom isearch function with wildcard.
        """

        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {
                "variables": {
                    "variable": arg2
                },
                "tiles": {
                    "tile": "EN01234"
                }
            }
            return fn_fmt, sf_fmt

        recursive = False
        custom_isearch_func = self.filesearch.create_isearch_func(
            custom_func, recursive)
        search_result = custom_isearch_func("20220101", "temperature")
        expected_result = iter(
            [str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv")])

        self.assertEqual(list(search_result), list(expected_result))

    def test_create_isearch_func_wc(self):
        """
        Test custom isearch function with wildcard.
        """

        def custom_func(arg1, arg2):
            fn_fmt = {"date": arg1, "suffix": "csv"}
            sf_fmt = {
                "variables": {
                    "variable": arg2
                },
                "tiles": {
                    "tile": "EN012*"
                }
            }
            return fn_fmt, sf_fmt

        recursive = False
        custom_isearch_func = self.filesearch.create_isearch_func(
            custom_func, recursive)
        search_result = custom_isearch_func("202201*", "temperature")
        expected_result = iter([
            str(item) for item in [
                self.tmpdir / "temperature" /"EN01234"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01234"/ "20220103_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01212"/ "20220103_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220101_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220102_ascat.csv",
                self.tmpdir / "temperature" /"EN01256"/ "20220103_ascat.csv"
            ]
        ])
        self.assertEqual(sorted(search_result), sorted(expected_result))


class TestChronFiles(CustomTestCase):
    """
    Tests for ChronFiles class, as well as MultiFileHandler
    """

    def setUp(self):
        """
        Setup test.
        """
        cls = CsvFile
        fn_templ = "{date}_ascat.csv"
        sf_templ = {"variables": "{variable}", "tiles": "{tile}"}
        cls_kwargs = None
        err = True
        fn_read_fmt = lambda timestamp: {"date": timestamp.strftime("%Y%m%d")}
        sf_read_fmt = {
            "variables": {
                "variable": "temperature"
            },
            "tiles": {
                "tile": "EN01234"
            }
        }
        self.chron_files = ChronFiles(self.tmpdir, cls, fn_templ, sf_templ,
                                      cls_kwargs, err, fn_read_fmt,
                                      sf_read_fmt)

    def test_search_date(self):
        """
        Test search date.
        """
        timestamp = datetime(2022, 1, 1)
        filenames = self.chron_files.search_date(
            timestamp, date_field_fmt="%Y%m%d", date_field="date")
        expected_filenames = [
            str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv")
        ]
        self.assertEqual(filenames, expected_filenames)

    def test_search_period(self):
        """
        Test search period.
        """
        dt_start = datetime(2022, 1, 1)
        dt_end = datetime(2022, 1, 3)
        filenames = self.chron_files.search_period(
            dt_start, dt_end, dt_delta=timedelta(days=1))
        expected_filenames = [
            str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv"),
            str(self.tmpdir / "temperature" / "EN01234" / "20220102_ascat.csv"),
            str(self.tmpdir / "temperature" / "EN01234" / "20220103_ascat.csv")
        ]
        self.assertTrue(filenames)
        self.assertEqual(filenames, expected_filenames)

    def test_search_period_exclusive(self):
        """
        Test search period.
        """
        dt_start = datetime(2022, 1, 1)
        dt_end = datetime(2022, 1, 3)
        filenames = self.chron_files.search_period(
            dt_start, dt_end, dt_delta=timedelta(days=1), end_inclusive=False)
        expected_filenames = [
            str(self.tmpdir / "temperature" / "EN01234" / "20220101_ascat.csv"),
            str(self.tmpdir / "temperature" / "EN01234" / "20220102_ascat.csv"),
        ]
        self.assertTrue(filenames)
        self.assertEqual(filenames, expected_filenames)

    def test_read_period(self):
        """
        Test read period.
        """
        dt_start = datetime(2022, 1, 1, hour=12, minute=30)
        dt_end = datetime(2022, 1, 2, hour=12, minute=30)
        data = self.chron_files.read_period(
            dt_start, dt_end, dt_delta=timedelta(days=1))

        self.assertTrue(data["date"].min() >= dt_start)
        self.assertTrue(data["date"].max() <= dt_end)


class TestCsvFiles(unittest.TestCase):
    """
    Tests for CsvFiles class.
    """

    def setUp(self):
        """
        Setup test.
        """
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv = CsvFiles(self.tmpdir)

    def test_read_write(self):
        """
        Test read/write CSV files.
        """
        dtype = np.dtype([("date", "datetime64[h]"),
                          ("temperature", "float32")])

        dates = np.arange("2000-01-01", "2000-01-10", dtype="datetime64[D]")

        tmp_data = {}

        for date in dates:
            dt = np.arange(
                date, date + np.timedelta64(1, "D"), dtype="datetime64[h]")
            temperature = random.choices(range(0, 40), k=dt.size)
            data = np.array(list(zip(dt, temperature)), dtype=dtype)

            self.csv.write(data, date.astype(datetime))
            tmp_data[date] = data

        for date in dates:
            data = self.csv.read(date.astype(datetime))
            np.testing.assert_array_equal(data, tmp_data[date])

    def tearDown(self):
        """
        Tear down test.
        """
        shutil.rmtree(self.tmpdir)


if __name__ == '__main__':
    unittest.main()
