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

import unittest
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from ascat.read_native.ragged_array_ts import CellFileCollection
from ascat.read_native.ragged_array_ts import SwathFileCollection
from ascat.read_native.ragged_array_ts import CellFileCollectionStack

from ascat.read_native.xarray_io import AscatH129Cell
from ascat.read_native.xarray_io import AscatSIG0Cell12500m

from ascat.read_native.xarray_io import AscatH129Swath
from ascat.read_native.xarray_io import AscatSIG0Swath6250m
from ascat.read_native.xarray_io import AscatSIG0Swath12500m

xr.set_options(display_max_rows=100)

TEST_DATA = Path("ascat_test_data")


class TestCellFileCollection(unittest.TestCase):
    """
    Test the merge function
    """

    def assertNpArrayEqual(self, d1, d2):
        try:
            np.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    def assertDataSetEqual(self, d1, d2):
        try:
            xr.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    # @classmethod
    # def setUpClass(cls):
    #     cls.ctg_old_fname = self.tmpdir / "contiguous_RA_old.nc"
    #     cls.idx_new_fname = self.tmpdir / "indexed_RA_new.nc"
    #     cls.idx_old_fname = self.tmpdir / "indexed_RA_old.nc"
    #     cls.ctg_new_fname = self.tmpdir / "contiguous_RA_new.nc"

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.tmpdir = Path(self.temporary_directory.name)
        self.cells_sig0_12500_idx = TEST_DATA / "hsaf" / "sig0" / "12500m" / "stack_cells" / "metop_c"

        #TODO make this data
        self.cells_h129_idx = TEST_DATA / "hsaf" / "h129" / "stack_cells"
        self.cells_h129_ctg = TEST_DATA / "hsaf" / "h129" / "merge_cells"

    def tearDown(self):
        self.temporary_directory.cleanup()
        # self.ctg_coll.close()

    def test_from_product_id(self):
        return
        h129 = CellFileCollection.from_product_id(
            self.cells_h129_idx,
            product_id="h129",
        )
        self.assertEqual(h129.ioclass, AscatH129Cell)

        sig0_12_5 = CellFileCollection.from_product_id(
            self.cells_sig0_12500_idx / "20221212000000_20221219000000",
            product_id="sig0_12.5",
        )

        self.assertEqual(sig0_12_5.ioclass, AscatSIG0Cell12500m)

        incorrect = CellFileCollection.from_product_id(
            self.cells_h129_idx,
            product_id="sig0_12.5",
        )

        self.assertEqual(incorrect.ioclass, AscatSIG0Cell12500m)

        # trying to read this data first raises a RuntimeWarning, since it can't find any files
        # for the requested names, then an AttributeError since it tries to read a None object.
        with self.assertRaises(AttributeError):
            with self.assertWarns(RuntimeWarning):
                incorrect.read(
                    start_dt=datetime(2021, 1, 1), end_dt=datetime(2021, 1, 2))

    def test_cells_in_collection(self):
        sig0_12_5 = CellFileCollection.from_product_id(
            self.cells_sig0_12500_idx / "20221212000000_20221219000000",
            product_id="sig0_12.5",
        )

        cells_in_collection = sig0_12_5.cells_in_collection

        expected_cells = [0, 1, 9]
        self.assertEqual(set(cells_in_collection), set(expected_cells))

    def test_read(self):
        coll = CellFileCollection.from_product_id(
            self.cells_sig0_12500_idx / "20221219000000_20230101000000",
            product_id="sig0_12.5",
        )

        cell_5m = coll.read(cell=9)

        # With a new cell size:
        cell_10m = coll.read(cell=9, new_grid=10)

        # the actual read cells should be exactly the same
        # the only difference will be on writing out
        self.assertDataSetEqual(cell_5m, cell_10m)

        id = coll.read(location_id=2148049)
        # assert something

    def test_create_cell_lookup(self):
        coll = CellFileCollection.from_product_id(
            self.cells_sig0_12500_idx / "20221219000000_20230101000000",
            product_id="sig0_12.5")
        self.assertIsNone(coll.cell_lut)
        coll.create_cell_lookup(10)
        self.assertIsNotNone(coll.cell_lut)
        self.assertEqual(set(coll.cell_lut[0]), {0, 1, 36, 37})

    def test_get_cell_path(self):
        source_dir = self.cells_sig0_12500_idx / "20221219000000_20230101000000"
        coll = CellFileCollection.from_product_id(
            source_dir, product_id="sig0_12.5")
        self.assertEqual(coll.get_cell_path(cell=0), source_dir / "0000.nc")
        with self.assertRaises(ValueError):
            # this should raise a value error since cell 2593 is not in the 12.5km FibGrid
            # with a cell size of 5km (max 2592)
            coll.get_cell_path(cell=2593)

        # however, cells within the possible range that simply don't have a file in the
        # folder should work just fine
        self.assertEqual(coll.get_cell_path(cell=100), source_dir / "0100.nc")

    def test_close(self):
        source_dir = self.cells_sig0_12500_idx / "20221219000000_20230101000000"
        coll = CellFileCollection.from_product_id(
            source_dir, product_id="sig0_12.5")
        self.assertIsNone(coll.fid)
        coll.read(cell=0)
        self.assertIsNotNone(coll.fid)
        coll.close()
        self.assertIsNone(coll.fid)


class TestCellFileCollectionStack(unittest.TestCase):

    def assertNanEqual(self, d1, d2):
        try:
            np.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.tmpdir = Path(self.temporary_directory.name)

        self.cells_sig0_12500_idx_metop_b = TEST_DATA / "hsaf" / "sig0" / "12500m" / "stack_cells" / "metop_b"
        self.cells_sig0_12500_idx_metop_c = TEST_DATA / "hsaf" / "sig0" / "12500m" / "stack_cells" / "metop_c"

        #TODO make this data
        self.cells_h129_idx = TEST_DATA / "hsaf" / "h129" / "stack_cells"
        self.cells_h129_ctg = TEST_DATA / "hsaf" / "h129" / "merge_cells"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_from_product_id(self):
        # return
        # h129 = CellFileCollectionStack.from_product_id(self.idx_data_sig0_h129,
        #                                           product_id="h129",)
        # self.assertEqual(h129.ioclass, AscatH129Cell)

        sig0_12_5 = CellFileCollectionStack.from_product_id(
            self.cells_sig0_12500_idx_metop_b,
            product_id="sig0_12.5",
        )

        self.assertEqual(sig0_12_5.ioclass, AscatSIG0Cell12500m)

        product_id_mismatch = CellFileCollectionStack.from_product_id(
            self.cells_sig0_12500_idx_metop_b,
            product_id="h129",
        )

        self.assertEqual(product_id_mismatch.ioclass, AscatH129Cell)

        # TODO currently this does not fail, but it should.
        # with self.assertRaises():
        # incorrect_data = product_id_mismatch.read(cell=0)

    def test_add_collection(self):
        stack = CellFileCollectionStack.from_product_id(
            self.cells_sig0_12500_idx_metop_b, product_id="sig0_12.5")

        self.assertEqual(set(stack.read(cell=1).sat_id.values), {4})

        # merge metop_b with metop_c
        stack.add_collection([self.cells_sig0_12500_idx_metop_c],
                             product_id="sig0_12.5")
        self.assertEqual(set(stack.read(cell=1).sat_id.values), {4, 5})

    def test_read(self):
        stack = CellFileCollectionStack.from_product_id(
            self.cells_sig0_12500_idx_metop_b, product_id="sig0_12.5")

        stack.read(cell=0)
        stack.read(location_id=2148049)
        stack.read(location_id=[1650078, 2148049])

    def test_merge_and_write(self):
        stack = CellFileCollectionStack.from_product_id(
            self.cells_sig0_12500_idx_metop_b, product_id="sig0_12.5")
        stack.merge_and_write(self.tmpdir / "5m", cells=[0, 1, 9])
        stack.merge_and_write(
            self.tmpdir / "10m", cells=[0, 1, 9], out_cell_size=10)

        # test that writing to different grid cell size produces expected output
        written_files_5m = {
            int(f.stem) for f in (self.tmpdir / "5m").glob("*.nc")
        }
        written_files_10m = {
            int(f.stem) for f in (self.tmpdir / "10m").glob("*.nc")
        }
        self.assertEqual(written_files_5m, {0, 1, 9})
        self.assertEqual(written_files_10m, {0, 4})

        # test merge_and_write without multiprocessing
        stack.merge_and_write(self.tmpdir / "5m", cells=[0, 1, 9], processes=1)


class TestSwathFileCollection(unittest.TestCase):

    def assertNanEqual(self, d1, d2):
        try:
            np.testing.assert_equal(d1, d2)
        except AssertionError as e:
            raise self.failureException(e)

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.tmpdir = Path(self.temporary_directory.name)

        self.idx_data = Path(
            "/home/charriso/test_cells/data-write/RADAR/hsaf/stack_cell_new/metop_a"
        )
        self.ref_idx_data = TEST_DATA / "hsaf" / "sig0/12500m/metop_c" / "stack_cells"
        self.ctg_data = Path(
            "/home/charriso/test_cells/data-write/RADAR/hsaf/stack_cell_merged_new/metop_a"
        )
        # self.swath_data = Path("/home/charriso/test_cells/data-write/RADAR/hsaf/metop_a")
        self.swath_data = TEST_DATA / "hsaf" / "h129" / "swaths"
        self.swath_data_sig0_6_25 = Path("")
        self.swath_data_sig0_12_5 = Path("")

    def tearDown(self):
        self.temporary_directory.cleanup()
        # self.ctg_coll.close()
        # self.idx_coll.close()

    def test_init(self):
        return
        a = SwathFileCollection.from_product_id(
            self.swath_data,
            product_id="h129",
            dask_scheduler="processes",
        )
        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)
        fname = a.get_filenames(start, end)
        outdir = Path(
            "/home/charriso/test_cells/data-write/RADAR/hsaf/tester/metop_a")
        from time import time

        # swath = xr.decode_cf(a.read(start, end, cell=0).load())
        # print(swath["backscatter"].attrs["missing_value"])
        # print(np.any(swath["backscatter"] == swath["backscatter"].attrs["missing_value"], axis=1).sum())
        # print(np.all(np.isin(swath["backscatter"], [swath["backscatter"].attrs["missing_value"], ]), axis=1).sum())
        # print(swath.sel(
        #     obs=~np.any(
        #         swath["backscatter"] == swath["backscatter"].attrs["missing_value"],
        #         axis=1
        #     )
        # ))
        # print(np.all(np.isnan(swath.backscatter), axis=1).sum())
        #########################################################
        st_time = time()
        a.stack(fname, outdir, processes=8)
        print(time() - st_time, "seconds to stack a week of swaths")

    def test_from_product_id(self):
        h129 = SwathFileCollection.from_product_id(
            self.swath_data,
            product_id="h129",
        )
        self.assertEqual(h129.ioclass, AscatH129Swath)

        sig0_6_25 = SwathFileCollection.from_product_id(
            self.swath_data_sig0_6_25,
            product_id="sig0_6.25",
        )
        self.assertEqual(sig0_6_25.ioclass, AscatSIG0Swath6250m)

        sig0_12_5 = SwathFileCollection.from_product_id(
            self.swath_data_sig0_12_5,
            product_id="sig0_12.5",
        )
        self.assertEqual(sig0_12_5.ioclass, AscatSIG0Swath12500m)

        # this works just fine because there are no checks to make sure the product_id actually
        # matches the data. (if those were possible, the product_id would be redundant)
        # However, it will fail at some point when trying to read the data, either because the
        # format for filenames is different and it won't find any files, or because the data
        # has different variables and it will try to read something that doesn't exist.
        incorrect = SwathFileCollection.from_product_id(
            self.swath_data,
            product_id="sig0_12.5",
        )
        self.assertEqual(incorrect.ioclass, AscatSIG0Swath12500m)
        self.assertEqual(
            incorrect.get_filenames(
                datetime(2021, 1, 12), datetime(2021, 1, 13)), [])

        # trying to read this data first raises a RuntimeWarning, since it can't find any files
        # for the requested names, then an AttributeError since it tries to read a None object.
        with self.assertRaises(ValueError):
            with self.assertWarns(RuntimeWarning):
                incorrect.read((datetime(2021, 1, 12), datetime(2021, 1, 13)))

    def test_read_and_process(self):
        swaths = SwathFileCollection.from_product_id(
            self.swath_data,
            product_id="h129",
            dask_scheduler="processes",
        )
        start_p1 = datetime(2021, 1, 12)  #, 0, 0, 0)
        end_p1 = datetime(2021, 1, 13)  #, 0, 3, 0)
        ds = swaths.read((start_p1, end_p1)).load()
        processed_ds = swaths.process(ds)

    def test_continuity(self, run_test=False):
        # testing "stack"
        if not run_test:
            return

        swaths = SwathFileCollection.from_product_id(
            self.swath_data,
            product_id="h129",
            dask_scheduler="processes",
        )
        start_p1 = datetime(2021, 1, 1)
        end_p1 = datetime(2021, 1, 4)
        fnames_p1 = swaths.get_filenames(start_p1, end_p1)
        start_p2 = datetime(2021, 1, 4)
        end_p2 = datetime(2021, 1, 7)
        fnames_p2 = swaths.get_filenames(start_p2, end_p2)
        tester = Path(
            "/home/charriso/test_cells/data-write/RADAR/hsaf/tester/")
        outdir = tester / "cells"
        outdir_p1 = outdir / "p1"
        # outdir_p1.mkdir(exist_ok=True, parents=True)
        outdir_p2 = outdir / "p2"
        # outdir_p2.mkdir(exist_ok=True, parents=True)
        swaths.stack(fnames_p1, outdir_p1, processes=8)
        swaths.stack(fnames_p2, outdir_p2, processes=8)
        #

        cells = CellFileCollectionStack.from_product_id(
            outdir, product_id="h129")
        cell_outdir = tester / "merged_cells"
        cell_outdir.mkdir(exist_ok=True, parents=True)
        # print(cells.read(cell=0))
        cells.merge_and_write(out_dir=cell_outdir)
        max_cell_date = None
        min_cell_date = None
        for file in list(cell_outdir.glob("*.nc")):
            with xr.open_dataset(file, mask_and_scale=True) as written_ds:
                row_sum = written_ds["row_size"].sum()
                num_obs = written_ds["obs"].size
                self.assertEqual(row_sum, num_obs)

                written_ds_max_time = written_ds["time"].values.max()
                written_ds_min_time = written_ds["time"].values.min()
                if max_cell_date is None or written_ds_max_time > max_cell_date:
                    max_cell_date = written_ds_max_time
                if min_cell_date is None or written_ds_min_time < min_cell_date:
                    min_cell_date = written_ds_min_time

        self.assertLessEqual(
            max_cell_date.astype("datetime64[D]"), end_p2.date())
        self.assertGreaterEqual(
            min_cell_date.astype("datetime64[D]"), start_p1.date())

    def test_output(self):
        return
        ref = CellFileCollection.from_product_id(
            self.ref_idx_data / "20221212000000_20221219000000",
            product_id="sig0_12.5")
        outdir = Path(
            "/home/charriso/test_cells/data-write/RADAR/hsaf/tester/metop_a")
        print("ref")
        print(ref.cells_in_collection)
        print(
            ref.read(cell=0,
                     mask_and_scale=False).sel(obs=range(0, 10)).load())

        this = CellFileCollection.from_product_id(outdir, product_id="h129")
        print("this")
        this = this.read(cell=0, mask_and_scale=True).load()
        print(this)


if __name__ == "__main__":
    unittest.main()
