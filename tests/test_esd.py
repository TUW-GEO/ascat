import gc
import logging
import time
import warnings
from pathlib import Path
import random
from typing import Any, Sequence, Union, Tuple, List

import numpy as np
import pytest
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from ascat.cell import CellGridFiles


@pytest.fixture
def h121_files() -> CellGridFiles:
    # root_path = "tests/ascat-test-data/hsaf/h121/stack_cells"
    root_path = "/data-write/RADAR/hsaf/h121_v2.0/time_series/metop_abc/"
    return CellGridFiles.from_product_id(root_path, "H121")

def calc_esd_ds(ds: xr.Dataset) -> xr.Dataset:
    ds = compute(ds)[0]
    ds = ds.cf_geom.to_contiguous_ragged()
    # for gpi in ds["location_id"].values:
    esd_ds = xr.concat(
        [calc_esd(ds.cf_geom.sel_instances([gpi])["backscatter40"])
        for gpi in ds["location_id"].values],
        dim="locations",
        combine_attrs="override"
    ).to_dataset(name="esd")
    esd_ds["location_id"] = ("locations", ds["location_id"].values)
    # esd_ds = AscatH121Cell.preprocessor(esd_ds)
    esd_ds.attrs["featureType"] = "point"
    # print(esd_ds.cf_geom.to_contiguous_ragged())
    return esd_ds

def calc_esd(ts: xr.DataArray) -> xr.DataArray:
    # ignore RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.std(ts, ddof=1, axis=len(ts.shape)-1) / np.sqrt(2.)

def test_calculate_esd_pointwise(h121_files: CellGridFiles, tmp_path: Path) -> None:
    client = Client(LocalCluster(n_workers=4, threads_per_worker=2, scheduler_port=8786))
    # h121_files.reprocess(out_dir=Path("/tmp/test"), func=calc_esd_ds, parallel=True, print_progress=True, ra_type="point")
    iterator = h121_files.iter_cells(delay=True, data_vars=['backscatter40'])
    results = client.map(calc_esd_ds, [ds for ds in [next(iterator) for _ in range(3)]])
    output_ds = xr.concat(client.gather(results), dim="locations", combine_attrs="override")
# import queue


def test_calculate_esd_2d(h121_files: CellGridFiles, tmp_path: Path) -> None:
    # result_queue = queue.Queue()
    def calc_esd_ds_2d(ds: xr.Dataset) -> xr.Dataset:
        ds = xr.decode_cf(compute(ds)[0])
        ds = ds.cf_geom.to_orthomulti()
        esd_ds = calc_esd(ds["backscatter40"]).to_dataset(name="esd")
        esd_ds.attrs["featureType"] = "point"
        # result_queue.put(esd_ds)
        ds.close()
        # gc.collect()
        return esd_ds
    pbar = ProgressBar()
    pbar.register()
    client = Client(LocalCluster(n_workers=8, threads_per_worker=2, scheduler_port=8786))
    # threaded scheduler client
    # client = Client(processes=False, threads_per_worker=2, n_workers=2, scheduler_port=8786)
    # h121_files.reprocess(out_dir=Path("/tmp/test"), func=calc_esd_ds_2d, parallel=True, print_progress=True, ra_type="point")
    iterator = h121_files.iter_cells(delay=True, data_vars=['backscatter40'], decode_cf=False)
    # for i in range(25):
    #     next(iterator)
    esds = [delayed(calc_esd_ds_2d)(ds) for ds in iterator]
    # results = client.map(calc_esd_ds_2d, [ds for ds in iterator])
    def write_to_zarr(
            ds: xr.Dataset,
            zarr_path: str,
            mode: str = 'a') -> Any:
        return ds.to_zarr(zarr_path, mode=mode, append_dim='locations' if mode=='a' else None)
        # ds.close()
        # return


    # write_to_zarr(esds.pop(0), "/tmp/esd_2d.zarr", mode='w')
    # writes = [delayed(write_to_zarr)(ds, "/tmp/esd_2d.zarr", mode='a') for ds in esds]
    # compute(writes)
    for i in tqdm(range(0, len(esds), 32), desc="Processing ESDs in chunks of 8"):
        data = esds[i:i+24]
        ds = xr.concat(compute(data)[0], dim="locations", combine_attrs="override")
        if i == 0:
            write_to_zarr(ds, "/tmp/esd_2d.zarr", mode='w')
        else:
            write_to_zarr(ds, "/tmp/esd_2d.zarr")
        ds.close()
        del ds
        client.run(gc.collect)
    # compute(writes)

    # results = client.map(write_to_zarr,
    #                      *zip(*[(delayed(calc_esd_ds_2d)(ds), "/tmp/esd_2d.zarr") for ds in iterator]))
    # for result in
    # results.pop(0).result()
    # results = []
    # for ds in iterator:
    #     results.append(client.submit(write_to_zarr, (delayed(calc_esd_ds_2d)(ds), "/tmp/esd_2d.zarr")))
    # results = client.map(write_to_zarr, [(delayed(calc_esd_ds_2d)(ds), "/tmp/esd_2d.zarr") for ds in iterator])
    # print(client.gather(results))

    # output_ds = xr.concat(client.gather(results), dim="locations", combine_attrs="override")
    # output_ds.to_zarr("/tmp/esd_2d.zarr", mode='w')
    # output_ds.to_netcdf("/tmp/esd_2d.nc", mode='w')
    # for cell in h121_files.iter_cells():


def test_iterate_one_file(h121_files: CellGridFiles, tmp_path: Path) -> None:
    client = Client(LocalCluster(n_workers=8, threads_per_worker=2, scheduler_port=8786))
    def iterate_cell(cell: int) -> None:
        ds = h121_files.read(cell=cell)[['backscatter40', 'location_id', 'time', 'row_size']].load().cf_geom.to_contiguous_ragged()
        for gpi in ds["location_id"].values:
            # ds.cf_geom.sel_instances([gpi])
            esd = calc_esd(ds.cf_geom.sel_instances([gpi])["backscatter40"]).to_dataset(name="esd")
    tasks = client.map(iterate_cell, [9, 100, 166, 230])
    results = client.gather(tasks)
        # esd = calc_esd(ds.cf_geom.sel_instances([gpi])["backscatter40"]).to_dataset(name="esd")
        # print(esd)

def test_2d_one_file(h121_files: CellGridFiles, tmp_path: Path) -> None:
    client = Client(LocalCluster(n_workers=4, threads_per_worker=2, scheduler_port=8786))
    def esd_cell(cell: int) -> None:
        ds = h121_files.read(cell=cell).cf_geom[['backscatter40']].load().cf_geom.to_orthomulti()
        print(ds)
        esd = calc_esd(ds["backscatter40"]).to_dataset(name="esd")
    tasks = client.map(esd_cell, [9, 100, 166, 230])
    results = client.gather(tasks)
    assert False

def test_to_orthomulti(h121_files: CellGridFiles, tmp_path: Path) -> None:
    ds = h121_files.read(cell=9, data_vars=['backscatter40']).load()
    print(ds.backscatter40.mean())
    print(ds.cf_geom.to_orthomulti())
    print(ds.cf_geom.to_orthomulti().backscatter40.mean())
    # calc_esd_ds_2d(ds)
    # print(ds.cf_geom.to_point_array().cf_geom.to_orthomulti())
    assert False


def test_load_backscatter_timing(h121_files: CellGridFiles, tmp_path: Path) -> None:
    """Test loading backscatter40 from 50 random cells and log timing."""
    # Print the tmp_path to see where it is
    print(f"tmp_path: {tmp_path}")

    # Set up logging
    log_file = tmp_path/'backscatter_timing.log'
    print(f"Log file will be created at: {log_file}")

    # Create the directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set up logging with both file and stream handlers
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    logger.info("Starting test_load_backscatter_timing")

    # Get all available cells
    all_cells = list(h121_files.available_cells)
    logger.info(f"Total available cells: {len(all_cells)}")

    # Select 50 random cells and sort them
    random_cells = random.sample(all_cells, min(50, len(all_cells)))
    random_cells.sort()  # Sort the cell IDs

    logger.info(f"Loading backscatter40 from {len(random_cells)} random cells")

    # Loop through each random cell
    for i, cell_id in tqdm(enumerate(random_cells), total=len(random_cells), desc="Loading cells"):
        start_time = time.time()

        # Get the file path for this cell
        cell_file = h121_files.fn_search(cell_id)[0]
        logger.info(f"Processing cell {cell_id}, file: {cell_file}")

        # Open with xr.open_dataset with decode, mask, etc. set to False
        ds = xr.open_dataset(
            cell_file,
            decode_cf=False,
            mask_and_scale=False,
            decode_times=False,
            engine='h5netcdf',
        )

        # Load the backscatter40 variable
        backscatter = ds['backscatter40'].load()

        # Close the dataset
        ds.close()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Log the timing
        logger.info(f"Cell {cell_id} ({i+1}/{len(random_cells)}): {elapsed_time:.4f} seconds")

    logger.info("Completed loading all cells")

    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()

    # Close all handlers
    for handler in logger.handlers:
        handler.close()

    # Verify the log file exists and has content
    if log_file.exists():
        print(f"Log file exists with size: {log_file.stat().st_size} bytes")
        with open(log_file, 'r') as f:
            print("Log file content:")
            print(f.read())
    else:
        print("Log file does not exist!")
