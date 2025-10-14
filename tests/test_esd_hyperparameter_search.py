"""
Hyperparameter search test for Dask cluster configuration for ESD computation.

This module performs systematic testing of different Dask cluster configurations
to optimize ESD computation performance over synthetic NetCDF files.
"""

import logging
import time
from pathlib import Path
import itertools
from typing import Tuple, List, Dict, Any

import numpy as np
import pytest
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from ascat.cell import CellGridFiles


def create_synthetic_h121_cell_file(cell_id: int, output_dir: Path, 
                                   grid,
                                   obs_per_location: int = 30) -> Path:
    """
    Create a synthetic H121 cell file with contiguous ragged array structure
    using actual GPIS from the FIB grid.
    
    Parameters
    ----------
    cell_id : int
        Cell identifier for this file
    output_dir : Path
        Directory where the file should be created
    grid : FibGrid
        The 12.5km FIB grid object containing actual GPIS
    obs_per_location : int
        Average number of observations per location
    
    Returns
    -------
    Path
        Path to the created synthetic H121 file
    """
    # Ensure cell_id is a Python int, not numpy type
    cell_id = int(cell_id)
    
    # Generate file name matching H121 convention
    filename = output_dir / f"{cell_id:04d}.nc"
    
    # Set random seed for reproducibility
    np.random.seed(42 + cell_id)
    
    # Get actual GPIS and coordinates from the FIB grid for this cell
    gpis, lons, lats = grid.grid_points_for_cell(cell_id)
    
    # Use all available points in this cell
    num_locations = len(gpis)
    lon = lons.astype(np.float32)
    lat = lats.astype(np.float32)
    alt = np.full(num_locations, np.nan, dtype=np.float32)
    
    # Use actual GPIS as location_id
    location_id = gpis.astype(np.int64)
    location_description = np.array([f"Cell_{cell_id}_GPI_{gpi}" for gpi in gpis], dtype=object)
    
    # Generate observations using contiguous ragged array structure
    # Vary observation counts per location for realism
    row_size = np.random.randint(obs_per_location - 5, obs_per_location + 5, num_locations, dtype=np.int32)
    total_obs = row_size.sum()
    
    # Time series (weekly observations over the year)
    base_time = np.datetime64("2020-01-01")
    time_data = []
    
    for loc_idx in range(num_locations):
        loc_obs = row_size[loc_idx]
        start_offset = np.random.randint(0, 365)
        daily_times = [base_time + np.timedelta64(start_offset + i*7, 'D') for i in range(loc_obs)]
        time_data.extend(daily_times)
    
    time_array = np.array(time_data, dtype='datetime64[ns]')
    
    # Synthetic backscatter40 data (-15 to 0 dB range)
    base_backscatter = np.random.uniform(-10.0, -5.0, total_obs)
    time_factor = np.sin(np.linspace(0, 4*np.pi, total_obs)) * 2.0
    noise = np.random.normal(0, 0.5, total_obs)
    backscatter40 = np.clip(base_backscatter + time_factor + noise, -15.0, 0.0).astype(np.float32)
    
    # Additional variables
    surface_flag = np.random.randint(0, 2, total_obs, dtype=np.int32)
    swath_indicator = np.random.randint(0, 3, total_obs, dtype=np.int32)
    as_des_pass = np.random.choice(['ASC', 'DES'], total_obs)
    surface_soil_moisture = np.clip(100 * np.exp(backscatter40 / 10), 0, 100).astype(np.float32)
    
    # Create xarray Dataset with CF-compliant structure
    ds = xr.Dataset(
        {
            'time': (['obs'], time_array),
            'location_id': (['locations'], location_id, {'cf_role': 'timeseries_id'}),
            'location_description': (['locations'], location_description),
            'row_size': (['locations'], row_size, {'sample_dimension': 'obs'}),
            'backscatter40': (['obs'], backscatter40, {'name': 'backscatter40'}),
            'surface_soil_moisture': (['obs'], surface_soil_moisture),
            'surface_flag': (['obs'], surface_flag),
            'as_des_pass': (['obs'], as_des_pass),
            'swath_indicator': (['obs'], swath_indicator),
            'sat_id': (['obs'], np.full(total_obs, 'METOP', dtype=object)),
        },
        coords={
            'lon': (['locations'], lon),
            'lat': (['locations'], lat),
            'alt': (['locations'], alt),
        },
        attrs={
            'id': f'{cell_id:04d}.nc',
            'date_created': '2025-01-01 00:00:00',
            'featureType': 'timeSeries'
        }
    )
    
    # Set coordinate attributes
    ds['lon'].attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
    ds['lat'].attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
    ds['alt'].attrs = {'long_name': 'altitude', 'units': 'm'}
    ds['time'].attrs = {'long_name': 'time'}
    ds = ds.set_coords(['location_id'])
    
    # Save to NetCDF
    ds.to_netcdf(filename, engine='h5netcdf')
    ds.close()
    
    return filename


def _create_synthetic_h121_cell_file_wrapper(args):
    """Wrapper function for parallel processing."""
    cell_id, output_dir, grid, obs_per_location = args
    return create_synthetic_h121_cell_file(cell_id, output_dir, grid, obs_per_location)


def create_synthetic_h121_files_parallel(test_cell_ids, output_dir, grid, obs_per_location=100,
                                         n_workers=4, threads_per_worker=1):
    """
    Create synthetic H121 cell files in parallel using a separate Dask cluster.
    
    Parameters
    ----------
    test_cell_ids : list
        List of cell IDs to create files for
    output_dir : Path
        Directory where the files should be created
    grid : FibGrid
        The 12.5km FIB grid object containing actual GPIS
    obs_per_location : int
        Average number of observations per location
    n_workers : int
        Number of workers for the parallel cluster
    threads_per_worker : int
        Threads per worker for the parallel cluster
    
    Returns
    -------
    list
        List of created file paths
    """
    # Create a separate cluster for file creation (different from reading cluster)
    creation_cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        scheduler_port=0,  # Let Dask choose port
        processes=False,
        silence_logs=logging.ERROR
    )
    creation_client = Client(creation_cluster)
    
    try:
        # Prepare arguments for parallel processing
        args_list = [(cell_id, output_dir, grid, obs_per_location) for cell_id in test_cell_ids]
        
        # Create delayed tasks
        delayed_tasks = [delayed(_create_synthetic_h121_cell_file_wrapper)(args) for args in args_list]
        
        # Compute in parallel
        with ProgressBar():
            file_paths = compute(*delayed_tasks)
        
        return file_paths
    
    finally:
        # Close the creation cluster
        creation_client.close()
        creation_cluster.close()


def calc_esd(ts: xr.DataArray) -> xr.DataArray:
    """
    Calculate ESD (Effective Standard Deviation) for a time series.
    
    This is the same function used in the original test_esd.py.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.std(ts, ddof=1, axis=len(ts.shape)-1) / np.sqrt(2.)


def setup_synthetic_data(tmp_path: Path) -> CellGridFiles:
    """
    Setup synthetic H121 data for testing.
    
    Parameters
    ----------
    tmp_path : Path
        Temporary directory for test files
    
    Returns
    -------
    CellGridFiles
        Instance of CellGridFiles for the synthetic data
    """
    # Use real FibGrid(12.5) instead of synthetic grid
    from fibgrid.realization import FibGrid
    from ascat.cell import RaggedArrayTs
    
    # Create FibGrid for cell identification
    fib_grid = FibGrid(12.5)
    
    # Get available cells from FibGrid and create synthetic files for a subset
    fib_cell_ids = list(set(fib_grid.arrcell))
    
    # Use a subset of cells for testing (evenly distributed from the ordered list)
    num_test_cells = 50 # Reduced for faster hyperparameter search
    if len(fib_cell_ids) >= num_test_cells:
        # Take cells evenly spaced from the list
        step = len(fib_cell_ids) // num_test_cells
        test_cell_ids = fib_cell_ids[::step][:num_test_cells]
    else:
        # If we have fewer cells than requested, use all available
        test_cell_ids = fib_cell_ids
    
    # Create files in parallel using a separate Dask cluster
    print(f"Creating {len(test_cell_ids)} synthetic H121 cell files in parallel...")
    created_files = create_synthetic_h121_files_parallel(
        test_cell_ids=test_cell_ids,
        output_dir=tmp_path,
        grid=fib_grid,
        obs_per_location=50,  # Further reduced for faster tests
        n_workers=1,  # Use more workers for file creation
        threads_per_worker=8
    )
    
    print(f"Successfully created {len(created_files)} synthetic files")
    
    class FibGridCellGridFiles:
        def __init__(self, root_path, file_class, grid):
            self.root_path = root_path
            self.file_class = file_class
            self.grid = grid
            
        @property
        def available_cells(self):
            # Return only the cells we actually created files for
            return test_cell_ids
            
        def fn_search(self, cell_id):
            filename = f"{cell_id:04d}.nc"
            file_path = self.root_path / filename
            return [file_path] if file_path.exists() else []
    
    return FibGridCellGridFiles(
        root_path=tmp_path,
        file_class=RaggedArrayTs,
        grid=fib_grid
    )


def run_esd_computation_test(
    cluster_config: Dict[str, Any], 
    synthetic_files: CellGridFiles,
    num_cells: int = 10
) -> Dict[str, Any]:
    """
    Run ESD computation test with specified Dask cluster configuration.
    
    Parameters
    ----------
    cluster_config : dict
        Configuration dictionary for Dask cluster
    synthetic_files : CellGridFiles
        Synthetic data files to process
    num_cells : int
        Number of cells to process
    
    Returns
    -------
    dict
        Results of the test including timing and performance metrics
    """
    # Get available cells and select subset for testing
    all_cells = list(synthetic_files.available_cells)
    test_cells = sorted(all_cells[:num_cells])
    
    # Extract cluster configuration
    n_workers = cluster_config['n_workers']
    threads_per_worker = cluster_config['threads_per_worker']
    processes = cluster_config.get('processes', True)
    
    print(f"\n--- Testing cluster: {n_workers} workers, {threads_per_worker} threads/worker, "
          f"processes={processes} on {num_cells} cells ---")

    def compute_esd_pointwise(ds, cell_id):
        backscatter = ds['backscatter40'].values
        row_size = ds['row_size'].values

        esd_results = []
        for loc_idx in range(len(row_size)):
            if loc_idx == 0:
                start_idx = 0
            else:
                start_idx = row_size[:loc_idx].sum()

            end_idx = start_idx + row_size[loc_idx]
            location_data = backscatter[start_idx:end_idx]

            # Calculate ESD for this location
            esd_value = calc_esd(location_data)
            esd_scalar = float(esd_value)  # Convert np array to scalar
            esd_results.append((cell_id, loc_idx, esd_scalar))

        return esd_results

    def compute_esd_vectorized(ds, cell_id):
        ds = ds.cf_geom[['backscatter40']].cf_geom.to_orthomulti()
        esd_da = calc_esd(ds["backscatter40"].values)
        cell_ids = np.full(esd_da.shape, cell_id).tolist()
        loc_indices = esd_da['location_id'].values.tolist()
        esd_results = list(zip(cell_ids, loc_indices, esd_da.tolist()))
        return esd_results

    # Set up Dask cluster
    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            scheduler_port=0,
            silence_logs=logging.ERROR
        )
        client = Client(cluster)
        
        # Define ESD computation function for parallel execution
        def compute_cell_esd(cell_id, esd_function):
            start_time = time.time()
            
            # Load cell data
            cell_file = synthetic_files.fn_search(cell_id)[0]
            ds = xr.open_dataset(cell_file, engine='h5netcdf')
            
            esd_results = esd_function(ds, cell_id)

            ds.close()
            elapsed_time = time.time() - start_time

            return cell_id, elapsed_time, esd_results


        # Compute ESD in parallel
        print(f"Computing ESD for {len(test_cells)} cells...")
        delayed_tasks = [delayed(compute_cell_esd)(cell_id, compute_esd_vectorized) for cell_id in test_cells]
        
        start_time = time.time()
        with ProgressBar():
            results = compute(*delayed_tasks)
        total_elapsed = time.time() - start_time
        
        # Process results
        all_esd_results = []
        for cell_id, cell_time, esd_results in results:
            all_esd_results.extend(esd_results)
        
        # Calculate performance metrics
        total_cells = len(test_cells)
        avg_cell_time = sum(r[1] for r in results) / total_cells
        total_locations = len(all_esd_results)
        
        # Close cluster
        client.close()
        cluster.close()
        
        return {
            'n_workers': n_workers,
            'threads_per_worker': threads_per_worker,
            'processes': processes,
            'total_time': total_elapsed,
            'avg_cell_time': avg_cell_time,
            'total_cells': total_cells,
            'total_locations': total_locations,
            'locations_per_second': total_locations / total_elapsed if total_elapsed > 0 else 0,
            'success': True
        }
        
    except Exception as e:
        print(f"Cluster configuration failed: {e}")
        return {
            'n_workers': n_workers,
            'threads_per_worker': threads_per_worker,
            'processes': processes,
            'total_time': float('inf'),
            'avg_cell_time': float('inf'),
            'total_cells': 0,
            'total_locations': 0,
            'locations_per_second': 0.0,
            'success': False,
            'error': str(e)
        }


def test_dask_cluster_hyperparameter_search(tmp_path: Path) -> None:
    """
    Perform hyperparameter search for Dask cluster configuration to optimize ESD computation.
    
    Tests various combinations of n_workers, threads_per_worker, and processes settings
    to find optimal configuration for ESD computation on 8 cores with 2 threads per core.
    """
    print("\n=== DASK CLUSTER HYPERPARAMETER SEARCH TEST ===")
    
    # Setup synthetic data
    synthetic_files = setup_synthetic_data(tmp_path)
    
    # Define hyperparameter search space
    # Machine has 8 cores with 2 threads per core = 16 total logical processors
    
    # Test configurations:
    # 1. Single-threaded scheduler (processes=False): n_workers varies, threads_per_worker=1
    single_thread_configs = [
        # {'n_workers': 1, 'threads_per_worker': 1, 'processes': False},
        # {'n_workers': 2, 'threads_per_worker': 1, 'processes': False},
        # {'n_workers': 4, 'threads_per_worker': 1, 'processes': False},
        # {'n_workers': 8, 'threads_per_worker': 1, 'processes': False},
    ]
    
    # 2. Multi-threaded single process (processes=False): varying threads
    multithread_configs = [
        {'n_workers': 1, 'threads_per_worker': 2, 'processes': False},
        # {'n_workers': 1, 'threads_per_worker': 4, 'processes': False},
        {'n_workers': 1, 'threads_per_worker': 8, 'processes': False},
        # {'n_workers': 1, 'threads_per_worker': 16, 'processes': False},
    ]
    
    # 3. Process-based scheduler (processes=True): balanced configurations
    process_configs = [
        {'n_workers': 1, 'threads_per_worker': 2, 'processes': True},
        {'n_workers': 1, 'threads_per_worker': 8, 'processes': True},
        {'n_workers': 2, 'threads_per_worker': 2, 'processes': True},
        {'n_workers': 4, 'threads_per_worker': 2, 'processes': True},
        {'n_workers': 8, 'threads_per_worker': 1, 'processes': True},
        {'n_workers': 4, 'threads_per_worker': 4, 'processes': True},
    ]
    
    all_configs = single_thread_configs + multithread_configs + process_configs
    
    print(f"\nTesting {len(all_configs)} different cluster configurations...")
    
    # Run all configurations
    results = []
    for i, config in enumerate(all_configs):
        print(f"\nConfiguration {i+1}/{len(all_configs)}: {config}")
        
        # Pause briefly between tests to avoid port conflicts
        if i > 0:
            time.sleep(1)
        
        result = run_esd_computation_test(config, synthetic_files, num_cells=50)
        results.append(result)
        
        # Print intermediate results
        if result['success']:
            print(f"  ✓ SUCCESS: {result['total_time']:.2f}s total, "
                  f"{result['locations_per_second']:.2f} locations/sec")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
    
    # Analyze and report results
    print("\n=== HYPERPARAMETER SEARCH RESULTS ===")
    
    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print("No successful cluster configurations found!")
        return
    
    # Sort by performance (locations per second)
    successful_results.sort(key=lambda x: x['locations_per_second'], reverse=True)
    
    print("\nTop 5 performing configurations:")
    for i, result in enumerate(successful_results[:5]):
        config_str = f"{result['n_workers']}w×{result['threads_per_worker']}t{'-p' if result['processes'] else '-m'}"
        print(f"{i+1}. {config_str}: {result['locations_per_second']:.2f} loc/sec, "
              f"{result['total_time']:.2f}s total, "
              f"{result['total_locations']}/{result['total_cells']} cells")
    
    # Find best configuration
    best_config = successful_results[0]
    print(f"\n!!! BEST CONFIGURATION !!!")
    print(f"Config: {best_config['n_workers']} workers × {best_config['threads_per_worker']} threads/worker, "
          f"processes={best_config['processes']}")
    print(f"Performance: {best_config['locations_per_second']:.2f} locations/second")
    print(f"Total time: {best_config['total_time']:.2f} seconds")
    print(f"Throughput: {best_config['total_locations']} locations in {len(synthetic_files.available_cells)} cells")
    
    # Provide recommendations based on results
    print("\n=== RECOMMENDATIONS ===")
    
    best_single = sorted([r for r in successful_results if not r['processes']], 
                        key=lambda x: x['locations_per_second'], reverse=True)
    best_multiprocess = sorted([r for r in successful_results if r['processes']], 
                              key=lambda x: x['locations_per_second'], reverse=True)
    
    if best_single:
        print(f"Best single-threaded config: {best_single[0]['n_workers']}w×{best_single[0]['threads_per_worker']}t"
              f" ({best_single[0]['locations_per_second']:.2f} loc/sec)")
    
    if best_multiprocess:
        print(f"Best multiprocessing config: {best_multiprocess[0]['n_workers']}w×{best_multiprocess[0]['threads_per_worker']}t"
              f" ({best_multiprocess[0]['locations_per_second']:.2f} loc/sec)")
    
    # Overall recommendation
    if best_multiprocess and best_single:
        if best_multiprocess[0]['locations_per_second'] > best_single[0]['locations_per_second']:
            print("✓ RECOMMENDATION: Use multiprocessing (processes=True) for best performance")
        else:
            print("✓ RECOMMENDATION: Use single-threaded/multithreaded (processes=False) for best performance")
    elif best_multiprocess:
        print("✓ RECOMMENDATION: Use multiprocessing (processes=True) for best performance")
    elif best_single:
        print("✓ RECOMMENDATION: Use single-threaded/multithreaded (processes=False) for best performance")
    
    print("\n=== TESTING COMPLETE ===")


if __name__ == "__main__":
    # Allow running test directly
    pytest.main([__file__ + "::test_dask_cluster_hyperparameter_search", "-v", "-s"])
