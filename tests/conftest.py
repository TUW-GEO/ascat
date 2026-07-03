"""
    Dummy conftest.py for ascat.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import os

# Disable HDF5 file locking before netCDF4/h5py (and thus the HDF5 library) are
# imported. On some CI filesystems HDF5's file locking blocks indefinitely when
# writing NetCDF files, hanging to_netcdf() on the file lock. The test suite only
# writes fresh temporary files from a single process, so locking is not needed.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Run dask synchronously in the test suite. netCDF4/HDF5 is not thread-safe, so
# xarray serialises writes through a process-global threading.Lock. Under dask's
# default *threaded* scheduler that lock can be left held across worker threads,
# and a later to_netcdf() then blocks forever acquiring it (a race that only
# trips on CI timing). The synchronous scheduler removes the worker threads
# entirely, so the write lock can never be contended.
import dask.config

dask.config.set(scheduler="synchronous")
