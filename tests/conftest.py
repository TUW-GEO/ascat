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

# Run dask synchronously in the test suite. netCDF4/HDF5 is not thread-safe and
# xarray serialises writes through process-global locks, so keeping dask out of
# worker threads makes the tests deterministic. (This alone did not fix the
# CI-only to_netcdf() hang -- see the CombinedLock patch below for that.)
import dask.config

dask.config.set(scheduler="synchronous")

# Make xarray's CombinedLock.acquire() all-or-nothing.
#
# Upstream it is `all(acquire(lock, blocking=blocking) for lock in self.locks)`,
# and `all()` short-circuits: the locks taken before the first failure are never
# released. CachingFileManager.__del__ uses exactly that non-blocking call and
# only releases when it returns True, so whenever the garbage collector
# finalises a file manager while any constituent lock (HDF5_LOCK, NETCDFC_LOCK
# or the per-file write lock) happens to be held, the locks acquired before the
# failure stay locked for the remaining lifetime of the process. The next
# netCDF write then blocks forever in SerializableLock.__enter__ -- the CI-only
# hang in to_netcdf(), only tripped on CI because it depends on GC timing.
from xarray.backends.locks import CombinedLock, acquire as _acquire_lock


def _acquire_all_or_nothing(self, blocking=True):
    acquired = []
    for lock in self.locks:
        if _acquire_lock(lock, blocking=blocking):
            acquired.append(lock)
        else:
            for held in acquired:
                held.release()
            return False
    return True


CombinedLock.acquire = _acquire_all_or_nothing
