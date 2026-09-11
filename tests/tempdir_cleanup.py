#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.
"""Helpers for removing temporary directories that may still hold open files."""

import gc
import warnings

from xarray.backends.file_manager import FILE_CACHE


def close_cached_files():
    """Close every file handle xarray still holds open.

    Tests routinely open datasets and leave them to be garbage collected.
    xarray keeps the underlying netCDF handles in a process-global LRU cache,
    and datasets often sit in reference cycles, so a handle can easily outlive
    the test that opened it. A dataset that is still alive simply reopens its
    file the next time it is touched, so dropping the cached handles is safe.
    """
    gc.collect()
    for key in list(FILE_CACHE):
        file = FILE_CACHE.pop(key, None)
        if file is None:
            continue
        try:
            file.close()
        except Exception as error:  # noqa: BLE001 - closing must never raise
            warnings.warn(f"could not close cached file {key}: {error}", stacklevel=2)


def cleanup_tempdir(tempdir):
    """Remove ``tempdir``, releasing open file handles first.

    On Windows a file that is still open cannot be deleted, so leftover handles
    make ``TemporaryDirectory.cleanup()`` fail with ``PermissionError``
    (``[WinError 32] The process cannot access the file because it is being
    used by another process``). POSIX happily unlinks open files, which is why
    this only ever bites on the Windows CI runners.
    """
    close_cached_files()
    try:
        tempdir.cleanup()
    except PermissionError as error:
        # A handle we do not own escaped; do not fail the test over a temporary
        # directory that the CI runner discards anyway.
        warnings.warn(f"could not remove {tempdir.name}: {error}", stacklevel=2)
