# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

from pathlib import Path


def get_current_path():
    """
    Get current file path.
    """
    return Path(__file__).resolve().parent


def get_testdata_path():
    """
    Get test data path.
    """
    return Path(__file__).resolve().parent / "ascat-test-data"
