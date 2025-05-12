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

from enum import Enum
from pathlib import Path
from typing import Dict, Type, Any, Tuple

from fibgrid.realization import FibGrid
from pygeogrids.netcdf import load_grid


class GridType(Enum):
    FIBGRID = "fibgrid"
    NAMED = "named"


class SingletonArgs(type):
    _instances: Dict[Tuple[Type, Tuple, frozenset], Any] = {}

    def __call__(cls, *args, **kwargs):
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]


class GridSingleton(metaclass=SingletonArgs):
    __slots__ = ['grid']

    def __init__(self, grid_class: Type, *args):
        self.grid = grid_class(*args)


class NamedFileGridRegistry:
    _grids: Dict[str, str] = {}

    @classmethod
    def register(cls, grid_name: str, grid_path: str) -> None:
        """Register a named grid with its file path."""
        cls._grids[grid_name] = Path(grid_path)

    @classmethod
    def get(cls, grid_name: str) -> str:
        """Retrieve the file path for a registered grid."""
        if grid_name not in cls._grids:
            raise KeyError(f"Grid '{grid_name}' is not registered.")
        return cls._grids[grid_name]


class NamedFileGrid:

    def __new__(cls, grid_name: str):
        grid_path = NamedFileGridRegistry.get(grid_name)
        return load_grid(grid_path)


class GridRegistry:
    _registry = {
        "fibgrid": FibGrid,
        "named": NamedFileGrid,
    }

    def register(
        self,
        grid_type_name: str,
        grid_class: type,
    ):
        """
        Register a grid class with a name for later retrieval.

        e.g. `register("fibgrid", FibGrid)` or `register("named", NamedFileGrid)`
        """
        if grid_type_name in self._registry:
            return
        self._registry[grid_type_name] = grid_class

    def get(self, grid_name):
        """
        Retrieve a grid instance based on its name.

        The grid name can be a simple name (e.g. "fibgrid") or a more complex
        name with parameters (e.g. "fibgrid_0.1"). The latter will be split
        into the grid type and its parameters.

        Parameters
        ----------
            grid_name (str): The name of the grid to retrieve.
        """
        parts = grid_name.split("_")

        if len(parts) >= 2 and parts[0] == "fibgrid":
            grid_type = "fibgrid"
            grid_spacing = float(parts[1])
            args = (grid_spacing,)
        elif len(parts) == 1:
            grid_type = "named"
            args = (grid_name,)
        else:
            grid_type = parts[0]
            args = tuple(parts[1:])

        grid_class = self._registry.get(grid_type)
        if grid_class is None:
            raise KeyError(f"Grid {grid_name} is not registered.")

        return GridSingleton(grid_class, *args).grid
