import inspect
from enum import Enum
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
        cls._grids[grid_name] = grid_path

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
        if grid_type_name in self._registry:
            return
        self._registry[grid_type_name] = grid_class

    def get(self, grid_name):
        match grid_name.split("_"):
            case ["fibgrid", grid_spacing]:
                grid_type = "fibgrid"
                grid_spacing = float(grid_spacing)
                args = (grid_spacing,)

            case [grid_name]:
                grid_type = "named"
                args = (grid_name,)

            case [name, *args]:
                grid_type = name
                args = tuple(args)

        grid_class = self._registry.get(grid_type)
        if grid_class is None:
            raise KeyError(f"Grid {grid_name} is not registered.")

        return GridSingleton(grid_class, *args).grid
