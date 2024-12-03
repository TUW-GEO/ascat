import inspect
from fibgrid.realization import FibGrid
from pygeogrids.netcdf import load_grid


class Singleton(type):
    """ Simple Singleton that keep only one value for all instances
    """
    def __init__(cls, name, bases, dic):
        super(Singleton, cls).__init__(name, bases, dic)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance


class SingletonArgs(type):
    """ Singleton that keep single instance for single set of arguments. E.g.:
    assert SingletonArgs('spam') is not SingletonArgs('eggs')
    assert SingletonArgs('spam') is SingletonArgs('spam')
    """
    _instances = {}
    _init = {}

    def __init__(cls, name, bases, dct):
        cls._init[cls] = dct.get('__init__', None)

    def __call__(cls, *args, **kwargs):
        init = cls._init[cls]
        if init is not None:
            key = (cls, frozenset(
                    inspect.getcallargs(init, None, *args, **kwargs).items()))
        else:
            key = cls

        if key not in cls._instances:
            cls._instances[key] = super(SingletonArgs, cls).__call__(*args, **kwargs)
        return cls._instances[key]


class GridSingleton(object):
    """ Class based on Singleton type to work with MongoDB connections
    """
    __metaclass__ = SingletonArgs

    def __init__(self, grid_class, arg1=None):
        self.grid = grid_class(arg1)


static_grids = dict()


class NamedGridRegistry:
    """
    Just stores a lookup table basically between instantiated grid objects and their names. Stupid.
    """
    @classmethod
    def register(cls, grid_name, grid_path):
        static_grids[grid_name] = grid_path

    @classmethod
    def get(cls, grid_name):
        return static_grids[grid_name]


class NamedFileGrid:
    def __new__(cls, grid_name):
        grid_path = NamedGridRegistry.get(grid_name)
        return load_grid(grid_path)


class GridRegistry:
    def __init__(self):
        self._registry = {
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

        grid_func = self._registry.get(grid_type)
        if grid_func is None:
            raise KeyError(f"Grid {grid_name} is not registered.")

        return GridSingleton(grid_func, *args).grid
