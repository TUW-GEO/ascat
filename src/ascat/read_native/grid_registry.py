
from fibgrid.realization import FibGrid


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class GridSingleton(metaclass=SingletonMeta):
    def __init__(self, grid_class, *args, **kwargs):
        self.grid = grid_class(*args, **kwargs)

class GridRegistry:
    def __init__(self):
        self._registry = {
            "fibgrid": FibGrid
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

            case [name, *args]:
                grid_type = name
                args = tuple(args)

        grid_func = self._registry.get(grid_type)
        if grid_func is None:
            raise KeyError(f"Grid {grid_name} is not registered.")

        return GridSingleton(grid_func, *args).grid

class PyGeoGridRegistry:
    def __init__(self):
        # Store mappings and cache
        self._registry = {}
        self._cache = {}

    def register(self, uid, creation_fn, overwrite=False):
        """ This function registers a unique ID with a function to create the object.

        Parameters
        ----------
        uid : str
            Unique identifier for the object.
        param creation_fn : callable
            A function that returns a new instance of the desired object.
        overwrite : bool or str
            Whether to overwrite an existing registration.

        Returns
        -------
        None
        """

        if uid in self._registry:
            if overwrite=="raise":
                raise ValueError(f"UID '{uid}' is already registered.")
            if not overwrite:
                return

        self._registry[uid] = creation_fn

    def get(self, uid):
        """
        Retrieves an object by UID, creating it if it doesn't exist.

        Parameters
        ----------
        uid : str
            Unique identifier for the object.

        Returns
        -------
        object
            The requested object.
        """
        if uid in self._cache:
            return self._cache[uid]

        if uid not in self._registry:
            raise KeyError(f"UID '{uid}' is not registered.")

        # Create, cache, and return the new object
        self._cache[uid] = self._registry[uid]()
        return self._cache[uid]

grid_registry = PyGeoGridRegistry()

grid_registry.register("Fib6.25", lambda: {"grid": FibGrid(6.25), "attrs":{"grid_sampling_km": 6.25}})
grid_registry.register("Fib12.5", lambda: {"grid": FibGrid(12.5), "attrs":{"grid_sampling_km": 12.5}})
grid_registry.register("Fib25", lambda: {"grid": FibGrid(25), "attrs":{"grid_sampling_km": 25}})
