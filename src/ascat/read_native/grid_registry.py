
from fibgrid.realization import FibGrid

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
