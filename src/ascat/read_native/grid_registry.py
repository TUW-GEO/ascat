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
