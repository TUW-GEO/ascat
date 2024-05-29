#!/usr/bin/env python3

from fibgrid.realization import FibGrid
from ascat.read_native.xarray_io import grid_cache

cell_io_catalog = {
    "H129": {"grid": grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]},
    "H129_V1.0": {"grid": grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]},
    "H121_V1.0": {"grid": grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)["grid"]},
    "H122": {"grid": grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]},
    "SIG0_6.25": {"grid": grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]},
    "SIG0_12.5": {"grid": grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)["grid"]},
}

def product_grid(product_id):
    product_id = product_id.upper()
    if product_id == "H129":
        return grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]
    elif product_id == "H129_V1.0":
        return grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]
    elif product_id == "H121_V1.0":
        return grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)["grid"]
    elif product_id == "H122":
        return grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]
    elif product_id == "SIG0_6.25":
        return grid_cache.fetch_or_store("Fib6.25", FibGrid, 6.25)["grid"]
    elif product_id == "SIG0_12.5":
        return grid_cache.fetch_or_store("Fib12.5", FibGrid, 12.5)["grid"]


class ProductInfo:
    def __init__(self, product_id):
        self.grid = product_grid(product_id)
