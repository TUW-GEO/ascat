# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Grid-aware reading of ragged-array cell files.

A :mod:`pygeogrids` ``CellGrid`` links grid point indices (gpi) and coordinates
to cell numbers, and each cell is stored in its own ragged-array file. A gpi (or
the nearest gpi to a lon/lat) is translated to a cell, the cell file is located
and read with the matching :mod:`ascat.ragged_array` class, and the single grid
point's time series is returned.

Use :class:`GriddedContiguousRaggedArray` for contiguous ragged cell files and
:class:`GriddedIndexedRaggedArray` for indexed ragged cell files.
"""

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from fibgrid.realization import FibGrid
from pygeogrids.grids import CellGrid

from ascat.array_utils import fill_value
from ascat.ragged_array import (
    ContiguousRaggedArray,
    IndexedRaggedArray,
    OrthogonalMultidimArray,
)


class GridRegistry:
    """
    Resolve grid names to :mod:`pygeogrids` grids.

    ``"fibgrid_<spacing>"`` (e.g. "fibgrid_12.5") builds a
    :class:`~fibgrid.realization.FibGrid` on demand; any other name must be
    registered with :meth:`register`, either as a grid object or a
    zero-argument factory (e.g. ``lambda: load_grid(path)``). Resolved grids are
    cached, so a grid is built or loaded only once per name.

    Each instance holds its own registry and cache. A shared module-level
    instance, :data:`grid_registry`, is used by the readers by default; create a
    separate ``GridRegistry()`` when you want isolated state (e.g. in tests).
    """

    def __init__(self):
        self._registry = {}
        self._cache = {}

    def register(self, name: str, grid) -> None:
        """Register a grid object or zero-argument factory under ``name``."""
        self._registry[name] = grid
        self._cache.pop(name, None)

    def get(self, name: str) -> CellGrid:
        """Return the grid for ``name``, building and caching it on first use."""
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def _resolve(self, name: str) -> CellGrid:
        if name in self._registry:
            grid = self._registry[name]
            return grid() if callable(grid) else grid
        parts = name.split("_")
        if len(parts) == 2 and parts[0] == "fibgrid":
            return FibGrid(float(parts[1]))
        raise KeyError(
            f"Grid '{name}' is not registered. Register it with "
            "grid_registry.register(name, grid), or use a 'fibgrid_<spacing>' "
            "name (e.g. 'fibgrid_12.5').")


#: Shared grid registry used by the readers when a grid name (str) is passed.
grid_registry = GridRegistry()


class GriddedRaggedArray:
    """
    Base class: read gridded ragged-array cell files by grid point or location.

    A ``CellGrid`` maps grid point indices (gpi) and coordinates to cell numbers;
    each cell is stored in its own ragged-array file. A gpi (or the nearest gpi
    to a lon/lat) is translated to a cell via ``grid.gpi2cell``, the cell file is
    located with ``fn_format`` and read, and the grid point's time series is
    selected by its identifier.

    Two caching strategies control how cell files are kept in memory:

    - ``cache=False`` (default): only the most recently read cell is kept, like
      pynetcf's ``GriddedNcTs``. Efficient for reads that stay within one cell;
      re-reads when hopping between cells.
    - ``cache=True``: every cell that is read is retained in memory. Efficient
      when reading many grid points spread across (and revisiting) cells, at the
      cost of memory.

    This base class is not used directly; use
    :class:`GriddedContiguousRaggedArray` or :class:`GriddedIndexedRaggedArray`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files (searched recursively).
    grid : pygeogrids.grids.CellGrid or str
        Grid object, or a name resolvable by the module-level
        :data:`grid_registry` (e.g. "fibgrid_12.5").
    fn_format : str, optional
        Format string for cell file names, formatted with the cell number, e.g.
        "{:04d}.nc" (default) or "H120_{:04d}.nc".
    cache : bool, optional
        Keep every read cell in memory instead of only the last one
        (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        cache: bool = False,
    ):
        self.root_path = Path(root_path)
        self.grid = grid_registry.get(grid) if isinstance(grid, str) else grid
        self.fn_format = fn_format
        self.cache = cache

        # cache=True: {cell: ragged array}
        self._cells = {}
        # cache=False: only the most recently opened cell is kept
        self._active_cell = None
        self._active_reader = None

    def _open_cell(self, cell: int):
        """Open the file for ``cell`` and return a ragged-array reader."""
        raise NotImplementedError

    def _cell_filename(self, cell: int) -> Path:
        """Locate the file for a cell under ``root_path``."""
        name = self.fn_format.format(cell)
        matches = sorted(self.root_path.glob("**/" + name))
        if not matches:
            raise FileNotFoundError(
                f"No cell file for cell {cell} ('{name}') under "
                f"'{self.root_path}'.")
        return matches[0]

    def read_cell(self, cell: int):
        """
        Read a whole cell.

        Parameters
        ----------
        cell : int
            Cell number.

        Returns
        -------
        data : ContiguousRaggedArray or IndexedRaggedArray
            Ragged array for the cell (cached per the caching strategy).
        """
        cell = int(cell)
        if self.cache:
            if cell not in self._cells:
                self._cells[cell] = self._open_cell(cell)
            return self._cells[cell]

        if cell != self._active_cell:
            self._active_reader = self._open_cell(cell)
            self._active_cell = cell
        return self._active_reader

    def gpi_from_coords(
        self, lon: float, lat: float, max_dist: float = np.inf
    ) -> int:
        """
        Nearest grid point index to a lon/lat.

        Parameters
        ----------
        lon, lat : float
            Coordinates.
        max_dist : float, optional
            Maximum allowed distance in meters (default: no limit).

        Returns
        -------
        gpi : int
            Grid point index of the nearest grid point.

        Raises
        ------
        ValueError
            If the nearest grid point is farther than ``max_dist``.
        """
        gpi, dist = self.grid.find_nearest_gpi(lon, lat)
        if dist > max_dist:
            raise ValueError(
                f"No grid point within {max_dist} m of ({lon}, {lat}); "
                f"nearest is {dist:.0f} m away.")
        return int(gpi)

    def read(
        self,
        gpi=None,
        lon=None,
        lat=None,
        max_dist: float = np.inf,
    ):
        """
        Read the time series of one or more grid points.

        Either ``gpi`` or both ``lon`` and ``lat`` must be given. Each may be a
        scalar (single grid point) or array-like (many grid points at once).

        Parameters
        ----------
        gpi : int or array-like of int, optional
            Grid point index/indices.
        lon, lat : float or array-like of float, optional
            Coordinates; the nearest grid point to each is used.
        max_dist : float, optional
            Maximum allowed distance in meters for a lon/lat lookup. For a
            single lon/lat this raises if exceeded; for arrays, coordinates
            beyond it are dropped.

        Returns
        -------
        ds : xarray.Dataset or None, or dict of {gpi: (xarray.Dataset or None)}
            For a scalar request, the time series (or None if the grid point has
            no data in its cell file). For an array request, a dict mapping each
            grid point index to its time series (or None). Cell files are read
            once per cell, so grouping many grid points is efficient.
        """
        if gpi is None:
            if lon is None or lat is None:
                raise ValueError("Provide either 'gpi' or both 'lon' and 'lat'.")
            if np.ndim(lon) == 0:
                gpi = self.gpi_from_coords(lon, lat, max_dist)
            else:
                near, dist = self.grid.find_nearest_gpi(lon, lat)
                gpi = np.asarray(near)[np.asarray(dist) <= max_dist]

        if np.ndim(gpi) == 0:
            cell = int(self.grid.gpi2cell(gpi))
            return self.read_cell(cell).sel_instance(int(gpi))

        return self._read_gpis(np.asarray(gpi))

    def _read_gpis(self, gpis: np.ndarray) -> dict:
        """Read many grid points, reading each cell file only once."""
        cells = np.atleast_1d(self.grid.gpi2cell(gpis))
        per_gpi = {}
        for cell in np.unique(cells):
            reader = self.read_cell(int(cell))
            for g in gpis[cells == cell]:
                per_gpi[int(g)] = reader.sel_instance(int(g))
        # return in the order of the requested gpis
        return {int(g): per_gpi[int(g)] for g in gpis}

    @staticmethod
    def _close_reader(reader):
        """Close a reader's underlying dataset, if any."""
        if reader is not None:
            reader.ds.close()

    def close(self):
        """Close all open cell files and drop cached cells."""
        for reader in self._cells.values():
            self._close_reader(reader)
        self._close_reader(self._active_reader)
        self._cells.clear()
        self._active_cell = None
        self._active_reader = None

    def clear_cache(self):
        """Close all open cell files and drop cached cells (alias of close)."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GriddedContiguousRaggedArray(GriddedRaggedArray):
    """
    Read gridded contiguous ragged-array cell files.

    See :class:`GriddedRaggedArray` for the gridded reading and caching
    behavior. Each cell is read with
    :class:`~ascat.ragged_array.ContiguousRaggedArray`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files.
    grid : pygeogrids.grids.CellGrid or str
        Grid object or registry name.
    fn_format : str, optional
        Cell file name format (default: "{:04d}.nc").
    count_var : str, optional
        Count variable name (default: "row_size").
    instance_dim : str, optional
        Instance dimension name (default: "locations").
    instance_id_var : str, optional
        Variable holding the instance identifiers, matching the grid gpi
        (default: "location_id").
    trim : bool, optional
        Drop fill/padding locations when reading a cell (default: True).
    cache : bool, optional
        Keep every read cell in memory (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        count_var: str = "row_size",
        instance_dim: str = "locations",
        instance_id_var: str = "location_id",
        trim: bool = True,
        cache: bool = False,
    ):
        super().__init__(root_path, grid, fn_format=fn_format, cache=cache)
        self.count_var = count_var
        self.instance_dim = instance_dim
        self.instance_id_var = instance_id_var
        self.trim = trim

    def _open_cell(self, cell: int) -> ContiguousRaggedArray:
        return ContiguousRaggedArray.from_file(
            self._cell_filename(cell),
            count_var=self.count_var,
            instance_dim=self.instance_dim,
            instance_id_var=self.instance_id_var,
            trim=self.trim,
        )


class GriddedIndexedRaggedArray(GriddedRaggedArray):
    """
    Read gridded indexed ragged-array cell files.

    See :class:`GriddedRaggedArray` for the gridded reading and caching
    behavior. Each cell is read with
    :class:`~ascat.ragged_array.IndexedRaggedArray`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files.
    grid : pygeogrids.grids.CellGrid or str
        Grid object or registry name.
    fn_format : str, optional
        Cell file name format (default: "{:04d}.nc").
    index_var : str, optional
        Index variable name (default: "locationIndex").
    sample_dim : str, optional
        Sample dimension name (default: "obs").
    instance_id_var : str, optional
        Variable holding the instance identifiers, matching the grid gpi
        (default: "location_id").
    cache : bool, optional
        Keep every read cell in memory (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        index_var: str = "locationIndex",
        sample_dim: str = "obs",
        instance_id_var: str = "location_id",
        cache: bool = False,
    ):
        super().__init__(root_path, grid, fn_format=fn_format, cache=cache)
        self.index_var = index_var
        self.sample_dim = sample_dim
        self.instance_id_var = instance_id_var

    def _open_cell(self, cell: int) -> IndexedRaggedArray:
        return IndexedRaggedArray.from_file(
            self._cell_filename(cell),
            index_var=self.index_var,
            sample_dim=self.sample_dim,
            instance_id_var=self.instance_id_var,
        )


class GriddedOrthoMultiArray(GriddedRaggedArray):
    """
    Read gridded orthogonal multidimensional array cell files (CF 9.3.1).

    See :class:`GriddedRaggedArray` for the gridded reading and caching
    behavior. Each cell is read with
    :class:`~ascat.ragged_array.OrthogonalMultidimArray`; all locations in a
    cell share the same element (e.g. time) axis.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files.
    grid : pygeogrids.grids.CellGrid or str
        Grid object or registry name.
    fn_format : str, optional
        Cell file name format (default: "{:04d}.nc").
    instance_dim : str, optional
        Instance dimension name (default: "locations").
    element_dim : str, optional
        Element dimension name (default: "time").
    element_coord : str, optional
        Shared element coordinate variable name (default: ``element_dim``).
    instance_id_var : str, optional
        Variable holding the instance identifiers, matching the grid gpi
        (default: "location_id").
    cache : bool, optional
        Keep every read cell in memory (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        instance_dim: str = "locations",
        element_dim: str = "time",
        element_coord: str = None,
        instance_id_var: str = "location_id",
        cache: bool = False,
    ):
        super().__init__(root_path, grid, fn_format=fn_format, cache=cache)
        self.instance_dim = instance_dim
        self.element_dim = element_dim
        self.element_coord = element_coord
        self.instance_id_var = instance_id_var

    def _open_cell(self, cell: int) -> OrthogonalMultidimArray:
        return OrthogonalMultidimArray.from_file(
            self._cell_filename(cell),
            instance_dim=self.instance_dim,
            element_dim=self.element_dim,
            element_coord=self.element_coord,
            instance_id_var=self.instance_id_var,
        )


def _pad_element_dim(ds, element_dim, target):
    """Pad the element dimension of an incomplete array to ``target`` columns."""
    size = ds.sizes.get(element_dim, 0)
    if size >= target:
        return ds
    extra = target - size
    new_vars = {}
    for v in ds.variables:
        if element_dim in ds[v].dims:
            axis = ds[v].dims.index(element_dim)
            width = [(0, 0)] * ds[v].ndim
            width[axis] = (0, extra)
            arr = np.pad(np.asarray(ds[v].values), width,
                         constant_values=fill_value(ds[v].dtype))
            new_vars[v] = (ds[v].dims, arr, dict(ds[v].attrs))
        else:
            new_vars[v] = ds[v]
    # rebuild in one step (avoids a transient inconsistent element size), then
    # restore which variables were coordinates
    coord_names = [c for c in ds.coords]
    result = xr.Dataset(new_vars)
    result = result.set_coords([c for c in coord_names if c in result.variables])
    result.attrs = dict(ds.attrs)
    for v in ds.variables:
        result[v].encoding = ds[v].encoding
    return result


def _cell_incomplete_block(task):
    """
    Read one contiguous ragged cell and return its padded incomplete block.

    Runs in a worker process, so it returns plain arrays (not an open dataset):
    ``(coord_names, {var: (dims, array, attrs)})``, with the instance dimension
    coordinate dropped (it is set once on the store skeleton).
    """
    fn, opts = task
    cra = ContiguousRaggedArray.from_file(
        fn, count_var=opts["count_var"], instance_dim=opts["instance_dim"],
        instance_id_var=opts["instance_id_var"], trim=opts["trim"])
    ds = cra.to_incomplete().ds
    if cra.sample_dim != opts["element_dim"]:
        ds = ds.rename({cra.sample_dim: opts["element_dim"]})
    ds = _pad_element_dim(ds, opts["element_dim"], opts["max_elem"])
    cra.ds.close()
    ds = ds.drop_vars(opts["instance_dim"])
    coord_names = [str(c) for c in ds.coords]
    data = {str(v): (ds[v].dims, np.asarray(ds[v].values), dict(ds[v].attrs))
            for v in ds.variables}
    return coord_names, data


def cells_to_incomplete_zarr(
    root_path,
    store,
    fn_pattern: str = "*.nc",
    count_var: str = "row_size",
    instance_dim: str = "locations",
    instance_id_var: str = "location_id",
    element_dim: str = "element",
    trim: bool = True,
    chunks: dict = None,
    shards: dict = None,
    n_workers: int = 1,
):
    """
    Convert contiguous ragged cell files to a monolithic incomplete Zarr store.

    Every cell file under ``root_path`` is read, converted to an incomplete
    multidimensional array (CF 9.3.2) — a dense ``(instance, element)`` block
    padded with fill values — and written into its slice of one store spanning
    every location across all cells. The measurements become
    ``(instance, element)`` arrays and the per-observation coordinates (e.g.
    time) are carried as matching 2-D arrays.

    An empty store is created first with the requested chunk/shard layout, then
    each cell is written into a disjoint region of the instance dimension, so
    the full array is never held in memory and cells can be processed in
    parallel. Zarr compresses the fill padding on disk.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the contiguous ragged cell files (searched
        recursively).
    store : str or pathlib.Path or MutableMapping
        Target Zarr store.
    fn_pattern : str, optional
        Glob pattern for the cell files (default: "*.nc").
    count_var : str, optional
        Count variable name (default: "row_size").
    instance_dim : str, optional
        Instance dimension name (default: "locations").
    instance_id_var : str, optional
        Instance identifier variable (default: "location_id").
    element_dim : str, optional
        Name of the (positional) element dimension in the output
        (default: "element").
    trim : bool, optional
        Drop fill/padding locations when reading each cell (default: True).
    chunks : dict, optional
        Chunk size per dimension, e.g. ``{"locations": 5000, "element": 500}``.
        Dimensions not given are a single chunk.
    shards : dict, optional
        Shard size per dimension (Zarr v3). Each shard size must be a multiple
        of the corresponding chunk size.
    n_workers : int, optional
        Number of worker threads for reading/converting cells (default: 1).
        Writes to the store are serialised; the parallelism speeds up the
        per-cell conversion.

    Returns
    -------
    store : str or pathlib.Path or MutableMapping
        The store that was written.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    import dask.array as da

    root = Path(root_path)
    files = sorted(root.glob("**/" + fn_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{fn_pattern}' under '{root}'.")
    chunks = dict(chunks or {})

    def read(fn):
        return ContiguousRaggedArray.from_file(
            fn, count_var=count_var, instance_dim=instance_dim,
            instance_id_var=instance_id_var, trim=trim)

    def incomplete(cra):
        ds = cra.to_incomplete().ds
        if cra.sample_dim != element_dim:
            ds = ds.rename({cra.sample_dim: element_dim})
        return ds

    # pass 1: layout (offset + size per cell), global max elements, ids, schema
    layout, id_parts, offset, max_elem, schema = [], [], 0, 0, None
    for fn in files:
        cra = read(fn)
        if cra.size == 0:
            cra.ds.close()
            continue
        max_elem = max(max_elem, int(np.asarray(cra.ds[count_var]).max()))
        layout.append((fn, offset, cra.size))
        id_parts.append(np.asarray(cra.instance_ids))
        offset += cra.size
        if schema is None:
            schema = incomplete(cra)
        cra.ds.close()
    if offset == 0:
        raise ValueError("No valid locations found in the cell files.")
    total = offset
    instance_ids = np.concatenate(id_parts)
    schema = _pad_element_dim(schema, element_dim, max_elem)

    # build an empty skeleton with the requested chunk/shard layout
    dim_size = {instance_dim: total, element_dim: max_elem}
    skel = {}
    for v in schema.variables:
        if v == instance_dim:
            continue
        dims = schema[v].dims
        shape = tuple(dim_size.get(d, schema.sizes[d]) for d in dims)
        cshape = tuple(min(chunks.get(d, s), s) for d, s in zip(dims, shape))
        skel[v] = (dims, da.full(shape, fill_value(schema[v].dtype),
                                 dtype=schema[v].dtype, chunks=cshape),
                   dict(schema[v].attrs))
    skeleton = xr.Dataset(skel).assign_coords({instance_dim: instance_ids})
    skeleton = skeleton.set_coords(
        [c for c in schema.coords if c in skeleton.variables
         and c != instance_dim])

    encoding = {}
    for v in skeleton.variables:
        dims = skeleton[v].dims
        enc = {"chunks": tuple(
            min(chunks.get(d, skeleton.sizes[d]), skeleton.sizes[d])
            for d in dims)}
        fv = fill_value(skeleton[v].dtype)
        if fv is not None and not np.issubdtype(skeleton[v].dtype,
                                                np.datetime64):
            enc["_FillValue"] = fv
        if shards:
            enc["shards"] = tuple(shards.get(d, skeleton.sizes[d]) for d in dims)
        encoding[v] = enc
    # metadata-only skeleton; the actual data is filled by the region writes
    # below (serialised with a lock), so the chunk-alignment guard is bypassed
    skeleton.to_zarr(store, mode="w", compute=False, encoding=encoding,
                     safe_chunks=False)

    # pass 2: convert each cell (optionally in parallel worker *processes* —
    # netCDF4 is not thread-safe) and write it into its region. Writes stay in
    # the main process so overlapping boundary chunks are never written
    # concurrently.
    opts = dict(count_var=count_var, instance_dim=instance_dim,
                instance_id_var=instance_id_var, element_dim=element_dim,
                trim=trim, max_elem=max_elem)

    def write_block(off, n, coord_names, data):
        block = xr.Dataset(data).set_coords(
            [c for c in coord_names if c in data])
        block.to_zarr(store, region={instance_dim: slice(off, off + n)},
                      safe_chunks=False)

    if n_workers and n_workers > 1:
        # spawn (not fork) so worker interpreters have no inherited HDF5 state
        ctx = multiprocessing.get_context("spawn")
        tasks = [(fn, opts) for fn, _, _ in layout]
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=ctx) as pool:
            for (fn, off, n), (coord_names, data) in zip(
                    layout, pool.map(_cell_incomplete_block, tasks)):
                write_block(off, n, coord_names, data)
    else:
        for fn, off, n in layout:
            coord_names, data = _cell_incomplete_block((fn, opts))
            write_block(off, n, coord_names, data)

    return store
