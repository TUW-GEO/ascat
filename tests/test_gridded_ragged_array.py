# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

import numpy as np
import pytest
import xarray as xr
from pygeogrids.grids import CellGrid

from ascat.ragged_array import open_cf
from ascat.gridded_ragged_array import (
    GridRegistry,
    GriddedContiguousRaggedArray,
    GriddedIndexedRaggedArray,
    GriddedOrthoMultiArray,
    cells_to_incomplete_zarr,
    grid_registry,
)


# grid: 5 grid points in two cells (0 and 5); gpi 12 has no data in its file
GPIS = np.array([10, 11, 12, 20, 21])
LONS = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
LATS = np.array([0.0, 1.0, 2.0, 40.0, 41.0])
CELLS = np.array([0, 0, 0, 5, 5])


def _grid():
    return CellGrid(LONS, LATS, CELLS, gpis=GPIS)


def _contiguous_cell_ds(location_ids, row_sizes):
    # only valid (non-negative) counts contribute observations; fill/padding
    # locations carry a negative fill count
    n = int(np.sum([r for r in row_sizes if r >= 0]))
    return xr.Dataset(
        {
            "row_size": (("locations",), np.array(row_sizes, dtype=np.int64),
                         {"sample_dimension": "obs"}),
            "location_id": (("locations",),
                            np.array(location_ids, dtype=np.int64)),
            "sm": (("obs",), np.arange(n, dtype="float32")),
        },
    )


def _indexed_cell_ds(location_ids, row_sizes):
    n = int(np.sum(row_sizes))
    loc_index = np.repeat(np.arange(len(location_ids)),
                          row_sizes).astype("int32")
    return xr.Dataset(
        {
            "locationIndex": (("obs",), loc_index,
                              {"instance_dimension": "locations"}),
            "location_id": (("locations",),
                            np.array(location_ids, dtype=np.int64)),
            "sm": (("obs",), np.arange(n, dtype="float32")),
        },
    )


@pytest.fixture()
def contig_root(tmp_path):
    # cell 0 holds gpis 10, 11 (but NOT 12); cell 5 holds gpis 20, 21
    _contiguous_cell_ds([10, 11], [2, 3]).to_netcdf(tmp_path / "0000.nc")
    _contiguous_cell_ds([20, 21], [1, 2]).to_netcdf(tmp_path / "0005.nc")
    return tmp_path


def _ortho_cell_ds(location_ids):
    n = len(location_ids)
    return xr.Dataset(
        {
            "location_id": (("locations",),
                            np.array(location_ids, dtype=np.int64)),
            "sm": (("locations", "time"),
                   np.arange(n * 3, dtype="float32").reshape(n, 3)),
        },
        coords={"time": np.array(["2020-01-01", "2020-01-02", "2020-01-03"],
                                 dtype="datetime64[ns]")},
    )


@pytest.fixture()
def indexed_root(tmp_path):
    _indexed_cell_ds([10, 11], [2, 3]).to_netcdf(tmp_path / "0000.nc")
    _indexed_cell_ds([20, 21], [1, 2]).to_netcdf(tmp_path / "0005.nc")
    return tmp_path


@pytest.fixture()
def ortho_root(tmp_path):
    _ortho_cell_ds([10, 11]).to_netcdf(tmp_path / "0000.nc")
    _ortho_cell_ds([20, 21]).to_netcdf(tmp_path / "0005.nc")
    return tmp_path


# --------------------------------------------------------------------------- #
# contiguous
# --------------------------------------------------------------------------- #
def test_contiguous_read_by_gpi(contig_root):
    gra = GriddedContiguousRaggedArray(contig_root, _grid())
    np.testing.assert_array_equal(gra.read(gpi=10)["sm"].values, [0., 1.])
    # gpi 21 is the 2nd location in cell 5 (row_start 1, row_size 2)
    np.testing.assert_array_equal(gra.read(gpi=21)["sm"].values, [1., 2.])


def test_contiguous_read_by_lonlat(contig_root):
    gra = GriddedContiguousRaggedArray(contig_root, _grid())
    by_gpi = gra.read(gpi=10)["sm"].values
    by_coords = gra.read(lon=0.0, lat=0.0)["sm"].values  # nearest gpi is 10
    np.testing.assert_array_equal(by_gpi, by_coords)


def test_contiguous_missing_gpi_returns_none(contig_root):
    # gpi 12 is a valid grid point in cell 0, but absent from the cell file
    gra = GriddedContiguousRaggedArray(contig_root, _grid())
    assert gra.read(gpi=12) is None


def test_contiguous_fn_format_and_trim(tmp_path):
    # custom filename format + fill/padding location that trim should drop
    ds = _contiguous_cell_ds([10, 11, np.iinfo(np.int64).min],
                             [2, 3, np.iinfo(np.int64).min])
    ds.to_netcdf(tmp_path / "H120_0000.nc")
    gra = GriddedContiguousRaggedArray(
        tmp_path, _grid(), fn_format="H120_{:04d}.nc", trim=True)
    assert gra.read_cell(0).size == 2  # padding location trimmed
    np.testing.assert_array_equal(gra.read(gpi=11)["sm"].values, [2., 3., 4.])


# --------------------------------------------------------------------------- #
# indexed
# --------------------------------------------------------------------------- #
def test_indexed_read_by_gpi(indexed_root):
    gra = GriddedIndexedRaggedArray(indexed_root, _grid())
    np.testing.assert_array_equal(gra.read(gpi=10)["sm"].values, [0., 1.])
    # gpi 21 is the 2nd location in cell 5 (locationIndex == 1 -> obs 1, 2)
    np.testing.assert_array_equal(gra.read(gpi=21)["sm"].values, [1., 2.])


def test_indexed_read_by_lonlat(indexed_root):
    gra = GriddedIndexedRaggedArray(indexed_root, _grid())
    by_gpi = gra.read(gpi=11)["sm"].values
    lon, lat = 1.0, 1.0  # nearest gpi is 11
    np.testing.assert_array_equal(gra.read(lon=lon, lat=lat)["sm"].values,
                                  by_gpi)


# --------------------------------------------------------------------------- #
# grid registry
# --------------------------------------------------------------------------- #
def test_grid_registry_register_get_cache():
    # isolated instance -> no leakage into the shared registry
    reg = GridRegistry()
    grid = _grid()
    reg.register("obj", grid)
    assert reg.get("obj") is grid

    calls = []

    def factory():
        calls.append(1)
        return _grid()

    reg.register("factory", factory)
    assert reg.get("factory") is reg.get("factory")   # cached
    assert len(calls) == 1                             # factory called once

    with pytest.raises(KeyError):
        reg.get("no_such_grid")


def test_registry_instances_are_independent():
    a, b = GridRegistry(), GridRegistry()
    a.register("only_in_a", _grid())
    assert a.get("only_in_a") is not None
    with pytest.raises(KeyError):
        b.get("only_in_a")


def test_reader_accepts_grid_name(contig_root):
    # the readers use the shared module-level registry for name lookups
    grid_registry.register("test_reader_grid", _grid())
    gra = GriddedContiguousRaggedArray(contig_root, "test_reader_grid")
    np.testing.assert_array_equal(gra.read(gpi=10)["sm"].values, [0., 1.])


# --------------------------------------------------------------------------- #
# orthogonal multidimensional array
# --------------------------------------------------------------------------- #
def test_ortho_read_by_gpi(ortho_root):
    gra = GriddedOrthoMultiArray(ortho_root, _grid())
    # location 11 is position 1 in cell 0 -> sm row 1
    np.testing.assert_array_equal(gra.read(gpi=11)["sm"].values, [3., 4., 5.])
    # location 21 is position 1 in cell 5
    np.testing.assert_array_equal(gra.read(gpi=21)["sm"].values, [3., 4., 5.])
    # gpi 12 is on the grid (cell 0) but absent from the cell file
    assert gra.read(gpi=12) is None


def test_ortho_read_by_lonlat(ortho_root):
    gra = GriddedOrthoMultiArray(ortho_root, _grid())
    by_gpi = gra.read(gpi=11)["sm"].values
    np.testing.assert_array_equal(gra.read(lon=1.0, lat=1.0)["sm"].values,
                                  by_gpi)


# --------------------------------------------------------------------------- #
# shared behavior (base class), exercised through all subclasses
# --------------------------------------------------------------------------- #
ALL_READERS = [
    (GriddedContiguousRaggedArray, "contig_root"),
    (GriddedIndexedRaggedArray, "indexed_root"),
    (GriddedOrthoMultiArray, "ortho_root"),
]


@pytest.mark.parametrize("cls,fixture", ALL_READERS)
def test_read_multiple_gpis(cls, fixture, request):
    gra = cls(request.getfixturevalue(fixture), _grid())
    gpis = [21, 10, 12]  # spans cells 5 and 0; gpi 12 has no data
    out = gra.read(gpi=gpis)
    assert isinstance(out, dict)
    assert list(out.keys()) == gpis          # returned in request order
    assert out[12] is None                   # absent grid point
    for g in (21, 10):
        np.testing.assert_array_equal(out[g]["sm"].values,
                                      gra.read(gpi=g)["sm"].values)


@pytest.mark.parametrize("cls,fixture", ALL_READERS)
def test_read_multiple_lonlats(cls, fixture, request):
    gra = cls(request.getfixturevalue(fixture), _grid())
    # coordinates of gpis 10 and 20
    out = gra.read(lon=[LONS[0], LONS[3]], lat=[LATS[0], LATS[3]])
    assert set(out) == {10, 20}
    np.testing.assert_array_equal(out[10]["sm"].values,
                                  gra.read(gpi=10)["sm"].values)


@pytest.mark.parametrize("cls,fixture", ALL_READERS)
def test_requires_gpi_or_coords(cls, fixture, request):
    gra = cls(request.getfixturevalue(fixture), _grid())
    with pytest.raises(ValueError):
        gra.read()


@pytest.mark.parametrize("cls,fixture", ALL_READERS)
def test_max_dist(cls, fixture, request):
    gra = cls(request.getfixturevalue(fixture), _grid())
    # a point far from every grid point with a tight max_dist should raise
    with pytest.raises(ValueError):
        gra.read(lon=50.0, lat=20.0, max_dist=1.0)


@pytest.mark.parametrize("cls,fixture", ALL_READERS)
def test_missing_cell_file_raises(cls, fixture, request):
    gra = cls(request.getfixturevalue(fixture), _grid())
    with pytest.raises(FileNotFoundError):
        gra.read_cell(99)


@pytest.mark.parametrize("cls,fixture", ALL_READERS)
def test_close_releases_and_reopens(cls, fixture, request):
    root = request.getfixturevalue(fixture)
    gra = cls(root, _grid(), cache=True)
    gra.read(gpi=10)
    gra.read(gpi=20)
    assert sorted(gra._cells) == [0, 5]

    gra.close()
    assert gra._cells == {}
    assert gra._active_cell is None and gra._active_reader is None

    # closing again is a no-op, and reads still work afterwards (files reopen)
    gra.close()
    assert gra.read(gpi=10)["sm"].values.size > 0


@pytest.mark.parametrize("cls,fixture", [
    (GriddedContiguousRaggedArray, "contig_root"),
    (GriddedIndexedRaggedArray, "indexed_root"),
])
def test_caching_modes(cls, fixture, request):
    root = request.getfixturevalue(fixture)
    # non-caching keeps only the most recent cell
    gra = cls(root, _grid(), cache=False)
    gra.read(gpi=10)
    assert gra._active_cell == 0
    gra.read(gpi=20)
    assert gra._active_cell == 5
    assert gra._cells == {}

    # caching keeps every cell that was read
    gra_c = cls(root, _grid(), cache=True)
    gra_c.read(gpi=10)
    gra_c.read(gpi=20)
    assert sorted(gra_c._cells) == [0, 5]
    gra_c.clear_cache()
    assert gra_c._cells == {}


# --------------------------------------------------------------------------- #
# monolithic incomplete zarr conversion
# --------------------------------------------------------------------------- #
def _contig_cell(location_ids, row_sizes):
    n = int(np.sum(row_sizes))
    return xr.Dataset(
        {"sm": (("obs",), np.arange(n, dtype="float32")),
         "row_size": (("locations",), np.array(row_sizes, dtype=np.int64),
                      {"sample_dimension": "obs"}),
         "location_id": (("locations",),
                         np.array(location_ids, dtype=np.int64))},
    )


def test_cells_to_incomplete_zarr(tmp_path):
    # two cells with different locations and different observation counts
    _contig_cell([10, 11], [2, 3]).to_netcdf(tmp_path / "0001.nc")
    _contig_cell([20], [1]).to_netcdf(tmp_path / "0002.nc")
    store = str(tmp_path / "out.zarr")

    cells_to_incomplete_zarr(tmp_path, store, fn_pattern="000*.nc",
                             element_dim="element",
                             chunks={"locations": 2, "element": 2})

    z = xr.open_zarr(store)
    assert z.sizes["locations"] == 3          # 10, 11, 20 across both cells
    assert z.sizes["element"] == 3            # global max row_size
    assert z["sm"].dims == ("locations", "element")
    # the requested chunking is applied
    import zarr
    assert zarr.open(store)["sm"].chunks == (2, 2)

    inc = open_cf(store, instance_id_var="location_id",
                  instance_dim="locations", element_dim="element",
                  engine="zarr")
    # each location's real observations sit in the leading columns
    np.testing.assert_array_equal(inc.sel_instance(11)["sm"].values[:3],
                                  [2., 3., 4.])
    assert inc.sel_instance(20)["sm"].values[0] == 0.
    assert set(inc.instance_ids) == {10, 11, 20}
