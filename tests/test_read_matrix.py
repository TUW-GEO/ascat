# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Read every product in all combinations of the generic and the xarray format.

A reader returns either the fields of the file or the generic format, as a
numpy array or as an xarray.Dataset. The four combinations share most of their
code but differ in the last steps, so one of them can break while the others
keep working, and a test which only reads the way it always did will not
notice. They are therefore checked together, for every product a reader
supports.

No SCA test data is shipped with the package, so those products are written
by the helpers of the SCA tests.
"""

from datetime import datetime

import numpy as np
import numpy.testing as nptest
import pytest
import xarray as xr

from ascat.eumetsat.level1 import AscatL1bFile
from ascat.eumetsat.level2 import AscatL2File
from ascat.eumetsat.sca.level1 import ScaL1bFile

from get_path import get_testdata_path
from test_sca import _name, write_szf, write_szr

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning")

#: Products which come with the package, "id": (reader, path, one per beam).
ASCAT_PRODUCTS = {
    "ascat_l1b_szf_eps": (
        AscatL1bFile, "eps_nat/ASCA_SZF_1B_M01_20180611041800Z"
        "_20180611055959Z_N_O_20180611050637Z.nat", True),
    "ascat_l1b_szf_hdf5": (
        AscatL1bFile, "hdf5/ASCA_SZF_1B_M01_20180611041800Z"
        "_20180611055959Z_N_O_20180611050637Z.h5", True),
    "ascat_l1b_szr_eps_fmv11": (
        AscatL1bFile, "eps_nat/ASCA_SZR_1B_M02_20071212071500Z"
        "_20071212085659Z_R_O_20081225063118Z.nat", False),
    "ascat_l1b_szr_eps_fmv12": (
        AscatL1bFile, "eps_nat/ASCA_SZR_1B_M02_20100609013900Z"
        "_20100609032058Z_R_O_20130824233100Z.nat", False),
    "ascat_l1b_szr_nc": (
        AscatL1bFile, "nc/W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPA"
        "+ASCAT_C_EUMP_20100609013900_18872_eps_o_125_l1.nc", False),
    "ascat_l1b_szr_bufr": (
        AscatL1bFile, "bufr/M02-ASCA-ASCSZR1B0200-NA-9.1-20100609013900"
        ".000000000Z-20130824233100-1280350.bfr", False),
    "ascat_l2_smo_eps": (
        AscatL2File, "eps_nat/ASCA_SMO_02_M01_20180612035700Z"
        "_20180612053856Z_N_O_20180612044530Z.nat", False),
    "ascat_l2_smo_bufr": (
        AscatL2File, "bufr/M01-ASCA-ASCSMO02-NA-5.0-20180612035700"
        ".000000000Z-20180612044530-1281300.bfr", False),
    "ascat_l2_ssm_nc": (
        AscatL2File, "nc/W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,METOPB"
        "+ASCAT_C_EUMP_20180612035700_29742_eps_o_250_ssm_l2.nc", False),
}

SCA_PRODUCTS = {
    "sca_l1b_szf": (ScaL1bFile, write_szf, True),
    "sca_l1b_szr": (ScaL1bFile, write_szr, False),
}

PRODUCT_IDS = sorted(ASCAT_PRODUCTS) + sorted(SCA_PRODUCTS)


@pytest.fixture(scope="session")
def sca_files(tmp_path_factory):
    """The SCA products, written once for the whole session."""
    path = tmp_path_factory.mktemp("sca")
    start = datetime(2026, 1, 2, 3, 4, 5)

    return {"sca_l1b_szf": write_szf(path / _name("SZF", start), start),
            "sca_l1b_szr": write_szr(path / _name("SZR", start), start)}


@pytest.fixture(params=PRODUCT_IDS)
def product(request, sca_files):
    """A reader, a file it can read, and whether it returns one entry per beam."""
    if request.param in ASCAT_PRODUCTS:
        reader, name, per_beam = ASCAT_PRODUCTS[request.param]
        data_path = get_testdata_path() / "eumetsat" / "ASCAT_generic_reader_data"
        return reader, data_path / name, per_beam

    reader, _, per_beam = SCA_PRODUCTS[request.param]

    return reader, sca_files[request.param], per_beam


#: Files are read once per format and kept, reading them is the slow part.
_read_cache = {}


def read(product, generic, to_xarray):
    """Read a product, or return what a previous test already read."""
    reader, path, per_beam = product
    key = (str(path), generic, to_xarray)

    if key not in _read_cache:
        _read_cache[key] = reader(path).read(generic=generic,
                                             to_xarray=to_xarray)

    return _read_cache[key]


def one(data, per_beam):
    """The dataset of the first beam, or the dataset itself."""
    return next(iter(data.values())) if per_beam else data


def fields(dataset):
    """Names of everything a dataset holds, whether numpy or xarray."""
    if isinstance(dataset, xr.Dataset):
        return set(dataset.data_vars) | set(dataset.coords)

    return set(dataset.dtype.names)


def named(dataset, prefix):
    """The one field starting with prefix, e.g. "lon" or "longitude"."""
    matching = sorted(f for f in fields(dataset) if f.startswith(prefix))
    assert len(matching) == 1, f"expected one {prefix}* field, got {matching}"

    return matching[0]


def values(dataset, name):
    """The values of a field, whether numpy or xarray."""
    if isinstance(dataset, xr.Dataset):
        return np.asarray(dataset[name].values).ravel()

    return np.ma.getdata(dataset[name]).ravel()


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("to_xarray", [False, True])
def test_read(product, generic, to_xarray):
    """Every combination reads and returns the type belonging to it."""
    _, _, per_beam = product
    data, metadata = read(product, generic, to_xarray)
    dataset = one(data, per_beam)

    assert isinstance(dataset, xr.Dataset if to_xarray else np.ndarray)
    assert fields(dataset)
    assert metadata


@pytest.mark.parametrize("generic", [False, True])
def test_numpy_and_xarray_hold_the_same_fields(product, generic):
    """Asking for xarray changes the container, not what is in it."""
    _, _, per_beam = product
    as_numpy = one(read(product, generic, False)[0], per_beam)
    as_xarray = one(read(product, generic, True)[0], per_beam)

    assert fields(as_numpy) == fields(as_xarray)


@pytest.mark.parametrize("to_xarray", [False, True])
def test_generic_renames_the_coordinates(product, to_xarray):
    """The generic format uses "lon" and "lat", the file its own names."""
    _, _, per_beam = product
    original = one(read(product, False, to_xarray)[0], per_beam)
    generic = one(read(product, True, to_xarray)[0], per_beam)

    assert {"lon", "lat"} <= fields(generic)
    assert not {"lon", "lat"} & (fields(original) - fields(generic))


@pytest.mark.parametrize("to_xarray", [False, True])
def test_the_coordinates_survive_the_rename(product, to_xarray):
    """Renaming a coordinate for the generic format does not change it."""
    _, _, per_beam = product
    original = one(read(product, False, to_xarray)[0], per_beam)
    generic = one(read(product, True, to_xarray)[0], per_beam)

    for prefix in ("lon", "lat"):
        before = values(original, named(original, prefix))
        after = values(generic, prefix)
        assert before.size == after.size
        nptest.assert_allclose(before.astype("f8"), after.astype("f8"),
                               atol=1e-3)


@pytest.mark.parametrize("generic", [False, True])
def test_dataset_can_be_written(product, generic, tmp_path):
    """The metadata a dataset carries has to survive being written."""
    _, _, per_beam = product
    data, _ = read(product, generic, True)

    one(data, per_beam).to_netcdf(tmp_path / f"{generic}.nc")
