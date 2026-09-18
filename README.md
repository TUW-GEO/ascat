# ascat #

[![PyPi](https://img.shields.io/pypi/v/ascat)](https://pypi.org/project/ascat/)
[![Readthedocs](https://readthedocs.org/projects/ascat/badge/?version=latest)](http://ascat.readthedocs.org/)
[![Downloads](https://img.shields.io/pypi/dm/ascat)](https://pypi.org/project/ascat)
[![Linux actions status](https://github.com/TUW-GEO/ascat/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/TUW-GEO/ascat/actions/workflows/ubuntu.yml)
[![Windows actions status](https://github.com/TUW-GEO/ascat/actions/workflows/windows.yml/badge.svg)](https://github.com/TUW-GEO/ascat/actions/workflows/windows.yml)
[![Coveralls](https://coveralls.io/repos/github/TUW-GEO/ascat/badge.svg?branch=master)](https://coveralls.io/github/TUW-GEO/ascat?branch=master)

Read and visualize data from the Advanced Scatterometer (ASCAT) on-board the series of Metop satellites.

## Citation ##

[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.4610836.svg)](https://doi.org/10.5281/zenodo.4610836)

If you use the software in a publication then please cite it using the Zenodo
DOI. Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.4610836 to
get the DOI of that version. You should normally always use the DOI for the
specific version of your record in citations. This is to ensure that other
researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at
http://help.zenodo.org/#versioning

## Installation ##

ASCAT data are distributed in BUFR, NetCDF, EPS Native and GRIB format.
Unfortunately the GRIB reader does not work on Windows, so if you need that
format then Linux or OS X are your only options.

Install the latest release from PyPI:

> ```bash
> pip install ascat
> ```

Reading H SAF GRIB products requires `pygrib`, which is available through the
optional `grib` extra (Linux/OS X only):

> ```bash
> pip install ascat[grib]
> ```

Plotting requires `matplotlib` and `eomaps`. They are not installed with the
package; they are in the `plot` dependency group:

> ```bash
> uv sync --group plot
> ```

## Supported datasets ##

This gives a short overview over the supported products. Please see the documentation for detailed examples of how to work with a product.

Read ASCAT data from different sources into a common format supported by pytesmo.

- [H SAF](http://h-saf.eumetsat.int/)
    - Surface Soil Moisture (SSM) and Root Zone Soil Moisture (RZSM) products
- [Copernicus Global Land Service (CGLS)](http://land.copernicus.eu/global/products/swi)
    - CGLS Soil Water Index (SWI) products
- [EUMETSAT](https://navigator.eumetsat.int/search?query=ascat)
    - ASCAT Soil Moisture at 12.5 km Swath Grid - Metop
    - ASCAT Soil Moisture at 25 km Swath Grid - Metop
    - ASCAT GDS Level 1 Sigma0 resampled at 12.5 km Swath Grid - Metop
    - ASCAT GDS Level 1 Sigma0 resampled at 25 km Swath Grid - Metop
    - ASCAT GDS Level 1 Sigma0 at full resolution (SZF) - Metop
- [EUMETSAT](https://navigator.eumetsat.int/search?query=scatterometer)
    - EPS-SG SCA Level 1b full resolution backscatter (SZF) - Metop-SG
    - EPS-SG SCA Level 1b re-sampled backscatter (SZR) - Metop-SG

ASCAT Level 1b and Level 2 files are read in EPS Native, NetCDF, BUFR and HDF5
format, SCA Level 1b files in NetCDF. The readers return either the fields as
they are named in the file, or a generic format which is the same across
products and instruments:

> ```python
> from ascat.eumetsat.level1 import AscatL1bFile
> from ascat.eumetsat.sca.level1 import ScaL1bFile
>
> # one entry per antenna beam, "lf-vv", "lm-vv", ...
> ascat, metadata = AscatL1bFile("ASCA_SZF_1B_M03_...nat").read()
> sca, metadata = ScaL1bFile("W_XX-EUMETSAT-...SCA-1B-SZF...nc").read()
>
> ascat["lf-vv"]["sig"]  # backscatter, as sca["lf-vv"]["sig"]
>
> # the fields of the file instead, under their own names
> ascat, metadata = AscatL1bFile("ASCA_SZF_1B_M03_...nat").read(generic=False)
>
> # as xarray.Dataset rather than numpy.ndarray
> ascat, metadata = AscatL1bFile("ASCA_SZF_1B_M03_...nat").read(to_xarray=True)
> ```

## Command line interface ##

The package installs the following commands. Every one of them takes `--help`,
which lists its arguments.

| Command | Purpose |
| --- | --- |
| `hsaf_download` | Download products from H SAF |
| `eumetsat_download` | Download products from EUMETSAT |
| `ascat_swath_agg` | Aggregate swath files over a time period |
| `ascat_swath_regrid` | Put swath files on a regular grid, nearest neighbour |
| `ascat_swath_resample` | Put swath files on a regular grid, inverse distance |
| `ascat_swaths_to_cells` | Stack swath files into cell time series |
| `ascat_convert_cell_format` | Convert cell files between CF array formats |
| `plot_szf` | Plot a full resolution backscatter file on a map |

The products the swath and cell commands know are `H29`, `H121`, `H122`,
`H129`, `H130`, `H139`, `SIG0_6.25` and `SIG0_12.5`, and for cell files also
`ERSH` and `ERSN`.

### Downloading ###

Credentials are read from a file, see the documentation for its format.

> ```bash
> hsaf_download -cf credentials.ini -r /products/h129/ -o /path/to/output \
>     -from 20200615 -to 20200617
>
> eumetsat_download -cf credentials.ini -p EO:EUM:DAT:METOP:ASCSZF1B \
>     -o /path/to/output -from 20200615 -to 20200617 -mw 4
> ```

### Aggregation of ASCAT SSM swath files ###

Surface soil moisture and backscatter40 from ASCAT swath files can be
aggregated over a user-defined time period (e.g. 1 day, 10 days, 1 month)
choosing one of the following methods: "mean", "median", "mode", "std", "min",
"max", "argmin", "argmax", "quantile", "first", "last". The time span for
processing is determined by the start and end times specified. Additionally,
thresholds can be set for masks - such as those for frozen soil probability,
snow cover probability, subsurface scattering probability, and surface soil
moisture sensitivity - to filter surface soil moisture data as part of the
aggregation process.

By default, data from all Metop satellites will be aggregated together. To
aggregate from a subset of satellites, pass a regex matching any combination of
`"A"`, `"B"`, and `"C"` to the `--sat` argument. E.g. `--sat A`, `--sat [AB]`,
`--sat *`, `--sat [ABC]`, etc.

> ```bash
> ascat_swath_agg /path/to/input/h129_v1.0/swaths/ /path/to/output \
>     --start_dt 2020-06-15T00:00:00 --end_dt 2020-06-17T00:00:00 \
>     --t_delta 1D --agg mean --snow_cover_mask 80 --frozen_soil_mask 80 \
>     --subsurface_scattering_mask 10 --ssm_sensitivity_mask 1 --sat [AB]
> ```

There is also an option that no masking is applied using the argument
``--no_masking``.

### Re-gridding of ASCAT SSM swath files ###

ASCAT swath files contain data that are provided on a Discrete Global Grid
(DGG) and can be converted to a regular lat/lon grid using a nearest neighbor
approach. Either a single swath file or folder containing the swath files can
be used as input argument.

> ```bash
> ascat_swath_regrid /path/to/input/file /path/to/output 0.1 \
>     --grid_store /path/to/tmp/folder --suffix _regrid_0.1deg
> ```

### Resampling of ASCAT SSM swath files ###

The same, using inverse distance weighting over a number of neighbours.

> ```bash
> ascat_swath_resample /path/to/input/file /path/to/output 0.1 \
>     --grid_store /path/to/tmp/folder --suffix _resample_0.1deg \
>     --neighbours 6 --radius 10000
> ```

### Stacking swath files into cell time series ###

Swath files are stored by acquisition time; stacking them gives one file per
grid cell, holding the time series of every grid point in that cell. The
keyword arguments at the end fill the placeholders of the file name pattern of
the product.

> ```bash
> ascat_swaths_to_cells /path/to/input/swaths /path/to/output H129 \
>     --start_date 2020-06-15 --end_date 2020-06-17 --dump_size 2GB \
>     sat=A year=2020
> ```

### Converting cell files between CF array formats ###

Cell files store their time series as one of the discrete sampling geometries
of the CF conventions, and can be converted between them.

> ```bash
> ascat_convert_cell_format /path/to/input/cells /path/to/output H129 contiguous
> ```

### Plotting a full resolution backscatter file ###

The measurements of an ASCAT or EPS-SG SCA Level 1b SZF file are put on a map,
coloured by whichever field is of interest. Which of the two instruments a file
belongs to follows from its name. Clicking a measurement shows what the file
says about it. This needs the `plot` dependency group, see above.

> ```bash
> plot_szf ASCA_SZF_1B_M03_20260619232100Z_...nat --parameter sig
>
> plot_szf W_XX-EUMETSAT-Darmstadt,SAT,SGB1-SCA-1B-SZF_...nc \
>     --parameter inc --size 2 --outpath map.png
> ```

A SZF file of ASCAT holds more than eight million measurements across its six
beams, so only every n-th one is drawn by default; `--max_points 0` draws all
of them. `--beams` selects which antenna beams to plot, out of `lf-vv`,
`lm-vv`, `la-vv`, `rf-vv`, `rm-vv` and `ra-vv`, which both instruments have,
and for SCA additionally the mid beams in HH, VH and HV polarization.

## Contribute ##

We are happy if you want to contribute. Please raise an issue explaining what is
missing or if you find a bug. We will also gladly accept pull requests for new
features or bug fixes.

## Guidelines ##

If you want to contribute please follow these steps:

- Fork the ascat repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get the test data repository.
- Set up the development environment with [uv](https://docs.astral.sh/uv/): ``uv sync``
  (add ``--extra grib`` for GRIB support, ``--group plot`` for plotting)
- Make a new feature branch from the ascat master branch
- Add your feature
- Please include tests for your contributions in one of the test directories
- Run the tests with ``uv run pytest``
- Submit a pull request
