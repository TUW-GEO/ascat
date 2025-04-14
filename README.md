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
Unfortunately neither BUFR nor GRIB readers work on Windows so if you need these
formats then Linux or OS X are your only options.

The following script will download and install all the needed packages.

Linux:

> ```bash
> conda env create -f environment.yml
> ```

Windows:

> ```bash
> conda env create -f environment_win.yml
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

## Command line interface ##

The latest ASCAT swath files (H122, H29, H129, H121) can be aggregated and/or regridded using a command line interface (CLI) provided by the ascat package.

### Aggregation of ASCAT SSM swath files ###

Surface soil moisture and backscatter40 from ASCAT swath files can be aggregated over a user-defined time period (e.g. 1 day, 10 days, 1 month) choosing one of the following methods: "mean", "median", "mode", "std", "min", "max", "argmin", "argmax", "quantile", "first", "last". The time span for processing is determined by the start and end times specified. Additionally, thresholds can be set for masks - such as those for frozen soil probability, snow cover probability, subsurface scattering probability, and surface soil moisture sensitivity - to filter surface soil moisture data as part of the aggregation process.

> ```bash
> ascat_swath_agg /path/to/input/h129_v1.0/swaths/ /path/to/output --start_dt 2020-06-15T00:00:00 --end_dt 2020-06-17T00:00:00 --t_delta 1D --agg mean --snow_cover_mask 80 --frozen_soil_mask 80 --subsurface_scattering_mask 10 --ssm_sensitivity_mask 1
> ```

There is also an option that no masking is applied using the argument ``--no-mask``.

### Re-gridding of ASCAT SSM swath files ###

ASCAT swath files contain data that are provided on a Discrete Global Grid (DGG) and can be converted to a regular lat/lon grid using a nearest neighbor approach. Either a single swath file or folder containing the swath files can be used as input argument.

> ```bash
> ascat_swath_regrid /path/to/input/file /path/to/output 0.1 --grid_store /path/to/tmp/folder --suffix _regrid_0.1deg
> ```

### Resampling of ASCAT SSM swath files ###

ASCAT swath files contain data that are provided on a Discrete Global Grid (DGG) and can be converted to a regular lat/lon grid using an inverse distance weighting. Either a single swath file or folder containing the swath files can be used as input argument.

> ```bash
> ascat_swath_resample /path/to/input/file /path/to/output 0.1 --grid_store /path/to/tmp/folder --suffix _resample_0.1deg --neighbour 6 --radius 10000
> ```

## Contribute ##

We are happy if you want to contribute. Please raise an issue explaining what is
missing or if you find a bug. We will also gladly accept pull requests for new
features or bug fixes.

## Guidelines ##

If you want to contribute please follow these steps:

- Fork the ascat repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get the test data repository.
- Make a new feature branch from the ascat master branch
- Add your feature
- Please include tests for your contributions in one of the test directories
- Submit a pull request
