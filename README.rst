=====
ascat
=====

.. image:: https://github.com/TUW-GEO/ascat/workflows/ascat_ubuntu/badge.svg
   :target: https://github.com/TUW-GEO/ascat/actions/workflows/ascat_ubuntu.yml

.. image:: https://github.com/TUW-GEO/ascat/workflows/ascat_windows/badge.svg
   :target: https://github.com/TUW-GEO/ascat/actions/workflows/ascat_windows.yml

.. image:: https://coveralls.io/repos/github/TUW-GEO/ascat/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/ascat?branch=master

.. image:: https://badge.fury.io/py/ascat.svg
    :target: http://badge.fury.io/py/ascat

.. image:: https://readthedocs.org/projects/ascat/badge/?version=latest
   :target: http://ascat.readthedocs.org/

Read and convert data acquired by ASCAT on-board the series of Metop satellites.

Works great in combination with `pytesmo <https://github.com/TUW-GEO/pytesmo>`_.

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4610836.svg
   :target: https://doi.org/10.5281/zenodo.4610836

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.4610836 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
============

ASCAT data are distributed in BUFR, NetCDF, EPS Native and GRIB format. Unfortunately neither BUFR nor GRIB readers work on Windows so if you need these formats then Linux or OS X are your only options.

The following script will download and install all the needed packages.

Linux:

.. code::

    conda env create -f environment.yml

Windows:

.. code::

    conda env create -f environment_win.yml


Supported datasets
==================

This gives a short overview over the supported products. Please see the documentation for detailed examples of how to work with a product.

Read ASCAT data from different sources into a common format supported by pytesmo.

- `H SAF <http://h-saf.eumetsat.int/>`_
    - Surface Soil Moisture (SSM) and Root Zone Soil Moisture (RZSM) products
- `Copernicus Global Land Service (CGLS) <http://land.copernicus.eu/global/products/swi>`_
    - CGLS Soil Water Index (SWI) products
- `EUMETSAT <https://navigator.eumetsat.int/search?query=ascat/>`_
    - ASCAT Soil Moisture at 12.5 km Swath Grid - Metop
    - ASCAT Soil Moisture at 25 km Swath Grid - Metop
    - ASCAT GDS Level 1 Sigma0 resampled at 12.5 km Swath Grid - Metop 
    - ASCAT GDS Level 1 Sigma0 resampled at 25 km Swath Grid - Metop 

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what is missing or if you find a bug. We will also gladly accept pull requests for new features or bug fixes.

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the ascat repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get the test data repository.
- Make a new feature branch from the ascat master branch
- Add your feature
- Please include tests for your contributions in one of the test directories
- Submit a pull request

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
