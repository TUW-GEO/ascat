=====
ascat
=====

.. image:: https://travis-ci.org/TUW-GEO/ascat.svg?branch=master
    :target: https://travis-ci.org/TUW-GEO/ascat

.. image:: https://coveralls.io/repos/github/TUW-GEO/ascat/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/ascat?branch=master

.. image:: https://badge.fury.io/py/ascat.svg
    :target: http://badge.fury.io/py/ascat

.. image:: https://readthedocs.org/projects/ascat/badge/?version=latest
   :target: http://ascat.readthedocs.org/

Read and convert data acquired by ASCAT on-board the series of Metop satellites. Written in Python.

Works great in combination with `pytesmo <https://github.com/TUW-GEO/pytesmo>`_.

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596434.svg
   :target: https://doi.org/10.5281/zenodo.596434

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.596434 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
============

The packages you have to install depend on the features you want to use. The H SAF soil moisture products are disseminated in BUFR, NetCDF or GRIB format. In order to read them you will have to install the appropriate packages which will be explained shortly. Unfortunately neither BUFR nor GRIB readers work on Windows so if you need these formats then Linux or OS X are your only options.

For installation we recommend `Miniconda <http://conda.pydata.org/miniconda.html>`_. So please install it according to the official installation instructions. As soon as you have the ``conda`` command in your shell you can continue.

The following script will download and install all the needed packages.

.. code::

    conda create -q -n ascat_env -c conda-forge python=3.6 numpy pandas netCDF4 pip pyproj pybufr-ecmwf cython h5py pygrib
    source activate ascat_dev
    pip install pygeobase pygeogrids pynetcf lxml
    pip install ascat

This script should work on Windows, Linux or OSX but on Windows you will get errors for the installation commands of pybufr-ecmwf and pygrib.

Supported Products
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

We are happy if you want to contribute. Please raise an issue explaining what is missing or if you find a bug. We will also gladly accept pull requests against our master branch for new features or bug fixes.

Development setup
-----------------

For Development we also recommend a ``conda`` environment. You can create one including test dependencies and debugger by running ``conda env create -f environment.yml``. This will create a new ``ascat_dev`` environment which you can activate by using ``source activate ascat_dev``.

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the ascat repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get the test data repository.
- make a new feature branch from the ascat master branch
- Add your feature
- Please include tests for your contributions in one of the test directories. We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
