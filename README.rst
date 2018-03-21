=====
ascat
=====

.. image:: https://travis-ci.org/TUW-GEO/ascat.svg?branch=master
    :target: https://travis-ci.org/TUW-GEO/ascat

.. image:: https://coveralls.io/repos/github/TUW-GEO/ascat/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/ascat?branch=master

.. image:: https://badge.fury.io/py/ascat.svg
    :target: http://badge.fury.io/py/ascat

.. image:: https://zenodo.org/badge/12761/TUW-GEO/ascat.svg
   :target: https://zenodo.org/badge/latestdoi/12761/TUW-GEO/ascat

.. image:: https://readthedocs.org/projects/ascat/badge/?version=latest
   :target: http://ascat.readthedocs.org/

Read and convert data acquired by ASCAT on-board the series of Metop satellites. Written in Python.

Works great in combination with `pytesmo <https://github.com/TUW-GEO/pytesmo>`_.

Citation
========

If you use the software in a publication then please cite it using the Zenodo DOI:

.. image:: https://zenodo.org/badge/12761/TUW-GEO/ascat.svg
   :target: https://zenodo.org/badge/latestdoi/12761/TUW-GEO/ascat

Installation
============

The packages you have to install depend on the features you want to use. The H SAF soil moisture NRT products are disseminated in BUFR (H16, H103, H101, H102, H08) or GRIB (H14) format. So to read them you will have to install the appropriate packages which will be explained shortly. Unfortunately neither BUFR nor GRIB readers work on Windows so if you need these formats then Linux or OS X are your only options.

For installation we recommend `Miniconda <http://conda.pydata.org/miniconda.html>`_. So please install it according to the official installation instructions. As soon as you have the ``conda`` command in your shell you can continue.

The following script will download and install all the needed packages.

.. code::

    conda create -q -n ascat python=2 numpy pandas netCDF4 pytest pip pyproj
    source activate ascat
    conda install -c conda-forge pybufr-ecmwf # for reading BUFR files
    conda install -c conda-forge pygrib=2.0.1 # for reading GRIB files
    pip install ascat

This script should work on Windows, Linux or OSX but on Windows you will get errors for the installation commands of pybufr-ecmwf and pygrib.

Supported Products
==================

This gives a short overview over the supported products. Please see the documentation for detailed examples of how to work with a product.

Read ASCAT data from different sources into a common format supported by pytesmo.

Time Series Products
--------------------

* Metop ASCAT Surface Soil Moisture (SSM) Climate Data Record (CDR) in time series format

  Available in netCDF format from `H SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ (H25, H108-H112)


* CGLS SWI(Soil Water Index) Time Series (SWI_TS)

  Available from the `Copernicus Global Land Service (CGLS) <http://land.copernicus.eu/global/products/swi>`_


* ASCAT SWI(Soil Water Index) Time Series

  Available in binary format from `TU Wien <http://rs.geo.tuwien.ac.at/products/>`_

Image products
--------------

H SAF
~~~~~

`H SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ provides several different image products:

* H16 - SSM ASCAT-B NRT R : Metop-B ASCAT soil moisture 12.5km sampling NRT
* H103 - SSM ASCAT-B NRT O : Metop-B ASCAT soil moisture 25km sampling NRT
* H101 - SSM ASCAT-A NRT R : Metop-A ASCAT soil moisture 12.5km sampling NRT
* H102 - SSM ASCAT-A NRT O : Metop-A ASCAT soil moisture 25km sampling NRT
* SM OBS 2 - H08 - Small scale surface soil moisture by radar scatterometer in BUFR format over Europe
* SM DAS 2 - H14 - Profile index in the roots region by scatterometer data assimilation in GRIB format, global

The products H16, H103, H101, H102 come in the same BUFR format. Since the default filenames are slightly different the following readers should be used:

* H16 - :py:class:`ascat.h_saf.H16img`
* H101 - :py:class:`ascat.h_saf.H101img`
* H102 - :py:class:`ascat.h_saf.H102img`
* H103 - :py:class:`ascat.h_saf.H103img`

They are available after registration from the `H SAF Website <http://hsaf.meteoam.it/soil-moisture.php>`_

The H07 (SM OBS 1) is discontinued and replaced by H101, H102, H16 and H103.

* H07 - :py:class:`ascat.h_saf.H07img`

EUMETSAT
~~~~~~~~

EUMETSAT provides ASCAT Level 1 and Level 2 data among others through the `EUMETSAT Data Centre <http://www.eumetsat.int/website/home/Data/DataDelivery/EUMETSATDataCentre/index.html>`_. At the moment this package supports the following products:

* ASCAT Soil Moisture at 12.5 km Swath Grid - Metop in BUFR format
  :py:class:`ascat.eumetsat.AscatAL2Ssm125` and :py:class:`ascat.eumetsat.AscatBL2Ssm125`.
* ASCAT Soil Moisture at 12.5 km Swath Grid - Metop in BUFR format - 3 Minute PDU files.
  :py:class:`ascat.eumetsat.AscatAL2Ssm125PDU` and :py:class:`ascat.eumetsat.AscatBL2Ssm125PDU`.
  For reading half orbits use :py:class:`ascat.eumetsat.AscatAL2Ssm125PDUChunked`
  and :py:class:`ascat.eumetsat.AscatBL2Ssm125PDUChunked`
* ASCAT Soil Moisture at 12.5 km Swath Grid - Metop in netCDF format
  :py:class:`ascat.eumetsat.AscatAL2Ssm125Nc` and :py:class:`ascat.eumetsat.AscatBL2Ssm125Nc`.
* ASCAT Soil Moisture at 25.0 km Swath Grid - Metop in BUFR format
  :py:class:`ascat.eumetsat.AscatAL2Ssm250` and :py:class:`ascat.eumetsat.AscatBL2Ssm250`.
* ASCAT Soil Moisture at 25.0 km Swath Grid - Metop in BUFR format - 3 Minute PDU files.
  :py:class:`ascat.eumetsat.AscatAL2Ssm250PDU` and :py:class:`ascat.eumetsat.AscatBL2Ssm250PDU`.
  For reading half orbits use :py:class:`ascat.eumetsat.AscatAL2Ssm250PDUChunked`
  and :py:class:`ascat.eumetsat.AscatBL2Ssm250PDUChunked`
* ASCAT Soil Moisture at 25.0 km Swath Grid - Metop in netCDF format
  :py:class:`ascat.eumetsat.AscatAL2Ssm250Nc` and :py:class:`ascat.eumetsat.AscatBL2Ssm250Nc`.

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what is missing or if you find a bug. We will also gladly accept pull requests against our master branch for new features or bug fixes.

Development setup
-----------------

For Development we also recommend a ``conda`` environment. You can create one including test dependencies and debugger by running ``conda env create -f environment.yml``. This will create a new ``ascat-dev`` environment which you can activate by using ``source activate ascat-dev``.

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

This project has been set up using PyScaffold 2.5.6. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.
