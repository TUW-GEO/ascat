=====
ascat
=====

.. image:: https://travis-ci.org/TUW-GEO/ascat.svg?branch=master
    :target: https://travis-ci.org/TUW-GEO/ascat

.. image:: https://coveralls.io/repos/TUW-GEO/ascat/badge.png?branch=master
  :target: https://coveralls.io/r/TUW-GEO/ascat?branch=master

.. image:: https://badge.fury.io/py/ascat.svg
    :target: http://badge.fury.io/py/ascat

Readers and converters for data acquired by the ASCAT sensor on-board the MetOP
satellites. Written in Python.

Supported Products
==================

This give a short overview over the supported products. Please see the
documentation for detailed examples of how to work with a product.

Read ASCAT data from different sources into a common format supported by pytesmo

Time Series Products
--------------------

* ASCAT SSM(Surface Soil Moisture) Time Series

  Available in netCDF format from `H-SAF
  <http://hsaf.meteoam.it/soil-moisture.php>`_ (H25 product)


* ASCAT SWI(Soil Water Index) Time Series

  Available in binary format from `TU Wien <http://rs.geo.tuwien.ac.at/products/>`_

Image products
--------------

`H-SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ provides several different
image products:

* SM OBS 1 - H07 - Large scale surface soil moisture by radar scatterometer in
  BUFR format over Europe
* H16 - SSM ASCAT-B NRT O : Metop-B ASCAT soil moisture 12.5km sampling NRT
* H103 - SSM ASCAT-B NRT O : Metop-B ASCAT soil moisture 25km sampling NRT
* H101 - SSM ASCAT-A NRT O : Metop-A ASCAT soil moisture 12.5km sampling NRT
* H102 - SSM ASCAT-A NRT O : Metop-A ASCAT soil moisture 25km sampling NRT
* SM OBS 2 - H08 - Small scale surface soil moisture by radar scatterometer in
  BUFR format over Europe
* SM DAS 2 - H14 - Profile index in the roots region by scatterometer data
  assimilation in GRIB format, global

The products H07, H16, H101, H102, H103 come in BUFR format and can be read by
the same reader. So examples for the H07 product are equally valid for the other
products.

They are available after registration from the `H-SAF Website
<http://hsaf.meteoam.it/soil-moisture.php>`_

Documentation
=============

|Documentation Status|

.. |Documentation Status| image:: https://readthedocs.org/projects/ascat/badge/?version=latest
   :target: http://ascat.readthedocs.org/

Installation
============

Detailed instructions will follow. For now please see the ``install`` section in
``.travis.yml`` in this repository.

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what is missing
or if you find a bug. We will also gladly accept pull requests against our master branch
for new features or bug fixes.

If you want to contribute please follow these steps:

- Fork the ascat repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get
  the test data repository.
- make a new feature branch from the ascat master branch
- add your feature
- please include tests for your contributions in one of the test directories
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch

Note
====

This project has been set up using PyScaffold 2.5.6. For details and usage
information on PyScaffold see http://pyscaffold.readthedocs.org/.

