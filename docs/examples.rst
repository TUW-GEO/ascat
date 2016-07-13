.. _examples-page:

Examples
********

Reading and plotting ASCAT H25 data from netCDF format
======================================================

This Example script reads and plots ASCAT H25 SSM data with different masking
options and also converts the data to absolute values using the included
porosity data.

If the standard file names assumed by the script have changed this can be
specified during initialization of the AscatH25_SSM object. Please see the
documentation of :class:`ascat.timeseries.AscatH25_SSM`

.. include::
   read_ASCAT_H25.rst

Reading and plotting H-SAF images
=================================

`H-SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ provides three different image products:

* SM OBS 1 - H07 - Large scale surface soil moisture by radar scatterometer in BUFR format over Europe
* H16 - SSM ASCAT-B NRT O : Metop-B ASCAT soil moisture 12.5km sampling NRT
* H103 - SSM ASCAT-B NRT O : Metop-B ASCAT soil moisture 25km sampling NRT
* H101 - SSM ASCAT-A NRT O : Metop-A ASCAT soil moisture 12.5km sampling NRT
* H102 - SSM ASCAT-A NRT O : Metop-A ASCAT soil moisture 25km sampling NRT
* SM OBS 2 - H08 - Small scale surface soil moisture by radar scatterometer in BUFR format over Europe
* SM DAS 2 - H14 - Profile index in the roots region by scatterometer data assimilation in GRIB format, global

The products H07, H16, H101, H102, H103 come in BUFR format and can be read by
the same reader. So examples for the H07 product are equally valid for the other
products.

The following example will show how to read and plot each of them.

.. include::
   Read_H_SAF_images.rst

Reading and plotting ASCAT data from binary format
==================================================

This example program reads and plots ASCAT SSM and SWI data with different masking options.

.. include::
   plot_ascat_data.rst
