.. _examples-page:

Examples
********

Reading and plotting Metop ASCAT Surface Soil Moisture CDR (NetCDF)
===================================================================

`H SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ provides the following Metop
ASCAT Surface Soil Moisture (SSM) Climate Data Record (CDR) products:

* H25 - Metop ASCAT SSM CDR2014 : Metop ASCAT Surface Soil Moisture CDR2014 time series 12.5 km sampling
* H109 - Metop ASCAT SSM CDR2015 : Metop ASCAT Surface Soil Moisture CDR2015 time series 12.5 km sampling
* H111 - Metop ASCAT SSM CDR2016 : Metop ASCAT Surface Soil Moisture CDR2016 time series 12.5 km sampling

The following CDR extensions are also provided by H SAF:

* H108 - Metop ASCAT SSM CDR2014-EXT : Metop ASCAT Surface Soil Moisture CDR2014-EXT time series 12.5 km sampling
* H110 - Metop ASCAT SSM CDR2015-EXT : Metop ASCAT Surface Soil Moisture CDR2015-EXT time series 12.5 km sampling
* H112 - Metop ASCAT SSM CDR2016-EXT : Metop ASCAT Surface Soil Moisture CDR2016-EXT time series 12.5 km sampling

.. include::
   read_hsaf_cdr.rst

Reading and plotting CGLS SWI_TS data (NetCDF)
==============================================

This example script reads the SWI_TS product of the Copernicus Global Land
Service.

If the standard file names assumed by the script have changed this can be
specified during initialization of the SWI_TS object. Please see the
documentation of :class:`ascat.cgls.SWI_TS`.

.. include::
   read_cgls_swi_ts.rst

Reading and plotting H SAF NRT Surface Soil Moisture products (BUFR)
====================================================================

`H SAF <http://hsaf.meteoam.it/soil-moisture.php>`_ provides the following NRT
surface soil moisture products:

* (H07 - SM OBS 1 : Large scale surface soil moisture by radar scatterometer in BUFR format over Europe) - discontinued
* H08 - SSM ASCAT NRT DIS : Disaggregated Metop ASCAT NRT Surface Soil Moisture at 1 km
* H14 - SM DAS 2 : Profile index in the roots region by scatterometer data assimilation in GRIB format
* H101 - SSM ASCAT-A NRT O12.5 : Metop-A ASCAT NRT Surface Soil Moisture 12.5km sampling
* H102 - SSM ASCAT-A NRT O25 : Metop-A ASCAT NRT Surface Soil Moisture 25 km sampling
* H16 - SSM ASCAT-B NRT O12.5 : Metop-B ASCAT NRT Surface Soil Moisture 12.5 km sampling
* H103 - SSM ASCAT-B NRT O25 : Metop-B ASCAT NRT Surface Soil Moisture 25 km sampling

The products H101, H102, H16, H103 come in BUFR format and can be read by the
same reader. So examples for the H16 product are equally valid for the other
products.

The product H07 is discontinued and replaced by Metop-A (H101, H102) and Metop-B
(H103, H16), both available in two different resolutions.

The following example will show how to read and plot each of them.

.. include::
   read_hsaf_nrt.rst

Reading and plotting TU Wien Metop ASCAT Surface Soil Moisture (Binary)
=======================================================================

This example program reads and plots Metop ASCAT SSM and SWI data with different
masking options. The readers are only provided for the sake of completeness,
because the data sets are outdated and superseded by the H SAF Surface Soil
Moisture Climate Data Records (e.g. H109, H111). The SWI data sets are replaced
by the CGLS SWI product.

.. include::
   read_tuw_ascat.rst


Reading and plotting TU Wien Metop ASCAT Vegetation Optical Depth (VOD)
=======================================================================

This example program reads and plots Metop ASCAT VOD data.

.. include::
   read_tuw_vod.rst

Reading and plotting lvl1b and lvl2 ASCAT data in generic manner
================================================================

This example script uses the generic lvl1b and lvl2 reader to get an generic
Image.


.. include::
   read_generic.rst