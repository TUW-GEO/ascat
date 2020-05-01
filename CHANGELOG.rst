=========
Changelog
=========

Version 1.1.0
=============

- Python 2.7 no longer supported
- Update pyscaffold v3.2.3
- Fix netCDF4.num2date conversion problem

Version 1.0.2
=============

- Update readme

Version 1.0.1
=============

- Add unzip support for AscatL1Bufr and add metadata information
- Fix numpy FutureWarning

Version 1.0
===========

- Adding generic readers for ASCAT Level 1b and Level 2 data in EPS Native, BUFR, NetCDF and HDF5 formats
- Update readme structure
- Fix read the docs error
- Add cython to travis requirements
- Add script to setup miniconda development environment
- Read static layers into memory, instead of using NetCDF variables
- Add reader for H115
- Update copyright year

Version 0.10
============

- Add reader for H112, H113 and H114
- Update copyright year
- Update of ascat test data fixing netCDF4 valid_range issue

Version 0.9
===========

- Fix bug in H-SAF static layer readers. It was not possible to read data over
  multiple cells.

Version 0.8
===========

- Add reader for ASCAT VOD time series data.
- Add readers for all H-SAF time series products.
- Automatically detect CGLS SWI-TS time series product date and version.

Version 0.7
===========

- Fix bugs in BUFR reading with newer numpy versions.

Version 0.6
===========

- Fix bug when reading CGLS SWI QFLAG values.
- Add chunked half-orbit readers for the three minute PDU BUFR files.

Version 0.5
===========

- Include resample interface for Level 2 BUFR data.

Version 0.4
===========

- Restructure ASCAT swath readers and add support for NetCDF, BUFR and BUFR PDU
  files from EUMETSAT.
- Fix read_ts function of CGLS SWI_TS reader.

Version 0.3
===========

- Add reader for Copernicus Global Land SWI_TS products.

Version 0.2
===========

- Fix pygrib support for pygrib 2.x for H14 products.
- Internal changes. Readers now based on pynetCF and pygeobase.

Version 0.1
===========

- Initial version with readers migrated from the pytesmo package.
