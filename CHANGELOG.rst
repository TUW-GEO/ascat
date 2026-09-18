=========
Changelog
=========

Unreleased
==========

- Re-enable the Windows test run in CI,
- Fix a CI hang in which ``to_netcdf()`` blocked forever while acquiring
  xarray's netCDF write lock.
- Close the netCDF handles cached by xarray before removing temporary
  directories in the tests.
- Fix reading ASCAT Level 1b EPS Native and HDF5 files with ``generic=True``
  and ``to_xarray=True``, which raised ``AttributeError`` because
  ``mask_dtype_nans()`` was applied to the intermediate ``dict`` instead of
  the resulting ``xarray.Dataset``. SZR/SZO sentinel values in ``inc``,
  ``azi``, ``sig`` and ``kp`` are now converted to NaN as intended.
- Fix merging of multiple ASCAT Level 1b EPS Native files (e.g.
  ``AscatL1bEpsFileList.read_period()``), which raised ``UnboundLocalError``.
  ``PRODUCT_TYPE`` is upper case in the MPHR but was compared against a lower
  case literal, so SZF data always took an unreachable branch.
- Fix reading ASCAT Level 1b EPS Native SZF files with ``generic=False``,
  which raised ``AttributeError`` because a fill value was read from every
  field although only the generic conversion produces masked arrays.
- Support reading zip archives as delivered by EUMETSAT, which hold the data
  file next to XML metadata. ``get_file_format()`` determines the format from
  the data file inside the archive and ``tmp_unzip()`` extracts it, so ".zip"
  files can be passed to the EPS Native, NetCDF and BUFR readers directly.
- Fix ``get_toi_subset()`` and ``get_roi_subset()`` on ``xarray.Dataset``
  input, which raised ``AttributeError`` whenever the subset was not empty.
- Return empty datasets instead of ``None`` from ``get_toi_subset()`` and
  ``get_roi_subset()`` when nothing is left after filtering. The ``None`` was
  concatenated during merging, which silently turned the merged arrays into
  object arrays, e.g. in ``read_period()`` for files outside the requested
  period or for antenna beams outside the region of interest.
- Skip files that contribute no records in ``read_period()``, so that the
  returned metadata describes only the files the data actually came from.
- Fix the file selection of ``search_period()`` for products with more than
  one file per day. ``dt_delta`` was used both as the search step and to widen
  the period, so ``end_inclusive=True`` returned files up to a whole
  ``dt_delta`` after ``dt_end``, while ``end_inclusive=False`` missed files
  dated after the last search step. The search now always covers one step
  beyond the period, and ``end_inclusive`` only decides whether files dated
  exactly at ``dt_end`` are included.
- Add ``ascat.eumetsat.sca.level1``, a reader for EPS-SG SCA Level 1b data.
  ``ScaL1bFile`` returns the SZF full resolution backscatter as one dataset per
  antenna beam and the SZR re-sampled backscatter as quintuplets, following the
  structure of the ASCAT Level 1b readers. ``ScaL1bFileList`` searches and
  reads a collection of such files.
- Read with ``generic=True`` by default. Only the SZF reader did so far, which
  made the format depend on the product being read.
- Return the fields of the file, and only those, when reading with
  ``generic=False``. The coordinates were renamed to "lon" and "lat" even
  then, and the ASCAT SZF readers added a summary flag "f_usable", and for
  format version 12 a combined flag field "flagfield", which are derived from
  the flag fields of the file rather than stored in it. Both are now part of
  the generic format only.
- Use the same generic field names across the ASCAT products. The land
  fraction of the SZF products is now called "f_land", as in the SZR products,
  instead of "land_frac", and "sat_id" was added to the SZF products.
- Fix the conversion to the generic format of ASCAT Level 1b HDF5 files, which
  never renamed anything because it tested the field names against the items,
  rather than the keys, of its look-up table.
- Read the summary flag computed with your own categories next to the one the
  reader computes. Passing ``flag_kwargs`` to the read method of an ASCAT
  Level 1b file adds a field "f_usable_user", e.g.
  ``{"ignore_noise_ool": True}``. This replaces the ``ignore_noise_ool``
  argument, which changed "f_usable" itself.
- Fix writing a dataset read with ``to_xarray=True`` to NetCDF. The metadata of
  the file is stored as dataset attributes, and netCDF only holds numbers,
  strings and one-dimensional arrays, so a timestamp among them was enough to
  make ``to_netcdf()`` fail for every reader. Timestamps are now stored as
  strings and the values which still do not fit are left out of the
  attributes; the metadata returned by the readers keeps all of them.
- Accept the same spacecraft names everywhere. ``Spacecraft`` and the file
  list classes now take the short forms ("a", "B1"), the identifiers used in
  the file names ("M02", "SGB1") and the full names ("METOP-A",
  "METOP-SG B1").
- Add ``ascat.eumetsat.sca.flags``, which derives the summary flag of EPS-SG
  SCA Level 1b measurements from the individual processing flags.
  ``set_flags()`` assigns each flag a category and reports the highest one set,
  following the nominal, degraded and unusable convention of the ASCAT readers.
  Pass ``rfi_red=False`` to keep a noise outlier from rendering a measurement
  unusable, ``ignore`` to leave out other flags, or ``flag_bits`` to categorise
  them differently. As for the ASCAT readers, ``flag_kwargs`` adds it to the
  data as "f_usable_user", next to the summary the product provides.
- Read the summary flags of EPS-SG SCA Level 1b files. They are added to the
  metadata as "quality_...", and ``read_quality()`` reads them on their own,
  without the measurements. For SZR they include how many grid points received
  a complete set of measurements and how old the total electron content data
  used for the Faraday rotation correction was, neither of which can be
  derived from the measurements.
- Read the swath grid of EPS-SG SCA Level 1b SZF files with ``read_grid()``.
  It holds the nodes onto which the SZR products resample the measurements and
  has its own dimensions, so it is not returned together with the beams.
- Add tests for the EPS-SG SCA reader, which write the products they read.
- Find swath files which start before the requested period but still cover
  part of it. ``SwathGridFiles.swath_search()`` and ``.read()`` take a
  ``dt_buffer`` argument for this, as the file lists already did. Swath files
  are named after their start time, so a file overlapping the beginning of the
  period was missed.
- Close the swath file opened to check whether it intersects the area of
  interest, which was read twice and left open once.
- Fix the generic format of the ASCAT Level 1b HDF5 SZF products, which held a
  field "beam_number" that was never written. The beam number is left out of
  the generic format, as it is for the EPS Native products, where the dataset
  the beam belongs to already says which one it is.
- Add a test reading every product in all combinations of the generic and the
  xarray format.
- Read only the polarizations of interest from an EPS-SG SCA Level 1b SZF
  file. Only the mid beams are measured in more than one polarization, so
  ``skip_polarization=["hh", "vh", "hv"]`` leaves six of the twelve beams and a
  quarter fewer measurements.
- Add ``plot_szf``, which puts the measurements of an ASCAT or EPS-SG SCA
  Level 1b SZF file on a map and tells what the file says about a measurement
  which is clicked. Both instruments are read by the same routine, as their
  readers return the same generic fields; which one a file belongs to follows
  from its name. Plotting needs matplotlib and eomaps, which are in the "plot"
  dependency group rather than installed with the package.

Version 2.8.1
=============

- Fix uv dependencies

Version 2.8.0
=============

- Add cf_conversions and rework ragged array representations
- Cleanup download functions
- Make ragged-array instance lookup memory-efficient

Version 2.7.0
=============

- Speed up reading of ASCAT Level 1b and Level 2 BUFR files by reading the
  required fields directly via ``eccodes`` array access instead of
  ``pdbufr.read_bufr(..., flat=True)`` (orders of magnitude faster for large files; output unchanged)
- Replace the ``pdbufr`` dependency with ``eccodes``
- Switch packaging and dependency management to ``uv`` (``uv_build`` backend)
- ``pygrib`` is now an optional ``grib`` extra

Version 2.6.5
=============

- Add test class reading ASCAT data in zarr format

Version 2.6.4
=============

- Update `product_info.AscatH129Swath.sf_pattern` and `.sf_read_fmt` to match operational
  product; update tests to match

Version 2.6.3
=============

- Update `product_info.AscatH129Swath.fn_pattern` to match operational product; update
  tests to match

Version 2.6.2
=============

- Add switch to make parallel processing optional in `SwathGridFiles.stack_to_cell_files`
- Add ragged array class for testing purposes

Version 2.6.1
=============

- Add new CLI argument "--sat" in the aggregation interface to allow users to filter METOP satellite data

Version 2.6.0
=============

- Refactors the temporal aggregator to process and write each timestep
  independently rather than loading and grouping the entire dataset at once
- Add tqdm to dependencies

Version 2.5.5
=============

- Introduces a --product_id CLI argument in both swath_regrid and swath_resample interfaces
- Updates inverse_distance_resampling signature to accept product_id and attempts filename-based inference
- Hooks the resampling step into the temporal aggregation flow with new --resample flags

Version 2.5.4
=============

- Bugfix in SSM aggregation

Version 2.5.3
=============

- Fix swath stacking bug (missing scaling)

Version 2.5.2
=============

- Add ASCAT SSM ICDR (H130, H139) to product info
- Code formatting

Version 2.5.1
=============

- Fix pypi issue

Version 2.5.0
=============

- Rewrite xarray readers/writers
- Add CLI for resampling ASCAT swath data

Version 2.4.4
=============

- Fix numpy v2.0 related issues
- Fix CI settings
- Add EODAG tutorial
- Add Filename class
- Level 1 and 2 reader use new Filename class

Version 2.4.3
=============

- Fix datatype cast issue

Version 2.4.2
=============

- Fix typo in flagfield name

Version 2.4.1
=============

- Add keyword to ignore ASCAT Level 1b flag noise out of limits

Version 2.4.0
=============

- Update file handling read_period
- Refactor xarray readers/writers into SwathFileCollection, CellFileCollection,
  and CellFileCollectionStack
- Add tutorial for xarray readers/writers
- Add CLI for aggregating ASCAT swath data
- Add CLI for regridding ASCAT swath data to a regular lat/lon grid

Version 2.3.1
=============

- Add str to path conversion for H SAF FTP download

Version 2.3.0
=============

- Update interface of file handling module

Version 2.2.0
=============

- Update to pyscaffold v4.5
- Update download module, supporting parallel download for EUMETSAT Data Store
- Update interface of file handling module

Version 2.1.1
=============

- Add draft for indexed ragged array netcdf file reader using xarray
- Add ASCAT netcdf swath file reader

Version 2.1.0
=============

- Add support reading ascat fmv 13 for szf, szr, szo

Version 2.0.6
=============

- Add timeout when reading corrupt EPS files
- Change beam names when reading ASCAT SZF data (adding -vv)

Version 2.0.5
=============

- Fix file handing bug (duplicated file names)

Version 2.0.4
=============

- Fix download
- Update CI

Version 2.0.3
=============

- Return metadata when reading ASCAT data

Version 2.0.2
=============

- Adapt EUMETSAT download API changes
- Update read native bufr

Version 2.0.1
=============

- Update ASCAT Level 1b SZF reader
- Fix test error of H14

Version 2.0.0
=============

- New interface reading ASCAT Level 1b and Level 2 data
- Removing old interfaces to TU Wien data
- Restructure package and harmonize interface class names
- Update documentation

Version 1.2.0
=============

- Add download interface for H SAF FTP and EUMETSAT Data Store
- Move CI to Github actions (Ubuntu and Windows CI)

Version 1.1.2
=============

- Fix dependencies in setup.cfg
- Pin dependency of h5py=2.10

Version 1.1.1
=============

- Update template name for consistency reason

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
