# Unreleased — v0.2 rewrite

**Major changes**
- **Burst-level downloads.** Sentinel-1 data is now fetched as just the bursts
  that intersect the AOI via `burst2safe`, instead of full ~250x170 km frames.
  Closes #23, #85, #88.
- **OPERA CSLC source.** A second SLC source `OperaCslcSearch` skips
  burst2safe + COMPASS entirely and pulls pre-made OPERA CSLC HDF5s
  (and the matching CSLC-STATIC layers) directly from ASF DAAC. Pick
  it via `sweets config --source opera-cslc`. Locked to OPERA's 5 m × 10 m
  posting; great for CONUS where OPERA has produced for the AOI.
- **NISAR GSLC source.** A third source class `NisarGslcSearch` wraps
  `opera_utils.nisar.run_download` to fetch pre-geocoded NISAR GSLC
  HDF5s via CMR (L-band, UTM, already geocoded). Pick it via
  `sweets config --source nisar-gslc --frequency A --polarizations HH`.
  Skips COMPASS, burst-db, orbits, and geometry stitching; dolphin
  reads the grid directly from the HDF5. Tropo correction is not yet
  supported on this path (NISAR GSLCs carry no stitched incidence angle).
- `Workflow.search` is a `Union[BurstSearch, OperaCslcSearch, NisarGslcSearch]`
  discriminated by a `kind` field; existing configs without a `kind`
  default to `safe` for backwards compat.
- **Tropospheric correction (opt-in).** New `--do-tropo` flag wires the
  OPERA L4 TROPO-ZENITH workflow from
  `opera_utils.tropo.create_tropo_corrections_for_stack` into a post-step
  that runs after dolphin produces unwrapped phase. Outputs land at
  `dolphin/tropo/tropo_correction_<dt>.tif` and `dolphin/tropo_corrected/<pair>.tropo_corrected.unw.tif`.
- **dolphin end-to-end.** Phase linking, interferogram network selection,
  stitching, unwrapping, timeseries inversion and velocity estimation are now
  delegated to a single `dolphin.workflows.displacement.run` call. The
  hand-rolled interferogram / stitch / unwrap orchestration was deleted.
- **`tyro` CLI.** `sweets config`, `sweets run` and `sweets server` are now
  defined with `tyro` instead of argparse, cutting ~200 lines and giving
  proper rich help. `sweets run <config_file>` is now positional.
- **pixi as the primary install.** `pyproject.toml` is reorganized so the
  `[tool.pixi.*]` sections are the canonical environment definition; an
  `environment.yml` synced from pixi is provided for non-pixi users.
- **`s1-reader`, `COMPASS`, `opera-utils` and `dolphin` fork pins.**
  sweets now installs all four from `scottstanie/<repo>@develop-scott`.
  The develop-scott branches carry numpy 2 fixes (polyfit scalar in
  s1-reader, `np.string_`/`np.unicode_` in COMPASS), the new tropo
  workflow / `search_tropo` CMR client in opera-utils, the
  GDT_Float16 GTIFF_KWARGS fix in opera-utils' `apply_tropo`, and a
  `_yaml_model._add_comments` fix in dolphin so Union-of-submodels
  schemas serialize cleanly. Closes #132.

**Removed**
- `sweets.interferogram` (replaced by dolphin's interferogram network).
- `sweets._missing_data`, `sweets.plotting`, `sweets._unzip` (no longer needed).
- `scripts/prep_mintpy.py` (broken with the new layout; mintpy export is
  TODO via dolphin's existing exporters).

**Fixed**
- **NISAR VRT wrappers instead of GeoTIFF rewrite.** GDAL's HDF5 driver
  can't parse NISAR's separate `xCoordinates` / `yCoordinates` grid
  arrays, so sweets used to rewrite each polarization as a ~19 MB
  CFloat32 GeoTIFF alongside every 40 MB subset HDF5. That's now a
  ~1 KB VRT that injects the real SRS + GeoTransform on top of the
  raw HDF5 subdataset — dolphin opens the VRT natively and the HDF5
  stays the single source of truth for pixel values. Conversion step
  dropped from O(n_pixels) to O(1).
- **NISAR wavelength auto-detect from filename.** dolphin's
  `model_post_init` now recognizes the NISAR D-102269 §3.4 granule
  prefix (`NISAR_L*` / `NISAR_S*`) and maps it to `NISAR_L_WAVELENGTH`
  / `NISAR_S_WAVELENGTH`, instead of opening the first CSLC with h5py
  to read `/science/LSAR/identification/radarBand`. Filename parsing
  is cheaper, works through any rename (sweets' VRT wrappers,
  GeoTIFF copies, subsetters), and doesn't lose information since
  the BETA PR products don't store a center-frequency dataset anyway.
  `NisarGslcSearch.wavelength()` uses the same prefix check. Without
  this, NISAR timeseries / velocity outputs landed in radians
  instead of meters (isce-framework/dolphin#704).
- **NISAR signature ranking + fallback.** `NisarGslcSearch.download()`
  now ranks (frequency, polarization) groups by `(stack size, pol match,
  freq match)` so a `polarizations` pin always beats a `frequency` pin
  on ties, and iterates groups in order — if the best group's products
  all yield empty stubs (AOI inside the bounding polygon but outside
  the actual grid extent, common on NISAR PR products), sweets falls
  through to the next signature instead of silently writing zero
  GeoTIFFs. Raises a clear diagnostic if every signature is empty.
- **Source-aware DEM bbox.** `Workflow._dem_bbox` used to pad the study
  bbox by 0.25 deg for every source, but COMPASS geocoding on the
  BurstSearch path needs DEM coverage for the full IW burst footprint
  (~20 x 85 km), not just the study area. BurstSearch now pads by 1 deg;
  NISAR / OPERA-CSLC keep the 0.25 deg buffer. Users can override with
  a new optional `dem_bbox` field. Water-mask downloads stay on the
  study-area bbox so the BurstSearch path doesn't waste ASF tile
  fetches on terrain outside the crop area.
- **Min-GSLC guard.** `Workflow.run` now raises a clear error before
  invoking dolphin when fewer than 2 GSLCs survive step 2, instead of
  letting dolphin fail deep inside `interferogram._make_ifg_pairs` with
  "No valid ifg list generation method specified".
- Driver-prefix heuristic for HDF5 vs NETCDF pushed down into
  `opera_utils.format_nc_filename` and `opera_utils.create_nodata_mask`
  (on `scottstanie/opera-utils@develop-scott`) and
  `dolphin.io.format_nc_filename` (on `scottstanie/dolphin@develop-scott`)
  so the NISAR raw-HDF5 subdataset path works end-to-end.
- NISAR wavelength auto-detect + corrected `NISAR_L_FREQUENCY` constant
  landed on `scottstanie/dolphin@develop-scott`
  (isce-framework/dolphin#704); parallel NISAR wavelength fix landed on
  `scottstanie/sarlet`.


# [0.2.0](https://github.com/opera-adt/dolphin/compare/v0.2.0...v0.3.0) - 2023-08-23

**Fixed**
- Geometry/`static layers` file creation from new COMPASS changes

**Dependencies**
- Upgraded `pydantic >= 2.1`
- Pinned minimum dolphin version due to mamba weirdness

# [0.1.0](https://github.com/isce-framework/sweets/commits/v0.1.0) - 2023-08-22


First version of processing workflow.

**Added**
- Created modules for DEM creation, ASF data download, geocoding SLCs, interferogram creation, and unwrapping.
- Created basic plotting utilities for interferograms and unwrapped interferograms.
- CLI commands `sweets config` and `sweets run` to configure a workflow and to run it.
