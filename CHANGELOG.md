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
