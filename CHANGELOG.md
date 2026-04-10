# Unreleased — v0.2 rewrite

**Major changes**
- **Burst-level downloads.** Sentinel-1 data is now fetched as just the bursts
  that intersect the AOI via `burst2safe`, instead of full ~250x170 km frames.
  Closes #23, #85, #88.
- **dolphin end-to-end.** Phase linking, interferogram network selection,
  stitching, unwrapping, timeseries inversion and velocity estimation are now
  delegated to a single `dolphin.workflows.displacement.run` call. The
  hand-rolled interferogram / stitch / unwrap orchestration was deleted.
- **`tyro` CLI.** `sweets config`, `sweets run` and `sweets server` are now
  defined with `tyro` instead of argparse, cutting ~200 lines and giving
  proper rich help.
- **pixi as the primary install.** `pyproject.toml` is reorganized so the
  `[tool.pixi.*]` sections are the canonical environment definition; an
  `environment.yml` synced from pixi is provided for non-pixi users.
- **`s1-reader` and `COMPASS` fork pins.** sweets now installs both
  s1-reader and COMPASS from `scottstanie/<repo>@develop-scott`, which
  carry numpy 2 fixes (polyfit scalar in s1-reader,
  `np.string_`/`np.unicode_` removed in COMPASS). Closes #132.

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
