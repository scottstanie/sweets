# sweets v0.2 revival notes

A breadcrumb file for the v0.2-rewrite branch — what changed, what's still
loose, and where to look next.

## What changed

| layer | before | after |
|---|---|---|
| **download (default)** | `ASFQuery` → ASF API + wget/aria2c, full S1 frames | `BurstSearch` → `burst2safe.burst2stack`, just the bursts that intersect the AOI |
| **download (alt source: OPERA)** | n/a | `OperaCslcSearch` → `opera_utils.download.{search,download}_cslcs` for pre-made OPERA CSLCs, plus `download_cslc_static_layers`. Pick via `--source opera-cslc`; locked to OPERA's 5×10 m posting. |
| **download (alt source: NISAR)** | n/a | `NisarGslcSearch` → `opera_utils.nisar.run_download` for pre-geocoded NISAR GSLC HDF5s (L-band, UTM). Pick via `--source nisar-gslc --frequency A --polarizations HH`. Skips COMPASS + geometry stitching entirely; dolphin reads the grid from the HDF5 directly. Tropo not yet supported on this path. |
| **s1-reader** | upstream `isce-framework/s1-reader` (broken on numpy 2, see #132) | `scottstanie/s1-reader@develop-scott` |
| **COMPASS** | upstream `opera-adt/COMPASS` (np.string_/np.unicode_ → numpy 2 crash) | `scottstanie/COMPASS@develop-scott` |
| **opera-utils** | conda-forge | `scottstanie/opera-utils@develop-scott` (carries the high-level tropo workflow + the CSLC download API + the NISAR download API + the Float16 GTIFF fix) |
| **dolphin** | upstream | `scottstanie/dolphin@develop-scott` (carries the `_yaml_model._add_comments` fix for Union-of-submodels schemas) |
| **geocoding** | COMPASS via `_geocode_slcs.py` | unchanged for `--source safe`; **skipped entirely** for `--source opera-cslc` |
| **interferograms / stitch / unwrap / timeseries** | hand-rolled in `sweets.interferogram` + `sweets.core` + `dolphin.unwrap.run` | one call to `dolphin.workflows.displacement.run` via the new `sweets._dolphin` adapter |
| **tropo correction** | n/a | new opt-in post-step (`--do-tropo`) wrapping `opera_utils.tropo.create_tropo_corrections_for_stack` + `apply_tropo_to_unwrapped` |
| **CLI** | argparse, ~280 lines, manual group dict shuffling | `tyro`, 3 dataclass-style subcommands; `--source`, `--do-tropo`, positional `sweets run <config>` |
| **packaging** | `pixi.toml` "thrown in" alongside the pip install | `pyproject.toml` is pixi-first; `[tool.pixi.*]` is the canonical env definition |
| **web UI** | partial scaffolding (uncommitted) | committed but **tabled** under `src/sweets/web/`; mypy/pre-commit excluded |

## Open issues this branch addresses

| # | title | notes |
|---|---|---|
| #23 | "Can SLC data be downloaded in burst as the basic unit" | yes — that's the headline of this branch |
| #27 | "Print out ASF Search query when configuring or starting download" | `BurstSearch.summary()` is logged before download |
| #29 | "Make `--start`, `--stop` and `--do-step` command line arguments" | tyro `sweets run --starting-step N` |
| #79 | "sweets looks for removed module in dolphin" | removed the legacy `dolphin.interferogram.Network` import path |
| #80 | "`--data-dir` doesn't work in `sweets config`" | `--out-dir` on the new tyro CLI is honored end-to-end |
| #85 | "Compatibility with `Burst2Safe`" | sweets now *uses* burst2safe |
| #88 | "Refactor download to allow OPERA gslcs" | done — `OperaCslcSearch` source class, `--source opera-cslc` CLI flag |
| #107 | "Sweets only checks for presence of files not of files needed" | `existing_safes()` is now a single, easy-to-extend hook; an integrity check belongs there |
| #132 | "Error in GSLC generation step" (numpy 2 polyfit) | fixed by switching to `scottstanie/s1-reader@develop-scott` |

## Open PRs against `main` that should be revisited

- **#128** `fix: updated OPERA_DATASET_ROOT to OPERA_DATASET_NAME in prep_mintpy.py`
  → `prep_mintpy.py` was deleted on this branch (the old per-burst stitched
  output layout it depends on no longer exists). Mintpy export should be
  re-added on top of dolphin's outputs (probably via `dolphin.io.export_mintpy`
  or similar). Ping the contributor.
- **#129** `CDSE endpoint for download.` → burst2safe handles the source
  selection internally; if the contributor specifically wants Copernicus
  Dataspace as the backend, that's a burst2safe feature request, not a sweets
  one.
- **#125** `[pre-commit.ci] pre-commit autoupdate` → just merge once this branch
  lands.

## What's still loose (ordered by usefulness)

1. **Upstream the fork fixes.** Every correctness fix in this branch
   that touched a dependency landed on `scottstanie/<repo>@develop-scott`:
   `dolphin` (NISAR wavelength + filename parser + HDF5-vs-NETCDF driver
   split + yaml_model Union handling), `opera-utils` (NETCDF/HDF5 driver
   split in `format_nc_filename` + `create_nodata_mask`, plus the Float16
   GTIFF_KWARGS fix), `COMPASS` (numpy 2 `np.string_`/`np.unicode_`),
   `s1-reader` (polyfit numpy 2 regression), `sarlet` (NISAR
   L-band constant). Separate PRs back to each upstream repo.
2. **Add a `sweets export-mintpy` subcommand** that wraps dolphin's mintpy
   exporter (closes #128 + #42). Should be a thin function in
   `sweets._mintpy.py`.
3. **Tropo correction for NISAR.** NISAR GSLCs don't have a separate
   CSLC-STATIC file, so there's no stitched `local_incidence_angle.tif`
   for `apply_tropo` to consume. `Workflow._run_tropo` currently warns
   and skips when the source is NISAR. Need either: (a) a sweets-side
   helper that extracts / computes incidence from the NISAR GSLC's
   orbit + DEM, (b) a separate NISAR-specific GeoTIFF we ship, or
   (c) user-supplied incidence raster path on the CLI.
4. **Wire pixi to run the smoke tests in CI.** `pixi run smoke-opera`,
   `pixi run smoke-safe`, `pixi run smoke-nisar` against pre-staged
   tiny stacks would catch the kind of "import works but pipeline
   doesn't" regression that bit several open issues.
5. **L-band frequencyA vs frequencyB carrier precision.** Current
   products don't populate `/science/LSAR/GSLC/grids/frequency{A,B}/centerFrequency`
   (per NISAR D-102269 §4 they should). `NisarGslcSearch.wavelength()`
   already has a runtime reader that will pick this up the moment
   spec-compliant products appear; for now it falls back to the
   filename-based L/S constant, which is within ~1% of the split-mode
   centers. Revisit when freqB-only test data shows up or mm-accurate
   displacement becomes a requirement.
6. **Web UI** — left exactly as Scott had it under `src/sweets/web/`. Excluded
   from mypy and from this revival's scope.

## Things I (Claude) deliberately did NOT do

- **Open PRs against upstream `isce-framework/dolphin`,
  `opera-adt/COMPASS`, or `opera-adt/opera-utils`.** All fixes landed on
  personal forks `scottstanie/<repo>@develop-scott`; opening upstream
  PRs is left to whoever is talking to the dolphin release channel /
  OPERA.
- **Touch the COMPASS / `_geocode_slcs.py` integration.** Geocoding still uses
  COMPASS; the hand-rolled config-file shuffling in `_geocode_slcs.py` is the
  same as on main. If we want to drop COMPASS in favor of an `isce3.geocode_slc`
  call directly, that's a separate (large) job.
- **Build a NISAR incidence-angle raster for tropo.** `Workflow._run_tropo`
  warns and skips when the source is NISAR. See "What's still loose".

## Smoke test results, round 4 (2026-04-11, three-source cross-validation)

End-to-end runs of all three source variants against the same LA AOI
(`-118.3957 33.7284 -118.3459 33.772`, Long Beach / San Pedro) and
the same Dec 2025 window, codified as three self-contained example
notebooks under `docs/`. Everything passes and the output `velocity.tif`
comes out in `meters / year` on every path.

| notebook | source | track/frame | wall time | note |
|---|---|---|---|---|
| `example_s1_burst.ipynb` | `safe` (S1 bursts + COMPASS) | T071 desc, IW2 | ~5 min | 5 cycles |
| `example_opera_cslc.ipynb` | `opera-cslc` | T071 desc | ~5 min | 5 cycles |
| `example_nisar.ipynb` | `nisar-gslc` | T034 asc frame 18 | ~1 min | 2 cycles |

**Bugs caught and fixed during this round:**

1. **dolphin `format_nc_filename` swapped HDF5 vs NETCDF driver
   wrong.** My earlier fix had it pick `HDF5:` for every `.h5` file,
   which broke CF-compliant HDF5s like OPERA CSLCs and COMPASS
   static_layers: GDAL's bare HDF5 driver opens the data but returns
   an identity geotransform. Switch to: NISAR raw HDF5s (granule
   prefix `NISAR_`) get HDF5:, everything else gets NETCDF:.
   Matching fix in `opera_utils._utils.format_nc_filename` and
   `opera_utils._cslc.create_nodata_mask`.
2. **dolphin `stitching.get_downsampled_vrts` wrapped GDAL subdataset
   strings in `Path()`.** `Path("HDF5:\"f.h5\":\"//data/los_east\"")`
   silently normalizes the inner `//` to `/`, turning a valid GDAL
   subdataset reference into an invalid one. Broke COMPASS
   static_layers stitching on the BurstSearch path. Fix: keep `fn`
   as a string via `fspath(fn)` and don't go through `pathlib.Path`.
3. **opera-utils' NISAR subset stripped `centerFrequency` and other
   scalar metadata** from the frequency group. Add a catch-all loop
   that copies every `ndim <= 1` Dataset in the source group. First
   attempt copied 2D datasets too and hung on the ~5 GB per-product
   `mask` raster — restricted to 1D + scalar after that.
4. **NISAR wavelength: no static table matches real products.** The
   initial fix parsed the filename MODE code and looked up Figure 3-1
   values (e.g. 4005 -> freqA 1229 MHz). But the actual BETA PR
   products report freqA = 1239 MHz for mode 4005, a 10 MHz / ~0.8%
   disagreement. Keep the MODE table as a fallback (still strictly
   better than the generic 1257.5 MHz constant) and document the
   drift, but prefer reading `centerFrequency` from the HDF5 directly
   whenever it's present. opera-utils now preserves it during subset
   extraction (see #3 above).

## Smoke test results, round 3 (2026-04-10, NISAR GSLC path)

End-to-end run with `--source nisar-gslc` against several AOIs around
the Los Angeles / Salinas area, picking up the NISAR PR GSLC BETA V1
products on CMR. After fixing the issues below, the pipeline runs
cleanly through search → per-product subset → VRT wrap → dolphin
phase linking / unwrap / timeseries / velocity in ~23 seconds of
pipeline wall time for a two-cycle stack on the Salinas AOI, with
outputs under `dolphin/timeseries/*.tif` and `dolphin/unwrapped/`.

**Bugs caught and fixed during this run** (all on the relevant fork
branches + the sweets v0.2-rewrite branch):

1. **`_choose_signature` scoring overweighted `frequency` pins.** A
   config pinning `frequency: A, polarizations: [VV]` picked a single-
   cycle frequencyA/[HH,HV] group over a single-cycle
   frequencyB/[VV,VH] group — choosing the one with zero pol overlap.
   Rewrote as `_rank_signatures` with sort key
   `(n_cycles, pol_match, freq_match)` so pol pin dominates.
2. **Chosen signature could yield zero usable GeoTIFFs silently.**
   NISAR PR products can advertise a frequency whose actual grid
   extent is narrower than the bounding polygon, so `process_file`'s
   subset writes a 25 KB metadata-only stub. sweets now iterates
   ranked signatures in order and falls through to the next one when
   the current group yields zero outputs, raising a clear error only
   if every signature is empty.
3. **Single-GSLC stacks crashed deep inside dolphin.** When only one
   valid GSLC survived, `interferogram._make_ifg_pairs` bailed with
   "No valid ifg list generation method specified". Added a guard in
   `Workflow.run` that raises a clear sweets-side error naming the
   likely culprits (narrow date range, over-specific pol / frequency
   pins, wrong track/frame).
4. **DEM bbox was conflated with crop bbox.** `Workflow._dem_bbox`
   used a 0.25 deg buffer for every source, but COMPASS geocoding on
   the BurstSearch path needs DEM coverage for the full IW burst
   footprint (~20 x 85 km). David Bekaert flagged this mid-debug.
   Split into source-aware defaults (1 deg for BurstSearch, 0.25 deg
   for the rest), a separate `_water_mask_bbox` that stays small on
   BurstSearch, and an optional `Workflow.dem_bbox` override field.
5. **Empty-stub HDF5 conversion crashed GeoTIFF write.** opera-utils'
   per-product subset can produce an HDF5 with no `/grids/frequencyX`
   group when the bbox is outside the actual grid extent. The
   downstream conversion path now logs a warning + skips instead of
   aborting the whole stack.
6. **HDF5 vs NETCDF driver prefix assumed NETCDF.** dolphin and
   opera-utils both built GDAL connection strings as
   `NETCDF:"file.h5":"//ds"`, which fails for NISAR's raw HDF5 (no CF
   metadata). Split by extension: `.h5` → `HDF5:`, `.nc` → `NETCDF:`.
   Landed on `scottstanie/dolphin@develop-scott`
   and `scottstanie/opera-utils@develop-scott`.
7. **NISAR HDF5 -> 19 MB GeoTIFF rewrite was overkill.** GDAL's HDF5
   driver opens the subdataset but reports an identity geotransform,
   so the first pass wrote one CFloat32 GeoTIFF per polarization to
   inject the georeferencing. Replaced with a ~1 KB VRT that wraps the
   HDF5 subdataset, reading the actual grid from the HDF5 and layering
   the geotransform on top in XML. Conversion step dropped from
   O(n_pixels) to O(1).
8. **Timeseries / velocity outputs were in radians, not meters.**
   dolphin's `model_post_init` only auto-detected OPERA-S1 and
   Capella; NISAR fell through with no warning and wrote units as
   `radians / year`. First fix attempt landed an h5py-based NISAR
   branch that reads `/science/LSAR/identification/radarBand` — but
   that opens the HDF5 on every run and doesn't work at all through
   sweets' VRT wrappers. Replaced with a two-line filename prefix
   check (`NISAR_L*` vs `NISAR_S*` per NISAR D-102269 §3.4), which
   is cheaper and works for raw HDF5, VRTs, GeoTIFF copies, or any
   rename that keeps the granule prefix. dolphin fork carries the
   filename fix + corrected `NISAR_L_FREQUENCY` constant (old value
   gave a 1.4 mm wavelength error). Parallel sarlet fix for
   `SENSOR_WAVELENGTHS["NISAR"]`. sweets' explicit wavelength
   pass-through was then deleted since dolphin handles it
   end-to-end; verified on the test-sweets-nisar Salinas run, which
   produced `velocity.tif` with `Unit Type: meters / year` and a
   std dev of ~26 mm — physically plausible.

   TODO (not a blocker): L-band frequencyA and frequencyB centers
   differ by ~1% (~2 mm wavelength), but current products don't
   store the actual carrier and sweets always downloads a
   single-frequency stack. Revisit when freqB-only test data is
   available or mm-accurate displacement becomes a requirement.

## Smoke test results, round 2 (2026-04-10, OPERA CSLC + tropo path)

End-to-end run with `--source opera-cslc --do-tropo` against the same
pecos AOI in
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos_opera/`.

| stage | time |
|---|---|
| OPERA CSLC + CSLC-STATIC download from ASF | ~3 min (18 CSLCs + 9 static layers) |
| stitch geometry from CSLC-STATIC | ~12 s |
| dolphin (phase linking + ifg + unwrap + timeseries + velocity) | ~3.8 min |
| tropo: search + crop + apply | ~70 s |
| tropo: subtract differential from unwrapped phase | <1 s |
| **total `sweets run --source opera-cslc --do-tropo`** | **~7.9 min** |

Notes:
- The OPERA path skips burst-db, orbits, and COMPASS entirely. For users
  in CONUS where OPERA has produced for the AOI, this is the faster
  default.
- The tropo step found 18 OPERA L4 TROPO-ZENITH products for the date
  range (one per CSLC sensing time across the 9 bursts × 2 dates).
  Pecos in summer is essentially noise — the corrected unwrapped phase
  std went from 4.61 → 4.68 rad, a non-meaningful change. The pipeline
  validation is structural; numerical validation needs a wetter or more
  topography-correlated AOI.
- Outputs:
  - `data/OPERA_L2_CSLC-S1_*.h5` (downloaded, 9 per date)
  - `data/static_layers/OPERA_L2_CSLC-S1-STATIC_*.h5` (per-burst, single date)
  - `dolphin/tropo/tropo_correction_<dt>.tif` (one per CSLC sensing time)
  - `dolphin/tropo/reference_tropo_correction_<dt>.tif`
  - `dolphin/tropo_corrected/<date1>_<date2>.tropo_corrected.unw.tif`

**Bugs caught and fixed during this run** (all on the relevant fork branches):

1. **`opera_utils._apply.GTIFF_KWARGS` had `nbits=16` with `dtype=float32`**
   — GDAL writes that as `Float16`, and rasterio's `_band_dtype` map
   has no entry for it (`KeyError: 15`) when reading the reference
   correction back. Fix: drop the `nbits` line. Landed on
   [`scottstanie/opera-utils@develop-scott`](https://github.com/scottstanie/opera-utils/commit/d4c4fe9).
2. **aiohttp's `aiodns` resolver DNS-times out** on some networks when
   `crop_tropo` opens the OPERA L4 TROPO URLs. Same fix burst2safe
   needed earlier: force `aiohttp.resolver.ThreadedResolver`. Done
   inside `sweets._tropo` as a side-effect import; no opera-utils
   change needed.
3. **dolphin's commented-yaml emitter** (`_yaml_model._add_comments`)
   raised `KeyError` walking an `anyOf` schema entry that's a `$ref`
   to a sub-model — standard Pydantic 2 behaviour for a Union of
   submodels. Fixed upstream in
   [`scottstanie/dolphin@develop-scott`](https://github.com/scottstanie/dolphin/commit/46762c6);
   sweets now pins that branch and the `_dolphin_yaml_compat` shim
   has been deleted.
4. **Stale tropo intermediates from a failed run.** The Float16
   reference correction tif from the buggy first run survived past the
   GTIFF_KWARGS fix and kept crashing reads. Wipe `dolphin/tropo/` if
   re-running after that kind of fix; the Workflow doesn't yet do this
   automatically (it would conflict with the desire to skip-if-exists).

## Smoke test results, round 1 (2026-04-10)

Full end-to-end run against the pecos AOI (`-102.96 31.22 -101.91 31.56`,
track 78, dates `2021-06-05` → `2021-06-22`, swath `IW2`) in
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos_revival/`.

**It works.** The workflow runs cleanly through download → DEM → water
mask → orbits → COMPASS geocoding (10 GSLCs + 5 static layers) → geometry
stitching → dolphin (phase linking → ifg → unwrap → timeseries → velocity).

Wall time:

| stage | time |
|---|---|
| burst2safe download (2 SAFEs, ~770 MB each, IW2-only) | ~4 min |
| COMPASS geocoding (10 CSLCs + 5 static layers) | ~4 min |
| geometry stitching | ~12 s |
| dolphin (phase linking + unwrap + timeseries + velocity) | ~2.7 min |
| **total `sweets run`** | **~5.8 min** (after the SAFEs are downloaded) |

Final outputs (under `dolphin/`):

- `interferograms/20210606_20210618.int.tif` + `.cor.tif` + `.mask.tif`
- `unwrapped/20210606_20210618.unw.tif` + `.unw.conncomp.tif`
- `timeseries/20210606_20210618.tif`, `velocity.tif`,
  `reference_point.txt`, `warped_watermask.tif`
- per-burst `linked_phase/`, `PS/`, masks etc.

**Bugs caught and fixed during the run** (all already on this branch):

1. **`sardem` water-mask path was broken on macOS** in two ways at once:
   - newer sardem hard-asserts `NASA_WATER` only supports `ENVI` output
   - sardem's `_unzip_file` does `unzip_cmd.split(" ")`, which mangles
     `~/Library/Application Support/sweets`
   Fix: derive the water mask from a Copernicus DEM (`heights > 0` ⇒ land)
   and ship a `~/.cache/sweets` cache dir with no spaces. See `dem.py` and
   `utils.get_cache_dir`.

2. **`COMPASS` still uses `np.string_` / `np.unicode_`**, removed in
   numpy 2.0. Fixed in
   [`scottstanie/COMPASS@develop-scott`](https://github.com/scottstanie/COMPASS/tree/develop-scott)
   (commit `a91a9aa`, 64 sites across `s1_geocode_slc.py`,
   `s1_geocode_metadata.py`, `h5_helpers.py`); sweets pins that branch via
   `[tool.pixi.pypi-dependencies]`. Verified locally that COMPASS now
   produces full ~273 MB CSLC HDF5s with byte-string attributes.

3. **`_get_cfg_setup` built the static-layers path with a date suffix**
   (`static_layers_<burst>_<date>.h5`), but COMPASS writes them per-burst
   without a date. Strip the date before adding the prefix.

4. **`dolphin.workflows.displacement` now requires
   `input_options.subdataset`** for HDF5/NetCDF inputs. Default to
   `/data/VV` in `_dolphin.build_displacement_config`.

5. **`_existing_gslcs()` accepted empty 6-KB CSLC shells** as
   "already done". COMPASS creates the shell early and only writes the
   data later, so any crash mid-run leaves a file that *looks* like a
   valid output. Reject anything below ~1 MB. This is exactly the
   failure mode in issue #107.

**Open caveat from yesterday's smoke test:** burst2safe rejects bboxes
that span more than one IW subswath with `Products from swaths IW1 and
IW2 do not overlap`. Workaround: pass `--swaths IW2` (or whichever
subswath your AOI lives in).

## Smoke-test recipe (pecos)

```bash
# 1. Build the env
pixi install

# 2. Configure a tiny workflow (note the equal sign for the negative bbox
#    longitudes — argparse-style CLIs choke on bare negative numbers).
pixi run sweets config \
  --bbox=-102.96 31.22 -101.91 31.56 \
  --start 2021-06-05 \
  --end 2021-06-22 \
  --track 78 \
  --swaths IW2 \
  --out-dir /Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos_revival/data \
  --work-dir /Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos_revival \
  --output /Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos_revival/sweets_config.yaml

# 3. Run it
pixi run sweets run /Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos_revival/sweets_config.yaml
```

If you already have the pecos SAFEs cached, you can reuse them by pointing
`--out-dir` at
`/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos/safes` and
running with `--starting-step 2` (download is skipped when SAFEs already exist).
