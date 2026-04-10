# sweets v0.2 revival notes

A breadcrumb file for the v0.2-rewrite branch — what changed, what's still
loose, and where to look next.

## What changed

| layer | before | after |
|---|---|---|
| **download (default)** | `ASFQuery` → ASF API + wget/aria2c, full S1 frames | `BurstSearch` → `burst2safe.burst2stack`, just the bursts that intersect the AOI |
| **download (alt source)** | n/a | `OperaCslcSearch` → `opera_utils.download.{search,download}_cslcs` for pre-made OPERA CSLCs, plus `download_cslc_static_layers`. Pick via `--source opera-cslc`; locked to OPERA's 5×10 m posting. |
| **s1-reader** | upstream `isce-framework/s1-reader` (broken on numpy 2, see #132) | `scottstanie/s1-reader@develop-scott` |
| **COMPASS** | upstream `opera-adt/COMPASS` (np.string_/np.unicode_ → numpy 2 crash) | `scottstanie/COMPASS@develop-scott` |
| **opera-utils** | conda-forge | `scottstanie/opera-utils@develop-scott` (carries the high-level tropo workflow + the CSLC download API) |
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

1. **Update `docs/sweets-demo.ipynb`** — references the old CLI flags
   (`--track`, etc.) and old output paths (`interferograms/stitched/`).
   Should be rewritten against the new dolphin output layout
   (`dolphin/timeseries/*.tif`, `dolphin/interferograms/*.tif`) and show
   both `--source safe` and `--source opera-cslc`.
2. **Update `README.md`** — still describes the frame-based download flow.
   Replace with the burst-subset usage, the OPERA CSLC alternative, the
   `--do-tropo` knob, and the pixi install path.
3. **Add a `sweets export-mintpy` subcommand** that wraps dolphin's mintpy
   exporter (closes #128 + #42). Should be a thin function in
   `sweets._mintpy.py`.
4. **Land the dolphin commented-yaml fix upstream.** The 1-liner is in
   `_yaml_model.py` at the `anyOf` branch — fall back to the `$ref` name
   when an entry has no `type`. Once that lands, drop
   `sweets/_dolphin_yaml_compat.py`.
5. **Wire pixi to run the smoke test in CI.** A `pixi run smoke` task that
   does `python -m sweets config ... && python -m sweets run` against a
   pre-staged tiny stack would catch the kind of "import works but
   pipeline doesn't" regression that bit several open issues.
6. **Average tropo across burst sensing times.** `apply_tropo_to_unwrapped`
   currently keys tropo files by `YYYYMMDD` and the index dict overwrites
   when there are multiple bursts on the same day; the actual tropo
   correction varies slightly across the ~10-second strip. For small AOIs
   the variance is negligible; for large AOIs an average (or
   nearest-by-burst) lookup would be more correct.
7. **Web UI** — left exactly as Scott had it under `src/sweets/web/`. Excluded
   from mypy and from this revival's scope.

## Things I (Claude) deliberately did NOT do

- **Touch `dolphin`.** The user's reference work uses a `develop-scott` fork of
  dolphin, but Scott himself maintains upstream dolphin, so sweets pins
  upstream `dolphin` for now. If you need an unreleased dolphin feature, swap
  in `dolphin = { git = "...", branch = "..." }` under `[tool.pixi.pypi-dependencies]`.
- **Touch upstream `opera-adt/COMPASS`.** All COMPASS fixes landed on the
  personal fork `scottstanie/COMPASS@develop-scott`; merging them upstream
  is left to whoever is talking to OPERA.
- **Touch the COMPASS / `_geocode_slcs.py` integration.** Geocoding still uses
  COMPASS; the hand-rolled config-file shuffling in `_geocode_slcs.py` is the
  same as on main. If we want to drop COMPASS in favor of an `isce3.geocode_slc`
  call directly, that's a separate (large) job.
- **Notebook updates** — out of scope for this swing.

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
   raises `KeyError` walking an `anyOf` schema entry that's a `$ref`
   to a sub-model — which is exactly what `Workflow.search:
   Union[BurstSearch, OperaCslcSearch]` produces. Worked around with a
   monkey-patch in `sweets._dolphin_yaml_compat`. Real fix is a 1-liner
   upstream in dolphin.
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
