# sweets v0.2 revival notes

A breadcrumb file for the v0.2-rewrite branch — what changed, what's still
loose, and where to look next.

## What changed

| layer | before | after |
|---|---|---|
| **download** | `ASFQuery` → ASF API + wget/aria2c, full S1 frames | `BurstSearch` → `burst2safe.burst2stack`, just the bursts that intersect the AOI |
| **s1-reader** | upstream `isce-framework/s1-reader` (broken on numpy 2, see #132) | `scottstanie/s1-reader@develop-scott` |
| **geocoding** | COMPASS via `_geocode_slcs.py` | unchanged |
| **interferograms / stitch / unwrap / timeseries** | hand-rolled in `sweets.interferogram` + `sweets.core` + `dolphin.unwrap.run` | one call to `dolphin.workflows.displacement.run` via the new `sweets._dolphin` adapter |
| **CLI** | argparse, ~280 lines, manual group dict shuffling | `tyro`, 3 dataclass-style subcommands, ~140 lines |
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
| #88 | "Refactor download to allow OPERA gslcs" | partial — the `BurstSearch` shape is small enough that adding an `OperaCslcSearch` sibling is straightforward (see TODO below) |
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

1. **Smoke-test the full pipeline against pecos** — see `scripts/demo_sweet.py`.
   The bbox there (`-102.96 31.22 -101.91 31.56`, track 78) matches the SAFEs
   already on disk at
   `/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/s1-testing/pecos/safes/`,
   so a re-run can use them as the cache and skip the download step.
2. **Update `docs/sweets-demo.ipynb`** — references the old CLI flags
   (`--track`, etc.) and old output paths (`interferograms/stitched/`). Should
   be rewritten against the new dolphin output layout
   (`dolphin/timeseries/*.tif`, `dolphin/interferograms/*.tif`).
3. **Update `README.md`** — still describes the frame-based download flow.
   Replace with the burst-subset usage and the pixi install path.
4. **Add a `sweets export-mintpy` subcommand** that wraps dolphin's mintpy
   exporter (closes #128 + #42). Should be a thin function in
   `sweets._mintpy.py`.
5. **Refactor the burst-id path for OPERA CSLC downloads (#88).** Pattern:
   add an `OperaCslcSearch` Pydantic model alongside `BurstSearch` and let the
   `Workflow.search` field be a discriminated union. The validators in
   `Workflow._sync_aoi` already handle dict input cleanly.
6. **Wire pixi to run the smoke test in CI.** A `pixi run smoke` task that
   does `python -m sweets config ... && python -m sweets run --starting-step 3`
   against a pre-staged tiny stack would catch the kind of "import works but
   pipeline doesn't" regression that bit several open issues.
7. **Web UI** — left exactly as Scott had it under `src/sweets/web/`. Excluded
   from mypy and from this revival's scope.

## Things I (Claude) deliberately did NOT do

- **Touch `dolphin`.** The user's reference work uses a `develop-scott` fork of
  dolphin, but Scott himself maintains upstream dolphin, so sweets pins
  upstream `dolphin` for now. If you need an unreleased dolphin feature, swap
  in `dolphin = { git = "...", branch = "..." }` under `[tool.pixi.pypi-dependencies]`.
- **Land a real COMPASS numpy 2 fix.** I added a runtime monkey-patch to
  `_geocode_slcs.py` to keep the smoke test moving. The proper fix is to
  PR `np.string_` → `np.bytes_` and `np.unicode_` → `np.str_` against
  `scottstanie/COMPASS` (~64 occurrences across `s1_geocode_*.py` etc.),
  cut a `develop-scott` branch, and pin sweets to it via
  `[tool.pixi.pypi-dependencies]` like we already do for s1-reader. Then
  drop the shim from `_geocode_slcs.py`.
- **Touch the COMPASS / `_geocode_slcs.py` integration.** Geocoding still uses
  COMPASS; the hand-rolled config-file shuffling in `_geocode_slcs.py` is the
  same as on main. If we want to drop COMPASS in favor of an `isce3.geocode_slc`
  call directly, that's a separate (large) job.
- **Notebook updates** — out of scope for this swing.

## Smoke test results (2026-04-10)

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
   numpy 2.0. Patched at the top of `_geocode_slcs.py` with a runtime
   shim before `import compass`. **Real fix is to land this on
   `scottstanie/COMPASS` and pin to it the same way we already pin
   s1-reader.**

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
