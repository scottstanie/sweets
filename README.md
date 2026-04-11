# sweets
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isce-framework/sweets/main.svg)](https://results.pre-commit.ci/latest/github/isce-framework/sweets/main)

End-to-end InSAR workflow that turns a single AOI + date range into unwrapped
interferograms, a displacement timeseries and a velocity raster. sweets handles
the messy parts — burst-subset downloading, geocoding, DEM + water-mask prep,
geometry stitching — and hands off the heavy numerical work to
[dolphin](https://github.com/isce-framework/dolphin) for phase linking,
network selection, unwrapping, timeseries inversion and velocity estimation.

## What sweets gives you

Three interchangeable input sources, all plumbed through the same
`Workflow` object:

| `--source` | What it is | Tools used |
|---|---|---|
| `safe` *(default)* | Raw Sentinel-1 bursts, downloaded as just the bursts that intersect your AOI via `burst2safe`, then geocoded by COMPASS. | burst2safe, s1-reader, COMPASS |
| `opera-cslc` | Pre-made [OPERA L2 CSLC-S1 HDF5s](https://www.jpl.nasa.gov/go/opera/products/cslc-product) from ASF DAAC, plus their matching CSLC-STATIC layers for geometry. Faster than `safe` when OPERA has produced for your AOI (mostly CONUS). | opera-utils |
| `nisar-gslc` | Pre-made [NISAR L2 GSLC HDF5s](https://nisar.jpl.nasa.gov/) via CMR — L-band, already geocoded in UTM. Skips COMPASS and geometry stitching entirely; dolphin reads the grid straight from the HDF5 via tiny VRT wrappers that sweets injects. | opera-utils |

An optional `--do-tropo` flag adds a post-dolphin OPERA L4 TROPO-ZENITH
correction step for `safe` and `opera-cslc` (not yet wired for NISAR).

Outputs land under `<work_dir>/dolphin/`:

- `interferograms/<date1>_<date2>.int.tif` + `.cor.tif`
- `unwrapped/<date1>_<date2>.unw.tif` + `.unw.conncomp.tif`
- `timeseries/<date1>_<date2>.tif`, `velocity.tif`, `reference_point.txt`
- with `--do-tropo`: `tropo/` and `tropo_corrected/<pair>.tropo_corrected.unw.tif`

Timeseries and velocity rasters are in meters and meters/year, with the
radar wavelength auto-detected from the source (S1, OPERA, NISAR).

## Install

sweets is pixi-first. The `[tool.pixi.*]` sections in `pyproject.toml` are the
canonical environment definition and pin the specific forks of `s1-reader`,
`COMPASS`, `opera-utils` and `dolphin` that carry the numpy 2 fixes and the
new workflows sweets uses.

```bash
git clone https://github.com/isce-framework/sweets.git && cd sweets
pixi install
pixi shell
```

That drops you into an env where `sweets`, `dolphin` and `compass` are all on
the `PATH` and sweets is installed in editable mode.

If you're stuck without pixi, there's a derived `environment.yml` suitable for
conda/mamba, but you'll have to pin the fork versions yourself.

## Usage

### Command line

```bash
sweets config --help
sweets run --help
```

To configure a workflow the minimum inputs are an AOI (`--bbox` or `--wkt`), a
date range (`--start` / `--end`), and — for the Sentinel-1 path — the relative
orbit (`--track`).

Raw Sentinel-1 bursts (default):

```bash
sweets config \
  --bbox=-102.96 31.22 -101.91 31.56 \
  --start 2021-06-05 --end 2021-06-22 \
  --track 78 \
  --swaths IW2 \
  --out-dir ./data \
  --work-dir ./pecos_demo \
  --output pecos_demo/sweets_config.yaml

sweets run pecos_demo/sweets_config.yaml
```

Pre-made OPERA CSLCs (faster for CONUS, no COMPASS needed):

```bash
sweets config \
  --bbox=-102.96 31.22 -101.91 31.56 \
  --start 2021-06-05 --end 2021-06-22 \
  --source opera-cslc \
  --out-dir ./data \
  --work-dir ./pecos_opera \
  --output pecos_opera/sweets_config.yaml

sweets run pecos_opera/sweets_config.yaml
```

With the optional tropospheric correction step:

```bash
sweets config --source opera-cslc --do-tropo ... --output pecos_opera/sweets_config.yaml
sweets run pecos_opera/sweets_config.yaml
```

NISAR GSLCs:

```bash
sweets config \
  --bbox=-121.10 36.55 -120.95 36.70 \
  --start 2025-10-01 --end 2025-12-15 \
  --source nisar-gslc \
  --track 42 --frame 70 \
  --out-dir ./data \
  --work-dir ./salinas_nisar \
  --output salinas_nisar/nisar.yaml

sweets run salinas_nisar/nisar.yaml
```

### Starting at a later step

`sweets run` accepts `--starting-step N` so you can skip earlier stages if the
outputs are already on disk:

```bash
sweets run config.yaml --starting-step 2  # skip download, run geocode + dolphin
sweets run config.yaml --starting-step 3  # just (re-)run dolphin
```

### Configuration from Python

You can also build a `Workflow` directly:

```python
from sweets.core import Workflow
from sweets.download import BurstSearch

w = Workflow(
    bbox=(-102.96, 31.22, -101.91, 31.56),
    search=BurstSearch(
        track=78,
        start="2021-06-05",
        end="2021-06-22",
        swaths=["IW2"],
        out_dir="data",
    ),
    work_dir="pecos_demo",
)
w.to_yaml("pecos_demo/sweets_config.yaml")
w.run()
```

The same pattern works for `OperaCslcSearch` and `NisarGslcSearch` — pick
whichever variant matches your data source. `Workflow.search` is a
discriminated union keyed on a `kind` field, so a YAML file with
`search: {kind: opera-cslc, ...}` round-trips correctly.

## License

This software is licensed under your choice of BSD-3-Clause or Apache-2.0
licenses. See the accompanying LICENSE file for further details.

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
