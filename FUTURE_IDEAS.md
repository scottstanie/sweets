# sweets future-ideas

Running brainstorm of features worth considering after v0.2 lands. Ordered
loosely by value-to-effort. Not a roadmap — most of these are single-PR
sized and any one of them could reasonably go next.

## Scientific / practitioner-facing

### 1. Corrections stack beyond tropo (ionosphere + SET + plate motion)
Tropo alone isn't credible for L-band NISAR — the ionosphere shift can be
multiple cm, and solid earth tides + plate motion matter for long-baseline
stacks even at C-band. `opera_utils` already has `solid_earth_tide` and
plate-motion helpers. What's missing is an umbrella config:

```python
class Corrections(YamlModel):
    tropo:  Optional[TropoOptions] = None     # existing
    iono:   Optional[IonoOptions]  = None     # GIM / F10.7-based
    set:    bool = False                      # solid earth tides
    plate:  bool = False                      # plate motion
```

and a single ordered post-dolphin loop that applies whichever are enabled.
Biggest credibility gap on the NISAR path today.

### 2. Reference-point control
dolphin auto-picks a reference from the longest coherent connected component.
Practitioners almost always want it pinned — to a `(lat, lon)`, a box average
over a stable region, or a named GNSS station. Expose a
`reference_point: Optional[tuple[float, float]]` on `Workflow`, plumb it into
`dolphin.timeseries.ReferencePoint`, and add a `--reference-point lat lon`
CLI flag. Single-PR feature that changes every published result.

### 3. GNSS integration (read-only first, then calibration)
One command that pulls GNSS velocities from the Nevada Geodetic Lab (or UNR
MIDAS) for stations inside the AOI, resamples them into the velocity
raster's CRS, and prints a per-station scatter of InSAR vs GNSS. Once that
works, the same helper doubles as a calibration source (fit a plane to the
residuals and subtract to remove long-wavelength ramps). Practitioners do
this by hand today.

### 4. Interferogram network knobs beyond `--max-bandwidth`
dolphin's `interferogram.Network` already supports SBAS-style (max temporal
+ perpendicular baseline), single-reference ("star"), and explicit pair
lists. sweets only exposes `max_bandwidth`. Add `network_type` +
thresholds to `DolphinOptions`.

### 5. Incremental / append-mode processing
"I have a stack up to Dec 2025 and a new acquisition came in today — extend
without redoing the old work." dolphin already supports ministack-style
appending. `sweets run --append` would detect new CSLCs, run phase linking
on just the new ministack, and re-run only the ifg / unwrap / timeseries
steps that depend on them. Real operational feature; currently users have
to rerun the whole thing.

### 6. Coherence-based masking after unwrap
Drop pixels below a temporal-coherence threshold (with optional
morphological cleanup) before the timeseries inversion. Noisy pixels
currently bleed into the fit. One-line wire-up:
`mask_by_temporal_coherence: 0.4` in `DolphinOptions`.

### 7. Pre-flight data availability check (`sweets check <config>`)
Raised in this conversation. Separate from the post-run report below.
After `sweets config` and before `sweets run`, it would:

- Run every search query dry
- Report per-source hit count, estimated download size, wall-time budget
- Draw the burst/frame footprints over the AOI
- **Flag multi-track hits** (descending track 71 + ascending track 64 in
  the same result set means the AOI straddles an orbit — currently
  sweets auto-picks one via `--track` but a missing pin produces silent
  mixed-track output)
- Print which bursts would survive the missing-data filter
- Estimate how many interferograms dolphin would form

Essentially "what is going to come out?" in advance. The user is right that
multi-track decomposition is its own project — pre-run just needs to warn,
not compose.

---

## Reports + visualization

### R1. `sweets report <work_dir>` → single HTML file
The single highest-leverage UX item. One command scans a finished run and
emits a self-contained HTML with:

- Velocity plot (auto colormap, percentile clipping, coordinate overlay)
- Cumulative displacement for the longest-baseline pair
- Network diagram (which acquisitions, which pairs, which dropped)
- Per-pair coherence histogram + summary stats
- Runtime breakdown per workflow step
- Side-by-side of the DEM, water mask, and final bounds mask

You can send the link to a collaborator without them installing anything.
Turns every sweets run into something shareable.

### R2. `sweets plot` subcommand (backed by bowser)
`bowser` already has `setup-dolphin <work_dir>` that writes a
`bowser_rasters.json` and a web UI that serves the timeseries / velocity /
coherence / ifg layers from a dolphin output tree. The right move is to
wrap it, not to reinvent it:

```bash
sweets plot <work_dir>  # = cd work_dir && bowser setup-dolphin . && bowser run
```

Ship bowser as an **optional install** (`sweets[viewer]` extra) because
its `titiler` dep has been flaky in the past. Sweets-core stays
slim; users who want the viewer add one extra pip install.

---

## Install / packaging / ergonomics

### P2. JSON Schema for `sweets_config.yaml`
**Tried in this branch** — see #P2-result below.

pydantic's `Workflow.model_json_schema()` emits a JSON Schema that the
[YAML language server](https://github.com/redhat-developer/yaml-language-server)
(the one VS Code, Neovim-with-yamlls, and JetBrains all use) reads for
autocomplete, hover docs, and inline validation. Add a
`sweets config --print-schema > sweets_config.schema.json` step and put
a one-line modeline in the generated YAML:

```yaml
# yaml-language-server: $schema=./sweets_config.schema.json
```

Every field that has a pydantic `description` gets a hover popup; every
`Literal[...]` field gets a dropdown; typos get red-underlined. Five
minutes of work per user gets them a full IDE experience on the config
file.

### P3. `sweets init` interactive wizard
Prompt for AOI (accept bbox, WKT, or drag-paste from ASF Vertex), dates,
source, track/frame. Write an annotated YAML with every default spelled
out. Currently a new user has to learn `sweets config --help` cold.

### P4. `rich.Progress` for long steps
Persistent progress bars for download / geocode / phase-link / unwrap /
timeseries with elapsed + ETA, instead of the current stream of log
lines. Would have caught the "is it stuck or downloading?" confusion
that came up while testing the NISAR path. `rich` is already a
transitive dep.

### P5. Structured (JSON) logs, opt-in
`--log-format json` emits one event per log line with step name, elapsed,
memory peak, error fields. Makes CI smoke tests diffable, makes metrics
scraping trivial, and lets the HTML report in R1 read a run's history
without re-parsing terminal text.

---

## Deliberately not doing (for now)

- **Full web UI as a first-party sweets thing.** The old scaffold lives
  on branch `web-ui-scaffold`; it was excluded from the v0.3 PR because
  it was never tested and carried untested deps (sqlmodel, titiler,
  etc.). bowser handles the interactive-map use case much better
  already, and R1 (HTML report) covers the "email the result" use case
  with zero dependencies. A custom sweets web UI would multiply
  maintenance without expanding the audience beyond what bowser already
  serves.
- **Automatic source-model fitting (Mogi, Okada, fault-slip).** Real
  InSAR practitioners have their own preferred inversion stacks. sweets
  competing with them is a tarpit.
- **Multi-track east/up decomposition as a built-in feature.** It's
  real and valuable but the "do it right" version is its own project
  (reference-frame alignment, baseline-overlap handling, cross-orbit
  timing). Leave as a downstream post-process for now; sweets' job is
  to produce one clean per-track velocity raster that the user can
  combine externally.

---

## P2-result: JSON Schema experiment (landed)

Tried it on the v0.2-rewrite branch and it just worked — `sweets config`
now writes a sidecar `<config>.schema.json` next to the YAML and
prepends a `# yaml-language-server: $schema=<basename>.schema.json`
modeline so editors with the YAML Language Server (VS Code
`redhat.vscode-yaml`, Neovim yamlls, JetBrains) pick it up automatically.
A standalone `sweets schema` subcommand dumps the same schema to stdout
for anyone who wants to pipe it into their own tooling.

Mechanics:

- `Workflow.model_json_schema()` emits JSON Schema Draft 2020-12 with
  a clean `oneOf + discriminator` on `search`. pydantic handles the
  discriminated union correctly out of the box — the schema maps
  `safe` / `opera-cslc` / `nisar-gslc` to `BurstSearch` /
  `OperaCslcSearch` / `NisarGslcSearch` definitions, so the editor
  knows which fields are valid for each source.
- Every pydantic `description=` on a field becomes a hover tooltip.
- `Literal[...]` fields (e.g. `unwrap_method`, `pol_type`,
  `source`) turn into dropdowns.
- Numeric/string constraints round-trip to the schema so typos and
  out-of-range values get red-underlined in the editor.
- Sidecar output is ~25 KB; `--no-with-schema` disables.

Not removing the feature. This section stays as a note on the
implementation so future maintainers know why there's a modeline in
every generated YAML.
