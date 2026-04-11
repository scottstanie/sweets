"""Sweets command-line interface (tyro-driven).

Three subcommands:

- ``sweets config``  — write a ``sweets_config.yaml`` from a few flags.
- ``sweets run``     — execute a workflow from a config file.
- ``sweets server``  — launch the (WIP) web UI server.

The CLI intentionally exposes only the most common knobs. For the long tail
(dolphin half-window, COMPASS posting, etc.) edit the YAML directly or
construct :class:`sweets.core.Workflow` in Python.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Optional

import tyro

SourceKind = Literal["safe", "opera-cslc", "nisar-gslc"]


@dataclass
class ConfigCmd:
    """Create a sweets_config.yaml from CLI arguments."""

    start: str
    """Start date for the burst search (YYYY-MM-DD)."""

    end: str
    """End date for the burst search (YYYY-MM-DD)."""

    bbox: Optional[tuple[float, float, float, float]] = None
    """AOI as left bottom right top in decimal degrees. One of --bbox or --wkt is required."""

    wkt: Optional[str] = None
    """AOI as a WKT polygon string, or path to a .wkt file. Overrides --bbox."""

    source: SourceKind = "safe"
    """Where the input SLCs come from. `safe` (default): raw S1 bursts via burst2safe + COMPASS. `opera-cslc`: pre-made OPERA CSLC HDF5s from ASF. `nisar-gslc`: pre-made NISAR GSLC HDF5s via CMR (L-band, UTM, already geocoded)."""

    track: Optional[int] = None
    """Relative orbit / track number. Required for --source safe; optional but recommended for --source opera-cslc and --source nisar-gslc. For NISAR this is the `Track` field on ASF Vertex (the RRR digits in the granule filename)."""

    frame: Optional[int] = None
    """NISAR track-frame number — the `Frame` field on ASF Vertex (the TTT digits in the granule filename, e.g. `71`). Only honored by --source nisar-gslc."""

    frequency: Literal["A", "B"] = "A"
    """NISAR frequency band (`A` = L-band, `B` reserved). Only honored by --source nisar-gslc."""

    out_dir: Path = field(default_factory=lambda: Path("data"))
    """Where downloaded SLC inputs will live."""

    work_dir: Path = field(default_factory=Path.cwd)
    """Top-level working directory for the workflow."""

    polarizations: list[str] = field(default_factory=lambda: ["VV"])
    """Polarizations to keep. Defaults to ['VV'] for S1/OPERA; pass --polarizations HH for NISAR."""

    swaths: Optional[list[str]] = None
    """Restrict to specific subswaths (e.g. ['IW2']). Only honored by --source safe."""

    n_workers: int = 4
    """Process pool size for COMPASS geocoding (--source safe only)."""

    do_tropo: bool = False
    """Run the OPERA L4 TROPO-ZENITH correction step after dolphin (off by default; not supported with --source nisar-gslc)."""

    output: Path = Path("sweets_config.yaml")
    """Where to write the config file."""

    with_schema: bool = True
    """Also write a sibling `<output>.schema.json` next to the YAML and
    prepend a `# yaml-language-server: $schema=...` modeline. Editors
    with the YAML Language Server (VS Code, Neovim-yamlls, JetBrains,
    etc.) use that to provide inline hover docs, autocomplete, and
    validation for every field in the sweets config. Pass
    --no-with-schema to skip."""

    def run(self) -> None:
        """Build and dump a Workflow config to YAML."""
        # Heavy imports go here so `sweets --help` is snappy.
        from sweets.core import Workflow

        if self.bbox is None and self.wkt is None:
            print("error: one of --bbox or --wkt is required", file=sys.stderr)
            raise SystemExit(2)

        search: dict = {
            "kind": self.source,
            "start": self.start,
            "end": self.end,
            "out_dir": self.out_dir,
        }
        if self.source == "safe":
            if self.track is None:
                print("error: --track is required for --source safe", file=sys.stderr)
                raise SystemExit(2)
            search["track"] = self.track
            search["polarizations"] = self.polarizations
            search["swaths"] = self.swaths
        elif self.source == "opera-cslc":
            # `track` is optional on OPERA — ASF will filter on AOI alone.
            if self.track is not None:
                search["track"] = self.track
        elif self.source == "nisar-gslc":
            if self.track is not None:
                search["track"] = self.track
            if self.frame is not None:
                search["frame"] = self.frame
            search["frequency"] = self.frequency
            search["polarizations"] = self.polarizations

        workflow = Workflow.model_validate(
            {
                "bbox": self.bbox,
                "wkt": self.wkt,
                "work_dir": self.work_dir,
                "n_workers": self.n_workers,
                "search": search,
                "tropo": {"enabled": self.do_tropo},
            }
        )
        workflow.to_yaml(self.output)
        if self.with_schema:
            _emit_schema_sidecar(self.output)
        print(f"wrote {self.output}", file=sys.stderr)


def _emit_schema_sidecar(yaml_path: Path) -> None:
    """Write a JSON schema next to the YAML and add a modeline comment.

    The schema is `Workflow.model_json_schema()` emitted verbatim; pydantic
    produces JSON Schema Draft 2020-12 with a `oneOf + discriminator` for
    the `Workflow.search` field, which the YAML Language Server handles
    natively. The modeline is read by the redhat.vscode-yaml extension
    (and every editor that speaks yamlls) to attach the schema at load
    time.
    """
    import json

    from sweets.core import Workflow

    schema_path = yaml_path.with_suffix(yaml_path.suffix + ".schema.json")
    schema_path.write_text(json.dumps(Workflow.model_json_schema(), indent=2) + "\n")

    existing = yaml_path.read_text()
    modeline = f"# yaml-language-server: $schema={schema_path.name}\n"
    if modeline.strip() not in existing:
        yaml_path.write_text(modeline + existing)
    print(f"wrote {schema_path}", file=sys.stderr)


@dataclass
class SchemaCmd:
    """Dump the JSON schema for the sweets workflow config to stdout."""

    def run(self) -> None:
        import json

        from sweets.core import Workflow

        print(json.dumps(Workflow.model_json_schema(), indent=2))


@dataclass
class ReportCmd:
    """Render a single-file HTML report for a finished sweets run."""

    work_dir: Annotated[Path, tyro.conf.Positional]
    """Path to the sweets work directory (the one that contains `dolphin/`)."""

    output: Optional[Path] = None
    """Where to write the report. Defaults to `<work_dir>/sweets_report.html`."""

    config_file: Optional[Path] = None
    """Override path to the `sweets_config.yaml` used for the run.
    Defaults to the first `*.yaml` found in `work_dir` (excluding
    `dolphin_config.yaml`)."""

    def run(self) -> None:
        from sweets._report import build_report

        path = build_report(
            work_dir=self.work_dir,
            output=self.output,
            config_path=self.config_file,
        )
        print(f"wrote {path}", file=sys.stderr)


@dataclass
class RunCmd:
    """Execute a sweets workflow from a config file."""

    config_file: Annotated[Path, tyro.conf.Positional]
    """Path to a sweets_config.yaml."""

    starting_step: int = 1
    """Skip earlier stages (1=download, 2=geocode, 3=dolphin)."""

    def run(self) -> None:
        """Load the workflow and run it."""
        from sweets.core import Workflow

        if not self.config_file.exists():
            msg = f"config file {self.config_file} does not exist"
            raise SystemExit(msg)
        workflow = Workflow.from_yaml(self.config_file)
        workflow.run(starting_step=self.starting_step)


@dataclass
class ServerCmd:
    """Launch the (WIP) sweets web UI server."""

    host: str = "127.0.0.1"
    """Bind address."""

    port: int = 8000
    """TCP port."""

    reload: bool = False
    """Auto-reload on code changes (dev mode)."""

    def run(self) -> None:
        """Run uvicorn against sweets.web.app:app."""
        try:
            import uvicorn
        except ImportError:
            print(
                "Web dependencies not installed. Install with one of:\n"
                "  pip install 'sweets[web]'\n"
                "  pixi install -e web",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        uvicorn.run(
            "sweets.web.app:app",
            host=self.host,
            port=self.port,
            reload=self.reload,
        )


def main() -> None:
    """Top-level CLI entry point."""
    cmd = tyro.extras.subcommand_cli_from_dict(
        {
            "config": ConfigCmd,
            "run": RunCmd,
            "schema": SchemaCmd,
            "report": ReportCmd,
            "server": ServerCmd,
        },
        prog="sweets",
        description="Sentinel-1 InSAR workflow runner.",
    )
    cmd.run()


if __name__ == "__main__":
    main()
