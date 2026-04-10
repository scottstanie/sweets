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

SourceKind = Literal["safe", "opera-cslc"]


@dataclass
class ConfigCmd:
    """Create a sweets_config.yaml from CLI arguments."""

    start: str
    """Start date for the burst search (YYYY-MM-DD)."""

    end: str
    """End date for the burst search (YYYY-MM-DD)."""

    track: int
    """Sentinel-1 relative orbit / track number."""

    bbox: Optional[tuple[float, float, float, float]] = None
    """AOI as left bottom right top in decimal degrees. One of --bbox or --wkt is required."""

    wkt: Optional[str] = None
    """AOI as a WKT polygon string, or path to a .wkt file. Overrides --bbox."""

    source: SourceKind = "safe"
    """Where the input SLCs come from. `safe` (default) downloads raw S1 bursts via burst2safe and runs COMPASS; `opera-cslc` pulls pre-made OPERA CSLC HDF5s from ASF (skips COMPASS, locked to OPERA's posting)."""

    out_dir: Path = field(default_factory=lambda: Path("data"))
    """Where downloaded SLC inputs (SAFEs or OPERA CSLC HDF5s) will live."""

    work_dir: Path = field(default_factory=Path.cwd)
    """Top-level working directory for the workflow."""

    polarizations: list[str] = field(default_factory=lambda: ["VV"])
    """Polarizations to keep (only honored by --source safe)."""

    swaths: Optional[list[str]] = None
    """Restrict to specific subswaths (e.g. ['IW2']). Only honored by --source safe."""

    n_workers: int = 4
    """Process pool size for COMPASS geocoding (--source safe only)."""

    do_tropo: bool = False
    """Run the OPERA L4 TROPO-ZENITH correction step after dolphin (off by default)."""

    output: Path = Path("sweets_config.yaml")
    """Where to write the config file."""

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
            "track": self.track,
            "out_dir": self.out_dir,
        }
        # SAFE-only knobs
        if self.source == "safe":
            search["polarizations"] = self.polarizations
            search["swaths"] = self.swaths

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
        print(f"wrote {self.output}", file=sys.stderr)


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
            "server": ServerCmd,
        },
        prog="sweets",
        description="Sentinel-1 InSAR workflow runner.",
    )
    cmd.run()


if __name__ == "__main__":
    main()
