#!/usr/bin/env python
"""Burst subset download strategy using burst2stack."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from dolphin.workflows.config import YamlModel
from pydantic import Field, field_validator

from .._log import get_log
from ._strategy import DownloadStrategy

logger = get_log(__name__)


class BurstSubsetDownload(DownloadStrategy, YamlModel):
    """Download strategy for burst subsets using burst2stack."""

    method: Literal["burst"] = "burst"

    # Mirror burst2stack args
    rel_orbit: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    extent: Optional[tuple[float, float, float, float]] = Field(
        None, description="Bounding box extent as (left, bottom, right, top)"
    )
    polarizations: Optional[list[str]] = Field(
        None, description="List of polarizations to include (e.g., ['VV', 'VH'])"
    )
    swaths: Optional[list[str]] = Field(
        None, description="List of swaths to include (e.g., ['IW1', 'IW2', 'IW3'])"
    )
    mode: str = Field("IW", description="The collection mode to use (IW or EW)")
    min_bursts: int = Field(1, description="The minimum number of bursts per swath")
    all_anns: bool = Field(
        False,
        description="Include product annotation files for all swaths, regardless of included bursts",
    )
    keep_files: bool = Field(False, description="Keep the intermediate files")
    work_dir: Path = Field(
        Path("data"), description="The directory to create the SAFE files in"
    )

    @field_validator("work_dir")
    @classmethod
    def _resolve_work_dir(cls, v):
        return Path(v).resolve()

    def _optional(self, flag: str, value, joiner: str = " ") -> list[str]:
        """Helper to format optional CLI arguments."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            if joiner == " ":
                return [flag, joiner.join(map(str, value))]
            else:
                return [flag, joiner.join(map(str, value))]
        return [flag, str(value)]

    def _repeat(self, flag: str, values: Optional[list]) -> list[str]:
        """Helper to format repeated CLI arguments."""
        if values is None:
            return []
        result = []
        for value in values:
            result.extend([flag, str(value)])
        return result

    def download(self, *, log_dir: Path) -> List[Path]:
        """Download burst subsets using burst2stack."""
        cmd_parts = ["burst2stack"]

        # Add required output directory
        cmd_parts.extend(["--output-dir", str(self.work_dir)])

        # Add optional arguments
        cmd_parts.extend(self._optional("--rel-orbit", self.rel_orbit))

        if self.start_date:
            cmd_parts.extend(["--start-date", self.start_date.strftime("%Y-%m-%d")])
        if self.end_date:
            cmd_parts.extend(["--end-date", self.end_date.strftime("%Y-%m-%d")])

        cmd_parts.extend(self._optional("--extent", self.extent, joiner=" "))
        cmd_parts.extend(self._repeat("--pols", self.polarizations))
        cmd_parts.extend(self._repeat("--swaths", self.swaths))
        cmd_parts.extend(["--mode", self.mode])
        cmd_parts.extend(["--min-bursts", str(self.min_bursts)])

        if self.all_anns:
            cmd_parts.append("--all-anns")
        if self.keep_files:
            cmd_parts.append("--keep-files")

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Log the command
        cmd_str = " ".join(cmd_parts)
        logger.info(f"Running burst2stack command: {cmd_str}")

        # Run burst2stack
        log_file = log_dir / "burst2stack.log"
        try:
            with open(log_file, "w") as f:
                _result = subprocess.run(
                    cmd_parts, check=True, stdout=f, stderr=subprocess.STDOUT, text=True
                )
        except subprocess.CalledProcessError as e:
            logger.error(f"burst2stack failed with return code {e.returncode}")
            logger.error(f"Check log file: {log_file}")
            raise

        # burst2stack writes full-fledged .SAFE directories inside work_dir
        safe_files = sorted(self.work_dir.glob("S*.SAFE"))
        logger.info(f"Found {len(safe_files)} SAFE files after burst2stack")

        if not safe_files:
            raise ValueError("No SAFE files found after burst2stack execution")

        return safe_files
