from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class DownloadStrategy(ABC):
    """Abstract base class for download strategies."""

    @abstractmethod
    def download(self, *, log_dir: Path) -> List[Path]:
        """Return list of .SAFE (or .zip) paths ready for the rest of the pipeline."""
        pass
