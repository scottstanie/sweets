"""Sentinel-1 burst download via burst2safe.

Replaces the old frame-based ASF query / wget pipeline. Bursts are downloaded
straight from the ASF DAAC into ``.SAFE`` directories that cover only the
requested area, dramatically reducing data volume for small AOIs.

Authentication relies on a ``~/.netrc`` entry for ``urs.earthdata.nasa.gov``,
which is what burst2safe and ``sentineleof`` already expect.

Examples
--------
>>> from datetime import datetime
>>> search = BurstSearch(
...     bbox=(-102.96, 31.22, -101.91, 31.56),
...     start=datetime(2021, 6, 1),
...     end=datetime(2021, 8, 10),
...     track=78,
... )
>>> safes = search.download()  # doctest: +SKIP
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional

from dateutil.parser import parse as parse_date
from dolphin.workflows.config import YamlModel
from pydantic import ConfigDict, Field, field_validator, model_validator
from shapely import wkt as shp_wkt
from shapely.geometry import Polygon, box

from ._log import get_log, log_runtime

logger = get_log(__name__)


FlightDirection = Literal["ASCENDING", "DESCENDING"]


class BurstSearch(YamlModel):
    """Sentinel-1 burst search/download configuration.

    Wraps :func:`burst2safe.burst2stack.burst2stack` so the user can pin a
    small AOI (bbox or WKT polygon) plus a date range and a track number,
    and get back ``.SAFE`` directories containing only the bursts that
    intersect the AOI.
    """

    out_dir: Path = Field(
        Path("data"),
        description="Directory where SAFE directories will be written.",
        validate_default=True,
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest as (left, bottom, right, top) in decimal degrees."
            " Either `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a WKT polygon string (or path to a `.wkt` file)."
            " Takes precedence over `bbox` if both are provided."
        ),
    )
    start: datetime = Field(
        ...,
        description="Search start time (parsed by `dateutil.parser`).",
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description="Search end time. Defaults to now.",
    )
    track: Optional[int] = Field(
        None,
        alias="relativeOrbit",
        description="Sentinel-1 relative orbit / track number.",
    )
    flight_direction: Optional[FlightDirection] = Field(
        None,
        alias="flightDirection",
        description="Restrict to ASCENDING or DESCENDING acquisitions.",
    )
    polarizations: list[str] = Field(
        default_factory=lambda: ["VV"],
        description="Polarizations to include (e.g. ['VV'], ['VV', 'VH']).",
    )
    swaths: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to specific subswaths (e.g. ['IW2']). If None, all swaths"
            " covering the AOI are downloaded."
        ),
    )
    min_bursts: int = Field(
        1,
        description="Minimum number of bursts a SAFE must contain to be kept.",
        ge=1,
    )
    all_anns: bool = Field(
        True,
        description=(
            "Include annotations for all swaths in the produced SAFE files."
            " Required by `s1-reader` / COMPASS, which always reads the IW2"
            " annotation regardless of the subswath being processed."
        ),
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, date):
            return datetime.combine(v, datetime.min.time())
        return parse_date(str(v))

    @field_validator("out_dir")
    @classmethod
    def _absolute_out_dir(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("flight_direction", mode="before")
    @classmethod
    def _normalize_flight_direction(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).upper()
        if s.startswith("A"):
            return "ASCENDING"
        if s.startswith("D"):
            return "DESCENDING"
        msg = f"Unrecognized flight direction: {v!r}"
        raise ValueError(msg)

    @field_validator("polarizations")
    @classmethod
    def _upper_pols(cls, v: list[str]) -> list[str]:
        return [p.upper() for p in v]

    @field_validator("swaths")
    @classmethod
    def _upper_swaths(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return [s.upper() for s in v] if v else v

    @model_validator(mode="after")
    def _check_aoi_and_dates(self) -> "BurstSearch":
        if not self.wkt and not self.bbox:
            msg = "Must provide either `bbox` or `wkt`"
            raise ValueError(msg)
        if self.wkt and Path(self.wkt).exists():
            self.wkt = Path(self.wkt).read_text().strip()
        if self.wkt:
            try:
                shp_wkt.loads(self.wkt)
            except Exception as e:
                msg = f"Invalid WKT polygon: {e}"
                raise ValueError(msg) from e
        if self.end < self.start:
            msg = f"`end` ({self.end}) must be after `start` ({self.start})"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def aoi(self) -> Polygon:
        """Return the search AOI as a shapely Polygon."""
        if self.wkt:
            return shp_wkt.loads(self.wkt)
        assert self.bbox is not None  # enforced in validator
        return box(*self.bbox)

    def summary(self) -> str:
        """Return a human-readable summary of the planned search."""
        bounds = self.aoi.bounds
        return (
            "BurstSearch:\n"
            f"  AOI bounds : {bounds}\n"
            f"  Dates      : {self.start.date()} -> {self.end.date()}\n"
            f"  Track      : {self.track}\n"
            f"  Direction  : {self.flight_direction or 'any'}\n"
            f"  Pols       : {self.polarizations}\n"
            f"  Swaths     : {self.swaths or 'any'}\n"
            f"  Output     : {self.out_dir}"
        )

    @log_runtime
    def download(self) -> list[Path]:
        """Download bursts covering the AOI as SAFE directories.

        Returns
        -------
        list[Path]
            Paths of the produced ``.SAFE`` directories.

        """
        # Imported lazily so importing this module is cheap and so users
        # without burst2safe still get a clear error.
        from burst2safe.burst2stack import burst2stack

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(self.summary())

        result = burst2stack(
            rel_orbit=self.track,
            start_date=self.start,
            end_date=self.end,
            extent=self.aoi,
            polarizations=self.polarizations,
            swaths=self.swaths,
            min_bursts=self.min_bursts,
            all_anns=self.all_anns,
            work_dir=self.out_dir,
        )
        safes = sorted(Path(p) for p in result)
        logger.info(f"Downloaded {len(safes)} SAFE directories to {self.out_dir}")
        if self.flight_direction is not None:
            safes = _filter_by_flight_direction(safes, self.flight_direction)
        return safes

    def existing_safes(self) -> list[Path]:
        """Return any SAFEs already present in `out_dir` (does not query ASF)."""
        return sorted(self.out_dir.glob("S1[AB]_*.SAFE"))


def _filter_by_flight_direction(
    safes: list[Path], flight_direction: FlightDirection
) -> list[Path]:
    """Drop SAFEs whose first manifest does not match `flight_direction`.

    burst2safe does not expose a flight-direction filter directly. We can
    cheaply infer it from the manifest.safe inside the .SAFE bundle.
    """
    import xml.etree.ElementTree as ET

    keep: list[Path] = []
    for s in safes:
        manifest = s / "manifest.safe"
        if not manifest.exists():
            keep.append(s)
            continue
        try:
            tree = ET.parse(manifest)
        except ET.ParseError as e:
            logger.warning(f"Could not parse {manifest}: {e}; keeping SAFE.")
            keep.append(s)
            continue
        text = ET.tostring(tree.getroot(), encoding="unicode")
        upper = flight_direction.upper()
        if upper in text.upper():
            keep.append(s)
        else:
            logger.info(f"Dropping {s.name}: not {upper}")
    return keep
