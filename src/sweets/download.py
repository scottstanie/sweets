"""Sensor source models: raw S1 bursts, pre-made OPERA CSLCs, or NISAR GSLCs.

Three source classes are exposed; all are :class:`YamlModel` Pydantic models
with the same external shape (an AOI, a date range, an optional track and
``out_dir``) so :class:`sweets.core.Workflow` can swap between them:

- :class:`BurstSearch` — wraps :func:`burst2safe.burst2stack.burst2stack`
  to download burst-trimmed ``.SAFE`` directories that the rest of the
  workflow then geocodes via COMPASS. Default; works anywhere S1 flies.
- :class:`OperaCslcSearch` — wraps :func:`opera_utils.download.download_cslcs`
  + :func:`opera_utils.download.download_cslc_static_layers` to grab
  pre-geocoded OPERA CSLC HDF5s + their static layers from ASF DAAC. Skips
  COMPASS entirely; locked to OPERA's 5 m × 10 m posting; CONUS-friendly
  but coverage depends on what OPERA has actually produced for the AOI.
- :class:`NisarGslcSearch` — wraps :func:`opera_utils.nisar.run_download`
  to grab pre-geocoded NISAR GSLC HDF5s (L-band, UTM, 5×10 m posting)
  with CMR-based search and optional bbox-level subsetting. Skips COMPASS
  and static layer stitching (NISAR GSLCs have no separate static layers
  product). Coverage and availability depend on NISAR's acquisition plan.

Authentication for any source relies on a ``~/.netrc`` entry for
``urs.earthdata.nasa.gov``.
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

    kind: Literal["safe"] = Field(
        default="safe",
        description="Discriminator for the source type. Always `safe`.",
    )
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


# ----------------------------------------------------------------------------
# OPERA CSLC source
# ----------------------------------------------------------------------------


class OperaCslcSearch(YamlModel):
    """Pre-made OPERA CSLC search/download configuration.

    Wraps :func:`opera_utils.download.download_cslcs` and
    :func:`opera_utils.download.download_cslc_static_layers` to fetch
    pre-geocoded OPERA CSLC HDF5s + their per-burst static layers from the
    ASF DAAC. Posting is whatever OPERA produced (currently 5 m × 10 m for
    Sentinel-1 OPERA CSLCs); use :class:`BurstSearch` instead if you need
    a custom posting.
    """

    kind: Literal["opera-cslc"] = Field(
        default="opera-cslc",
        description="Discriminator for the source type. Always `opera-cslc`.",
    )
    out_dir: Path = Field(
        Path("data"),
        description=(
            "Directory where the OPERA CSLC HDF5s and static layers will be"
            " written. Static layers go into a `static_layers/` subdirectory."
        ),
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
    burst_ids: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to specific OPERA burst IDs (e.g. ['t078_165573_iw2']);"
            " if None, ASF returns whichever bursts intersect the AOI."
        ),
    )
    max_jobs: int = Field(
        3,
        ge=1,
        description="Concurrent download jobs.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators (mirrors BurstSearch shape)
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

    @model_validator(mode="after")
    def _check_aoi_and_dates(self) -> "OperaCslcSearch":
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
        assert self.bbox is not None
        return box(*self.bbox)

    @property
    def static_layers_dir(self) -> Path:
        """Return the directory where CSLC-STATIC HDF5s live."""
        return self.out_dir / "static_layers"

    def summary(self) -> str:
        """Return a human-readable summary of the planned search."""
        return (
            "OperaCslcSearch:\n"
            f"  AOI bounds : {self.aoi.bounds}\n"
            f"  Dates      : {self.start.date()} -> {self.end.date()}\n"
            f"  Track      : {self.track}\n"
            f"  Burst IDs  : {self.burst_ids or 'auto (from AOI)'}\n"
            f"  Output     : {self.out_dir}"
        )

    def _resolve_burst_ids(self) -> list[str]:
        """Get the list of OPERA burst IDs covering the AOI.

        If the user supplied burst_ids explicitly, use them. Otherwise query
        ASF with the AOI + track to discover them. Querying without an
        explicit list returns one result *per acquisition*, so we
        deduplicate to unique burst IDs before passing to download_cslcs
        (which expects burst IDs and applies the date filter itself).
        """
        if self.burst_ids:
            return list(self.burst_ids)
        from opera_utils.download import search_cslcs

        bounds: tuple[float, float, float, float] = tuple(self.aoi.bounds)  # type: ignore[assignment]
        results = search_cslcs(
            start=self.start,
            end=self.end,
            bounds=bounds,
            track=self.track,
        )
        seen: dict[str, None] = {}
        for r in results:  # type: ignore[union-attr]
            props = getattr(r, "properties", {})
            bid = props.get("operaBurstID") or props.get("burstID")
            if bid:
                seen[bid.lower().replace("-", "_")] = None
        burst_ids = sorted(seen)
        if not burst_ids:
            msg = (
                "No OPERA CSLCs found for the requested AOI / track / dates."
                " Coverage may be missing — fall back to BurstSearch + COMPASS."
            )
            raise RuntimeError(msg)
        logger.info(f"Resolved {len(burst_ids)} OPERA burst IDs from ASF")
        return burst_ids

    @log_runtime
    def download(self) -> list[Path]:
        """Download OPERA CSLC HDF5 files into `out_dir`.

        Returns
        -------
        list[Path]
            Paths to the downloaded ``.h5`` files (one per burst per date).

        """
        from opera_utils.download import download_cslcs

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(self.summary())

        burst_ids = self._resolve_burst_ids()
        files = download_cslcs(
            burst_ids=burst_ids,
            output_dir=self.out_dir,
            start=self.start,
            end=self.end,
            max_jobs=self.max_jobs,
        )
        files = sorted(Path(f) for f in files)
        logger.info(f"Downloaded {len(files)} OPERA CSLC files to {self.out_dir}")
        return files

    @log_runtime
    def download_static_layers(self) -> list[Path]:
        """Download the CSLC-STATIC HDF5 files for the resolved burst IDs."""
        from opera_utils.download import download_cslc_static_layers

        self.static_layers_dir.mkdir(parents=True, exist_ok=True)
        burst_ids = self._resolve_burst_ids()
        files = download_cslc_static_layers(
            burst_ids=burst_ids,
            output_dir=self.static_layers_dir,
            max_jobs=self.max_jobs,
        )
        files = sorted(Path(f) for f in files)
        logger.info(
            f"Downloaded {len(files)} CSLC-STATIC files to {self.static_layers_dir}"
        )
        return files

    def existing_cslcs(self) -> list[Path]:
        """Return any OPERA CSLC HDF5s already present in `out_dir`."""
        return sorted(self.out_dir.glob("OPERA_L2_CSLC-S1_*.h5"))

    def existing_static_layers(self) -> list[Path]:
        """Return any CSLC-STATIC HDF5s already present in `static_layers_dir`."""
        return sorted(self.static_layers_dir.glob("OPERA_L2_CSLC-S1-STATIC_*.h5"))

    # Mirror BurstSearch.existing_safes for symmetry — used by Workflow to
    # check whether the download step can be skipped.
    def existing_files(self) -> list[Path]:
        return self.existing_cslcs()


# ----------------------------------------------------------------------------
# NISAR GSLC source
# ----------------------------------------------------------------------------


class NisarGslcSearch(YamlModel):
    """Pre-made NISAR GSLC search/download configuration.

    Wraps :func:`opera_utils.nisar.run_download` to search CMR for NISAR
    GSLC products covering the AOI + date range, fetch the matching HDF5s,
    and optionally subset each one to the AOI in a single pass. NISAR
    GSLCs are already geocoded (UTM projection) and have no separate
    "static layers" product, so the downstream workflow skips both COMPASS
    and the geometry stitching step.
    """

    kind: Literal["nisar-gslc"] = Field(
        default="nisar-gslc",
        description="Discriminator for the source type. Always `nisar-gslc`.",
    )
    out_dir: Path = Field(
        Path("data"),
        description="Directory where the NISAR GSLC HDF5s will be written.",
        validate_default=True,
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest as (left, bottom, right, top) in decimal degrees."
            " Used for both the CMR query and the bbox subset. Either `bbox`"
            " or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a WKT polygon string (or path to a `.wkt`"
            " file). Converted to a bbox by `run_download`."
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
        alias="relative_orbit_number",
        description=(
            "NISAR relative orbit / track number — the `Track` field on ASF"
            " Vertex, the `RRR` digits in the granule filename. Constant"
            " across repeat passes. Combined with `frame` it pins a single"
            " repeat-pass stack."
        ),
    )
    frame: Optional[int] = Field(
        None,
        alias="track_frame_number",
        description=(
            "NISAR track-frame number — the `Frame` field on ASF Vertex, the"
            " `TTT` digits in the granule filename (e.g. `71`). Constant"
            " across repeat passes."
        ),
    )
    frequency: Optional[Literal["A", "B"]] = Field(
        default=None,
        description=(
            "NISAR frequency band: `A` (L-band primary) or `B`. If left as"
            " the default (None), sweets peeks at the first matching CMR"
            " hit and uses whichever frequency is actually present in the"
            " HDF5. Different NISAR product releases ship different bands"
            " (early BETA was A; recent PR products are B), so guessing is"
            " usually wrong."
        ),
    )
    polarizations: Optional[list[str]] = Field(
        None,
        description=(
            "Polarizations to keep (e.g. ['HH']). If left as the default"
            " (None), sweets uses every polarization present under the"
            " resolved frequency in the first matching CMR hit."
        ),
    )
    short_name: str = Field(
        default="NISAR_L2_GSLC_BETA_V1",
        description="CMR collection short-name to query.",
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Concurrent download jobs.",
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

    @field_validator("polarizations")
    @classmethod
    def _upper_pols(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return [p.upper() for p in v] if v else v

    @model_validator(mode="after")
    def _check_aoi_and_dates(self) -> "NisarGslcSearch":
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
        if self.wkt:
            return shp_wkt.loads(self.wkt)
        assert self.bbox is not None
        return box(*self.bbox)

    def hdf5_subdataset(self) -> str:
        """Return the dolphin `input_options.subdataset` path for this config.

        NISAR GSLCs put the complex data at
        ``/science/LSAR/GSLC/grids/frequency{A,B}/{POL}``. If `frequency`
        and `polarizations` are unset, this peeks at the first cached HDF5
        in `out_dir` (or, if there isn't one yet, the first matching CMR
        hit) to learn what's actually in the product.
        """
        freq, pols = self._resolve_frequency_and_pols()
        return f"/science/LSAR/GSLC/grids/frequency{freq}/{pols[0]}"

    def summary(self) -> str:
        return (
            "NisarGslcSearch:\n"
            f"  AOI bounds       : {self.aoi.bounds}\n"
            f"  Dates            : {self.start.date()} -> {self.end.date()}\n"
            f"  Track            : {self.track or 'any'}\n"
            f"  Frame            : {self.frame or 'any'}\n"
            f"  Frequency        : {self.frequency or 'auto'}\n"
            f"  Polarizations    : {self.polarizations or 'auto'}\n"
            f"  CMR short_name   : {self.short_name}\n"
            f"  Output           : {self.out_dir}"
        )

    def _resolve_frequency_and_pols(self) -> tuple[str, list[str]]:
        """Pick the actual `frequency` + polarizations to feed dolphin / run_download.

        Order of preference:

        1. Already-downloaded HDF5 in ``out_dir`` — peek inside.
        2. First CMR hit — open it remotely.

        Returns the user's overrides where they make sense (e.g. they
        asked for `HH` and the file does have it), otherwise falls back
        to whichever frequency / polarization is actually present in
        the HDF5. Logs a warning when the user-requested values don't
        match what's available.
        """
        local = self.existing_files()
        if local:
            freq, pols = _peek_nisar_grid(local[0])
        else:
            from opera_utils.nisar import search

            results = search(
                bbox=tuple(self.aoi.bounds),  # type: ignore[arg-type]
                relative_orbit_number=self.track,
                track_frame_number=self.frame,
                start_datetime=self.start,
                end_datetime=self.end,
                short_name=self.short_name,
            )
            if not results:
                msg = (
                    "No NISAR GSLC products found for the requested AOI /"
                    " track / frame / dates. Cannot resolve frequency."
                )
                raise RuntimeError(msg)
            with results[0]._open() as hf:
                freq, pols = _peek_nisar_grid_from_handle(hf)
        return self._reconcile(freq, pols)

    def _reconcile(
        self, available_freq: str, available_pols: list[str]
    ) -> tuple[str, list[str]]:
        """Reconcile user request against what's actually in the file."""
        if self.frequency and self.frequency != available_freq:
            logger.warning(
                f"NISAR: requested frequency={self.frequency!r} but the"
                f" product only has frequency{available_freq!r}; using"
                f" frequency{available_freq!r}."
            )
        freq = available_freq
        if self.polarizations:
            kept = [p for p in self.polarizations if p in available_pols]
            dropped = [p for p in self.polarizations if p not in available_pols]
            if dropped:
                logger.warning(
                    f"NISAR: requested polarizations {dropped} not present"
                    f" in product (available: {available_pols}); using"
                    f" {kept or available_pols} instead."
                )
            pols = kept or available_pols
        else:
            pols = available_pols
        return freq, pols

    @log_runtime
    def download(self) -> list[Path]:
        """Search + download + bbox-subset NISAR GSLC HDF5s into `out_dir`."""
        from opera_utils.nisar import run_download

        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(self.summary())

        # Peek at the first matching product to learn the actual frequency
        # + polarizations the file carries. Different NISAR releases pack
        # different bands and pols, and run_download crashes if you ask
        # for the wrong frequency.
        freq, pols = self._resolve_frequency_and_pols()
        logger.info(f"NISAR resolved: frequency={freq}, polarizations={pols}")

        bounds: tuple[float, float, float, float] = tuple(self.aoi.bounds)  # type: ignore[assignment]
        result = run_download(
            bbox=bounds,
            relative_orbit_number=self.track,
            track_frame_number=self.frame,
            start_datetime=self.start,
            end_datetime=self.end,
            frequency=freq,
            polarizations=pols,
            short_name=self.short_name,
            num_workers=self.num_workers,
            output_dir=self.out_dir,
        )
        files = sorted(Path(p) for p in result)
        logger.info(f"Downloaded {len(files)} NISAR GSLC files to {self.out_dir}")
        return files

    def existing_files(self) -> list[Path]:
        """Return any NISAR GSLC HDF5s already present in `out_dir`."""
        return sorted(self.out_dir.glob("NISAR_L2_*GSLC*.h5"))


def _peek_nisar_grid(h5path: Path) -> tuple[str, list[str]]:
    """Open a NISAR GSLC HDF5 and return (frequency_letter, polarizations)."""
    import h5py

    with h5py.File(h5path, "r") as hf:
        return _peek_nisar_grid_from_handle(hf)


def _peek_nisar_grid_from_handle(hf) -> tuple[str, list[str]]:  # noqa: ANN001
    """Inspect an open NISAR GSLC HDF5 file handle for grid layout."""
    grids_path = "/science/LSAR/GSLC/grids"
    if grids_path not in hf:
        msg = f"NISAR HDF5 has no `{grids_path}` group"
        raise RuntimeError(msg)
    freq_groups = [k for k in hf[grids_path].keys() if k.startswith("frequency")]
    if not freq_groups:
        msg = f"NISAR HDF5 `{grids_path}` has no frequency subgroup"
        raise RuntimeError(msg)
    # Pick the first available frequency (filename order: A before B).
    freq_groups.sort()
    freq_path = f"{grids_path}/{freq_groups[0]}"
    freq_letter = freq_groups[0].removeprefix("frequency")
    pols = [
        k
        for k in hf[freq_path].keys()
        if k in ("HH", "VV", "HV", "VH", "RH", "RV", "LH", "LV")
    ]
    return freq_letter, pols
