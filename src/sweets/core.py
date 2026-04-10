"""End-to-end Sentinel-1 InSAR workflow.

Three stages:

1. **Download** the bursts that cover the AOI (small bbox), using burst2safe.
2. **Geocode** each burst into an OPERA-style geocoded SLC, using COMPASS.
3. **Run dolphin** to phase-link, form interferograms, stitch, unwrap, and
   invert a displacement timeseries.

The workflow is defined as a single :class:`Workflow` Pydantic model that
can be serialized to / loaded from a ``sweets_config.yaml``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from dolphin.utils import set_num_threads
from dolphin.workflows.config import YamlModel
from opera_utils import group_by_burst
from pydantic import ConfigDict, Field, computed_field, field_validator, model_validator
from shapely import geometry, wkt as shp_wkt

from ._burst_db import get_burst_db
from ._dolphin import DolphinOptions, run_displacement
from ._geocode_slcs import create_config_files, run_geocode, run_static_layers
from ._geometry import stitch_geometry
from ._log import get_log, log_runtime
from ._netrc import setup_nasa_netrc
from ._orbit import download_orbits
from ._types import Filename
from .dem import create_dem, create_water_mask
from .download import BurstSearch

if TYPE_CHECKING:
    from dolphin.workflows.displacement import OutputPaths

logger = get_log(__name__)


class Workflow(YamlModel):
    """End-to-end Sentinel-1 InSAR workflow configuration."""

    work_dir: Path = Field(
        default_factory=Path.cwd,
        description="Root of working directory for processing.",
        validate_default=True,
    )

    bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description=(
            "AOI as (left, bottom, right, top) in decimal degrees. Either"
            " `bbox` or `wkt` must be set."
        ),
    )
    wkt: Optional[str] = Field(
        default=None,
        description="AOI as a WKT polygon (or path to a `.wkt` file). Overrides bbox.",
    )

    search: BurstSearch = Field(
        ...,
        description="Burst search / download configuration.",
    )

    dem_filename: Path = Field(
        default_factory=lambda data: data["work_dir"] / "dem.tif",
        description=(
            "DEM in EPSG:4326. If left as the default, sweets downloads a"
            " Copernicus DEM via sardem."
        ),
    )
    water_mask_filename: Path = Field(
        default_factory=lambda data: data["work_dir"] / "watermask.tif",
        description=(
            "Water mask in EPSG:4326 (uint8 GTiff, 1=land, 0=water). If left"
            " as the default, sweets derives one from a Copernicus DEM."
        ),
    )
    orbit_dir: Path = Field(
        default=Path("orbits"),
        description="Directory for Sentinel-1 precise orbit files.",
        validate_default=True,
    )

    slc_posting: tuple[float, float] = Field(
        default=(10, 5),
        description="Geocoded SLC posting (y, x) in meters.",
    )
    pol_type: Literal["co-pol", "cross-pol"] = Field(
        default="co-pol",
        description="Polarization type to geocode (COMPASS knob).",
    )

    dolphin: DolphinOptions = Field(
        default_factory=DolphinOptions,
        description="Configuration for the dolphin displacement workflow.",
    )

    n_workers: int = Field(
        default=4,
        description="Process pool size for COMPASS geocoding.",
        ge=1,
    )
    threads_per_worker: int = Field(
        default=8,
        description="OMP_NUM_THREADS for each geocoding worker.",
        ge=1,
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing intermediate / output files.",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("wkt", mode="before")
    @classmethod
    def _check_file_and_parse_wkt(cls, v):
        if v is None:
            return v
        if Path(v).exists():
            v = Path(v).read_text().strip()
        try:
            shp_wkt.loads(v)
        except Exception as e:
            msg = f"Invalid WKT string: {e}"
            raise ValueError(msg) from e
        return v

    @field_validator("work_dir", "orbit_dir")
    @classmethod
    def _expand_dirs(cls, v):
        return Path(v).expanduser().resolve()

    @model_validator(mode="before")
    @classmethod
    def _sync_aoi(cls, values: Any) -> Any:
        """Push the top-level bbox/wkt down into the BurstSearch.

        The outer ``Workflow.bbox`` / ``Workflow.wkt`` are the canonical AOI;
        the nested ``search`` field gets the same bbox so burst2safe knows
        what to download. Only ``bbox`` is forced — ``search.wkt`` is left
        alone so non-rectangular search polygons can be specified at the
        BurstSearch level if a user wants.
        """
        if not isinstance(values, dict):
            return values
        if "search" not in values:
            values["search"] = {}
        elif isinstance(values["search"], BurstSearch):
            values["search"] = values["search"].model_dump(
                exclude_unset=True, by_alias=True
            )
        outer_bbox = values.get("bbox")
        outer_wkt = values.get("wkt")
        inner = values["search"]
        inner_bbox = inner.get("bbox")
        inner_wkt = inner.get("wkt")
        bbox = outer_bbox or inner_bbox
        wkt_value = outer_wkt or inner_wkt
        if not bbox and not wkt_value:
            msg = "Must specify `bbox` or `wkt` (on Workflow or `search`)"
            raise ValueError(msg)
        if bbox is not None:
            values["bbox"] = bbox
            if not inner_bbox:
                inner["bbox"] = bbox
        if outer_wkt is not None:
            values["wkt"] = outer_wkt
        return values

    @model_validator(mode="after")
    def _set_bbox_and_wkt(self) -> "Workflow":
        if self.bbox is None and self.wkt is not None:
            self.bbox = shp_wkt.loads(self.wkt).bounds
        if self.bbox is not None and self.wkt is None:
            left, bottom, right, top = self.bbox
            self.wkt = shp_wkt.dumps(geometry.box(left, bottom, right, top))
        assert self.bbox is not None
        if self.bbox[1] > self.bbox[3]:
            msg = f"Latitude min must be lower than max, got {self.bbox}"
            raise ValueError(msg)
        if self.bbox[0] > self.bbox[2]:
            msg = f"Longitude min must be lower than max, got {self.bbox}"
            raise ValueError(msg)
        return self

    # ------------------------------------------------------------------
    # Computed paths
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def log_dir(self) -> Path:
        return self.work_dir / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gslc_dir(self) -> Path:
        return self.work_dir / "gslcs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def geom_dir(self) -> Path:
        return self.work_dir / "geometry"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dolphin_dir(self) -> Path:
        return self.work_dir / "dolphin"

    @property
    def _dem_bbox(self) -> tuple[float, float, float, float]:
        assert self.bbox is not None
        return (
            self.bbox[0] - 0.25,
            self.bbox[1] - 0.25,
            self.bbox[2] + 0.25,
            self.bbox[3] + 0.25,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, config_file: Filename = "sweets_config.yaml") -> None:
        """Save this configuration to a YAML file."""
        logger.info(f"Saving config to {config_file}")
        self.to_yaml(config_file)

    @classmethod
    def load(cls, config_file: Filename = "sweets_config.yaml") -> "Workflow":
        """Load a configuration from a YAML file."""
        logger.info(f"Loading config from {config_file}")
        return cls.from_yaml(config_file)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _existing_safes(self) -> list[Path]:
        return self.search.existing_safes()

    # COMPASS-written CSLC HDF5s are tens to hundreds of MB. A 6-KB shell is
    # a leftover from a crashed run that wrote the attribute scaffolding but
    # never the data — accepting those silently breaks dolphin downstream
    # (issue #107). Treat anything below this size as not-yet-produced.
    _MIN_VALID_GSLC_BYTES = 1 * 1024 * 1024

    def _existing_gslcs(self) -> list[Path]:
        return [
            p
            for p in sorted(self.gslc_dir.glob("t*/*/t*.h5"))
            if not p.name.startswith("static_")
            and p.stat().st_size >= self._MIN_VALID_GSLC_BYTES
        ]

    def _existing_static_layers(self) -> list[Path]:
        return sorted(self.gslc_dir.glob("t*/*/static_*.h5"))

    @log_runtime
    def _download(self) -> list[Path]:
        existing = self._existing_safes()
        if existing and not self.overwrite:
            logger.info(
                f"Found {len(existing)} existing SAFE dirs in"
                f" {self.search.out_dir}; skipping burst2safe download."
            )
            return existing
        return self.search.download()

    @log_runtime
    def _geocode_slcs(
        self, safes: list[Path], dem_file: Path, burst_db_file: Path
    ) -> tuple[list[Path], list[Path]]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        compass_cfg_files = create_config_files(
            slc_dir=safes[0].parent,
            burst_db_file=burst_db_file,
            dem_file=dem_file,
            orbit_dir=self.orbit_dir,
            bbox=self.bbox,
            y_posting=self.slc_posting[0],
            x_posting=self.slc_posting[1],
            pol_type=self.pol_type,
            out_dir=self.gslc_dir,
            overwrite=self.overwrite,
            using_zipped=False,
        )

        existing = {p.name: p for p in self._existing_gslcs()}
        logger.info(f"Found {len(existing)} existing GSLCs")
        gslc_files: list[Path] = []
        todo: list[Path] = []
        for cfg in compass_cfg_files:
            name = _cfg_to_filename(cfg)
            if name in existing:
                gslc_files.append(existing[name])
            else:
                todo.append(cfg)

        if todo:
            run = partial(run_geocode, log_dir=self.log_dir)
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                gslc_files.extend(pool.map(run, todo))

        # Static layers (one per burst, not per date)
        static_existing = {p.name: p for p in self._existing_static_layers()}
        first_per_burst = [
            cfgs[0] for cfgs in group_by_burst(compass_cfg_files).values()
        ]
        static_files: list[Path] = []
        static_todo: list[Path] = []
        for cfg in first_per_burst:
            name = _cfg_to_static_filename(cfg)
            if name in static_existing:
                static_files.append(static_existing[name])
            else:
                static_todo.append(cfg)

        if static_todo:
            run_sl = partial(run_static_layers, log_dir=self.log_dir)
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                static_files.extend(pool.map(run_sl, static_todo))

        return sorted(gslc_files), sorted(static_files)

    @log_runtime
    def _stitch_geometry(self, static_files: list[Path]) -> list[Path]:
        from dolphin._types import Bbox

        bbox = Bbox(*self.bbox) if self.bbox is not None else None
        return stitch_geometry(
            geom_path_list=[Path(p) for p in static_files],
            geom_dir=self.geom_dir,
            dem_filename=self.dem_filename,
            looks=self.dolphin.strides,
            bbox=bbox,
            overwrite=self.overwrite,
        )

    @log_runtime
    def _run_dolphin(self, gslc_files: list[Path]) -> "OutputPaths":
        mask = self.water_mask_filename if self.water_mask_filename.exists() else None
        return run_displacement(
            cslc_files=gslc_files,
            work_directory=self.dolphin_dir,
            options=self.dolphin,
            mask_file=mask,
            bounds=self.bbox,
            config_yaml=self.work_dir / "dolphin_config.yaml",
        )

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    @log_runtime
    def run(self, starting_step: int = 1) -> "OutputPaths":
        """Run the full workflow.

        Parameters
        ----------
        starting_step : int
            Skip earlier stages if intermediate outputs are already on disk.
            ``1`` = download, ``2`` = geocode, ``3`` = dolphin.

        Returns
        -------
        dolphin.workflows.displacement.OutputPaths
            Output paths produced by dolphin.

        """
        setup_nasa_netrc()
        set_num_threads(self.threads_per_worker)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if starting_step <= 1:
            with ThreadPoolExecutor(max_workers=4) as pool:
                dem_fut = pool.submit(create_dem, self.dem_filename, self._dem_bbox)
                mask_fut = pool.submit(
                    create_water_mask, self.water_mask_filename, self._dem_bbox
                )
                burst_db_fut = pool.submit(get_burst_db)
                wait([dem_fut, mask_fut, burst_db_fut])
                dem_fut.result()
                mask_fut.result()
                burst_db_file = burst_db_fut.result()
            self._download()
        else:
            burst_db_file = get_burst_db()

        if starting_step <= 2:
            safes = self._existing_safes()
            if not safes:
                msg = (
                    f"No SAFE directories found in {self.search.out_dir};"
                    " cannot geocode."
                )
                raise RuntimeError(msg)
            download_orbits(self.search.out_dir, self.orbit_dir)
            _, static_files = self._geocode_slcs(
                safes, self.dem_filename, burst_db_file
            )
            self._stitch_geometry(static_files)

        # Always re-collect GSLCs from disk before dolphin so a starting_step=3
        # run still finds them.
        gslc_files = self._existing_gslcs()
        logger.info(f"Found {len(gslc_files)} GSLC files for dolphin")
        if not gslc_files:
            msg = f"No GSLCs found in {self.gslc_dir}; cannot run dolphin."
            raise RuntimeError(msg)
        return self._run_dolphin(gslc_files)


def _cfg_to_filename(cfg_path: Path) -> str:
    """COMPASS runconfig path -> expected GSLC HDF5 filename.

    e.g. ``geo_runconfig_20221029_t078_165578_iw3.yaml``
        -> ``t078_165578_iw3_20221029.h5``
    """
    date = cfg_path.name.split("_")[2]
    burst = "_".join(cfg_path.stem.split("_")[3:])
    return f"{burst}_{date}.h5"


def _cfg_to_static_filename(cfg_path: Path) -> str:
    """COMPASS runconfig path -> expected static-layers HDF5 filename."""
    burst = "_".join(cfg_path.stem.split("_")[3:])
    return f"static_layers_{burst}.h5"
