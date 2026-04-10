"""Tropospheric correction step using OPERA L4 TROPO-ZENITH products.

Wraps :func:`opera_utils.tropo.create_tropo_corrections_for_stack` (the
high-level workflow on `scottstanie/opera-utils@develop-scott`) and applies
the resulting per-date LOS-projected delays to dolphin's unwrapped phase
outputs.

The flow is:

1. Register an :class:`OperaCslcReader` so opera_utils' SLCReader registry
   knows how to parse OPERA CSLC HDF5s for ``datetime`` / ``bounds``.
2. Call ``create_tropo_corrections_for_stack`` with the workflow's CSLC
   stack, the (already-stitched) DEM and the (already-stitched) local
   incidence angle raster, producing one ``tropo_correction_<dt>.tif`` per
   acquisition referenced to the first date.
3. For each unwrapped interferogram pair from dolphin, look up the two
   tropo files, compute the differential, convert metres of LOS delay to
   radians of phase via ``4 * pi / wavelength``, and subtract from the
   unwrapped phase. Write the corrected raster alongside the originals.

The OPERA CSLC reader is registered eagerly when this module is imported,
so any sweets code path that triggers tropo gets the backend automatically.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np
import rasterio
import rioxarray as rxr
from pydantic import BaseModel, Field
from shapely import wkt as shp_wkt

from ._log import get_log, log_runtime

if TYPE_CHECKING:
    pass

logger = get_log(__name__)

# Sentinel-1 C-band carrier wavelength used by OPERA CSLCs and burst2safe
# SAFEs alike.
S1_WAVELENGTH_M = 0.05546576

# OPERA tropo correction filenames look like
# `tropo_correction_20210606T005125.tif`. The reference (first) date is
# saved as `reference_<dt>.tif`.
_TROPO_FILENAME_RE = re.compile(r"^(?:reference|tropo_correction)_(\d{8}T\d{6})\.tif$")


class TropoOptions(BaseModel):
    """Sweets-side configuration for the optional tropo correction step."""

    enabled: bool = Field(
        default=False,
        description=(
            "Whether to run the OPERA L4 TROPO-ZENITH correction after"
            " dolphin produces unwrapped interferograms."
        ),
    )
    height_max: float = Field(
        default=10000.0,
        description="Max DEM height (m) included when cropping the tropo cube.",
    )
    margin_deg: float = Field(
        default=0.3,
        description="Padding (degrees) added around the AOI when cropping.",
    )
    interp_method: str = Field(
        default="linear",
        description="Interpolation method passed through to apply_tropo.",
    )
    num_workers: int = Field(
        default=2,
        ge=1,
        description="Parallel workers for crop_tropo / apply_tropo.",
    )


# ----------------------------------------------------------------------------
# OPERA CSLC SLCReader implementation
# ----------------------------------------------------------------------------


class OperaCslcReader:
    """SLCReader implementation for OPERA CSLC HDF5 files.

    The CSLC HDF5 layout (as written by COMPASS / OPERA) puts identification
    metadata under `/identification/` — `zero_doppler_start_time`,
    `bounding_polygon` and the per-burst incidence angles live in the static
    layers file rather than the per-date CSLC, but for the tropo workflow we
    only need the datetime and bounds, both of which are in `/identification`.
    """

    @staticmethod
    def _identification(slc_file: Path) -> dict[str, object]:
        with h5py.File(slc_file, "r") as hf:
            ident = hf["/identification"]
            return {k: ident[k][()] for k in ident.keys()}

    def read_datetime(self, slc_file: Path) -> datetime:
        ident = self._identification(slc_file)
        raw = ident["zero_doppler_start_time"]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        # OPERA's HDF5 attribute is `YYYY-MM-DD HH:MM:SS.ffffff`
        return datetime.fromisoformat(str(raw).strip())

    def read_bounds(self, slc_file: Path) -> tuple[float, float, float, float]:
        ident = self._identification(slc_file)
        raw = ident["bounding_polygon"]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        poly = shp_wkt.loads(str(raw))
        west, south, east, north = poly.bounds
        return (west, south, east, north)

    def read_incidence_angle(self, slc_file: Path) -> float:
        # The per-burst incidence is in the static_layers file, not the
        # per-date CSLC. The tropo workflow always passes a separate
        # `incidence_angle` raster (the stitched local_incidence_angle.tif),
        # so this method is intentionally a no-op stub — extract_stack_info
        # never calls it.
        msg = (
            "OperaCslcReader does not provide per-file incidence angle;"
            " pass `incidence_angle_path` to apply_tropo / the workflow."
        )
        raise NotImplementedError(msg)


def _register_opera_cslc_reader() -> None:
    """Idempotently register the OPERA CSLC reader with opera_utils.tropo."""
    from opera_utils.tropo._slc_stack import _sensor_registry, register_sensor

    for name in ("opera-cslc", "sentinel1"):
        if name not in _sensor_registry:
            register_sensor(name, OperaCslcReader())


def _force_threaded_dns_resolver() -> None:
    """Sidestep aiohttp's c-ares resolver, which DNS-times out on some networks.

    Both burst2safe and the OPERA tropo `crop_tropo` open HTTPS URLs via
    aiohttp under the hood; aiohttp's default ``AsyncResolver`` (aiodns)
    fails with ``Timeout while contacting DNS servers`` on networks where
    c-ares can't reach a usable resolver. Falling back to the stdlib
    threaded resolver fixes it without changing behavior elsewhere.
    """
    import aiohttp.resolver

    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver


_register_opera_cslc_reader()
_force_threaded_dns_resolver()


# ----------------------------------------------------------------------------
# Workflow integration
# ----------------------------------------------------------------------------


@log_runtime
def create_tropo_corrections(
    slc_files: list[Path],
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    options: Optional[TropoOptions] = None,
) -> list[Path]:
    """Run the OPERA tropo workflow over a stack of SLC files.

    Parameters
    ----------
    slc_files
        Paths to the per-date CSLC HDF5s (or COMPASS-produced GSLC HDF5s).
    dem_path
        DEM raster used by apply_tropo to interpolate to surface heights.
    incidence_angle_path
        Per-pixel incidence-angle raster (produced by stitch_geometry).
    output_dir
        Directory where ``tropo_correction_<dt>.tif`` files will be written.
    options
        Sweets-side knobs (defaults if None).

    Returns
    -------
    list[Path]
        Sorted paths to the produced tropo correction GeoTIFFs.

    """
    from opera_utils.tropo import create_tropo_corrections_for_stack

    options = options or TropoOptions()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return create_tropo_corrections_for_stack(
        slc_files=list(slc_files),
        dem_path=Path(dem_path),
        output_dir=output_dir,
        sensor="opera-cslc",
        incidence_angle=Path(incidence_angle_path),
        margin_deg=options.margin_deg,
        height_max=options.height_max,
        subtract_first_date=True,
        num_workers=options.num_workers,
    )


def _parse_tropo_filename(p: Path) -> Optional[datetime]:
    """Return the datetime encoded in a tropo correction filename."""
    m = _TROPO_FILENAME_RE.match(p.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def _index_tropo_files_by_date(tropo_files: list[Path]) -> dict[str, Path]:
    """Build a `YYYYMMDD -> tropo_correction.tif` lookup."""
    out: dict[str, Path] = {}
    for p in tropo_files:
        dt = _parse_tropo_filename(p)
        if dt is None:
            continue
        out[dt.strftime("%Y%m%d")] = p
    return out


def _ifg_dates(ifg_path: Path) -> Optional[tuple[str, str]]:
    """Pull the (date1, date2) pair out of a `<date1>_<date2>.unw.tif` name."""
    m = re.match(r"(\d{8})_(\d{8})", ifg_path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


@log_runtime
def apply_tropo_to_unwrapped(
    unwrapped_files: list[Path],
    tropo_files: list[Path],
    output_dir: Path,
    wavelength: float = S1_WAVELENGTH_M,
) -> list[Path]:
    """Subtract differential tropo phase from each unwrapped interferogram.

    For an interferogram pair (date1, date2):

        corrected = unwrapped - (4*pi/wavelength) * (tropo[date2] - tropo[date1])

    The tropo correction GeoTIFFs are stored in metres of LOS displacement;
    multiply by ``4*pi/wavelength`` to convert to radians of phase. The
    sign matches dolphin's unwrap convention (positive = range increase).

    Parameters
    ----------
    unwrapped_files
        Paths to dolphin-produced ``<date1>_<date2>.unw.tif`` rasters.
    tropo_files
        Paths to ``tropo_correction_<dt>.tif`` rasters from
        :func:`create_tropo_corrections`.
    output_dir
        Where corrected unwrapped phase rasters will be written
        (filename suffix ``.tropo_corrected.unw.tif``).
    wavelength
        Radar carrier wavelength in metres. Defaults to S1 C-band.

    Returns
    -------
    list[Path]
        Paths to the written corrected rasters.

    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_date = _index_tropo_files_by_date(list(tropo_files))
    if not by_date:
        msg = f"No tropo correction files found in {tropo_files!r}"
        raise ValueError(msg)

    factor = 4.0 * np.pi / wavelength
    written: list[Path] = []
    for unw in sorted(unwrapped_files):
        pair = _ifg_dates(unw)
        if pair is None:
            logger.warning(f"Skipping {unw.name}: no date pair in filename")
            continue
        d1, d2 = pair
        if d1 not in by_date or d2 not in by_date:
            logger.warning(
                f"Skipping {unw.name}: no tropo for"
                f" {d1 if d1 not in by_date else d2}"
            )
            continue

        with rasterio.open(unw) as src:
            unw_arr = src.read(1)
            profile = src.profile.copy()

        # Reproject + resample tropo rasters to match the unwrapped grid.
        target = rxr.open_rasterio(unw, masked=True).squeeze(drop=True)
        t1 = rxr.open_rasterio(by_date[d1], masked=True).squeeze(drop=True)
        t2 = rxr.open_rasterio(by_date[d2], masked=True).squeeze(drop=True)
        t1_match = t1.rio.reproject_match(target)
        t2_match = t2.rio.reproject_match(target)
        diff_m = (t2_match - t1_match).values

        corrected = unw_arr.astype(np.float32) - (factor * diff_m).astype(np.float32)

        out = output_dir / f"{d1}_{d2}.tropo_corrected.unw.tif"
        profile.update(dtype="float32", count=1, compress="deflate")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(corrected, 1)
        written.append(out)
        logger.info(f"Wrote {out.name}")

    return written


@log_runtime
def run_tropo_correction(
    slc_files: list[Path],
    unwrapped_files: list[Path],
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    options: Optional[TropoOptions] = None,
    wavelength: float = S1_WAVELENGTH_M,
) -> list[Path]:
    """Build tropo corrections + subtract them from dolphin's unwrapped phases.

    See :func:`create_tropo_corrections` and :func:`apply_tropo_to_unwrapped`
    for the underlying steps.
    """
    options = options or TropoOptions()
    output_dir = Path(output_dir).resolve()
    tropo_corr_dir = output_dir / "tropo"
    corrected_dir = output_dir / "tropo_corrected"

    tropo_files = create_tropo_corrections(
        slc_files=slc_files,
        dem_path=dem_path,
        incidence_angle_path=incidence_angle_path,
        output_dir=tropo_corr_dir,
        options=options,
    )

    return apply_tropo_to_unwrapped(
        unwrapped_files=unwrapped_files,
        tropo_files=tropo_files,
        output_dir=corrected_dir,
        wavelength=wavelength,
    )
