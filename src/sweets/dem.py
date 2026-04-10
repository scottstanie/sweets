from __future__ import annotations

from os import fspath
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import sardem.dem

from sweets._log import get_log, log_runtime
from sweets._types import Filename
from sweets.utils import get_cache_dir

logger = get_log(__name__)


@log_runtime
def create_dem(output_name: Filename, bbox: Tuple[float, float, float, float]) -> Path:
    """Download a Copernicus Global DEM clipped to `bbox`."""
    output_name = Path(output_name).resolve()
    if output_name.exists():
        logger.info(f"DEM already exists: {output_name}")
        return output_name

    sardem.dem.main(
        output_name=fspath(output_name),
        bbox=bbox,
        data_source="COP",
        cache_dir=get_cache_dir(),
        output_format="GTiff",
        output_type="Float32",
    )
    return output_name


@log_runtime
def create_water_mask(
    output_name: Path, bbox: Tuple[float, float, float, float]
) -> Path:
    """Create a binary land(1) / water(0) mask from a Copernicus DEM.

    The legacy ``NASA_WATER`` SRTMSWBD path is broken on macOS (sardem's
    ``unzip_cmd.split(" ")`` chokes on the Application Support cache path)
    and is also restricted to ENVI output. Instead we lean on the same COP
    source already used by :func:`create_dem`: any COP pixel at or below
    sea level is treated as water. This is a coarse approximation — fine
    inland, less ideal for coastal AOIs — but it gives dolphin a usable
    mask without needing a second remote source.
    """
    output_name = Path(output_name).resolve()
    if output_name.exists():
        logger.info(f"Water mask already exists: {output_name}")
        return output_name

    dem_tmp = output_name.with_suffix(".dem.tmp.tif")
    sardem.dem.main(
        output_name=fspath(dem_tmp),
        bbox=bbox,
        data_source="COP",
        cache_dir=get_cache_dir(),
        output_format="GTiff",
        output_type="Float32",
    )

    with rasterio.open(dem_tmp) as src:
        heights = src.read(1)
        profile = src.profile.copy()

    mask = (heights > 0).astype(np.uint8)
    profile.update(dtype="uint8", nodata=0, count=1, compress="deflate")
    with rasterio.open(output_name, "w", **profile) as dst:
        dst.write(mask, 1)
    dem_tmp.unlink(missing_ok=True)
    return output_name
