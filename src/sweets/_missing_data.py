from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import reduce
from math import nan
from pathlib import Path
from typing import Optional, Sequence, Union

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from dolphin._types import Filename
from osgeo import gdal
from shapely import geometry, intersection_all, union_all, wkt

from sweets._log import get_log

logger = get_log(__name__)


def get_geodataframe(
    gslc_files: Sequence[Filename], max_workers: int = 5
) -> gpd.GeoDataFrame:
    """Get a GeoDataFrame of the CSLC footprints.

    Parameters
    ----------
    gslc_files : list[Filename]
        List of CSLC files.
    max_workers : int
        Number of threads to use.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as e:
        polygons = list(e.map(get_cslc_polygon, gslc_files))

    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    gdf["count"] = 1
    gdf["filename"] = [p.stem for p in gslc_files]
    gdf["date"] = pd.to_datetime(gdf.filename.str.split("_").str[3])
    gdf["burst_id"] = gdf.filename.str[:15]
    return gdf


def get_cslc_polygon(
    opera_file: Filename, buffer_degrees: float = 0.0
) -> Union[geometry.Polygon, None]:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0
    """
    dset_name = "/identification/bounding_polygon"
    with h5py.File(opera_file) as hf:
        if dset_name not in hf:
            logger.debug(f"Could not find {dset_name} in {opera_file}")
            return None
        wkt_str = hf[dset_name][()].decode("utf-8")
    return wkt.loads(wkt_str).buffer(buffer_degrees)


def get_common_dates(
    *, gslc_files: Optional[Sequence[Filename]] = None, gdf=None
) -> list[str]:
    """Get the date common to all GSLCs."""
    if gdf is None:
        if gslc_files is None:
            raise ValueError("Need `gdf` or `gslc_files`")
        gdf = get_geodataframe(gslc_files)

    grouped_by_burst = _get_per_burst_df(gdf)
    common_dates = list(
        reduce(
            lambda x, y: x.intersection(set(y)),  # type: ignore
            grouped_by_burst.date[1:],
            set(grouped_by_burst.date[0]),
        )
    )
    return pd.Series(common_dates).dt.strftime("%Y%m%d").tolist()


def _filter_gslcs_by_common_dates(gslc_files: list[Filename]) -> list[Path]:
    common_datestrs = get_common_dates(gslc_files=gslc_files)
    return [
        Path(p) for p in gslc_files if any(d in Path(p).stem for d in common_datestrs)
    ]


def _get_per_burst_df(
    gdf: gpd.GeoDataFrame, how: str = "intersection"
) -> gpd.GeoDataFrame:
    func = union_all if how == "union" else intersection_all
    grouped = gpd.GeoDataFrame(
        gdf.groupby("burst_id").agg({"count": "sum", "date": list, "geometry": func})
    ).reset_index()
    grouped = grouped.set_crs(epsg=4326)
    return grouped


def plot_count_per_burst(
    *,
    gdf: Optional[gpd.GeoDataFrame] = None,
    gslc_files: Optional[Sequence[Filename]] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot the number of GSLC files found per burst."""
    if gdf is None:
        if gslc_files is None:
            raise ValueError("Need `gdf` or `gslc_files`")
        gdf = get_geodataframe(gslc_files)
    gdf_grouped = _get_per_burst_df(gdf)

    if ax is None:
        fig, ax = plt.subplots(ncols=1)
    kwds = dict(
        column="count",
        legend=True,
        cmap="tab10",
        legend_kwds={"label": "Count", "orientation": "horizontal"},
        linewidth=0.8,
        edgecolor="0.8",
    )

    gdf_grouped.plot(ax=ax, **kwds)
    return gdf_grouped


def plot_mini_timeseries(
    ts_file: Filename,
    sub: int = 20,
    ncols: int = 10,
    vm: float = 5,
    save_to: str = "mini_timeseries",
):
    """Plot a mintpy timeseries file heavily subsampled for a quick check."""
    import h5py
    import numpy as np

    with h5py.File(ts_file, "a") as hf:
        dates = np.array(hf["date"]).astype(str)
        if save_to in hf:
            ts = hf[save_to][()]
        else:
            ts = hf["timeseries"][:, ::sub, ::sub]
            if save_to:
                hf[save_to] = ts

    ntotal = len(ts)
    nrows = ntotal // ncols if (ntotal % ncols == 0) else ntotal // ncols + 1
    figsize = (ncols * 0.6, nrows * 0.6)

    try:
        import proplot as plt

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, refwidth=0.5)
    except ImportError:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i in range(nrows):
        for j in range(ncols):
            idx = j * ncols + i
            if idx >= len(ts):
                break
            ax = axes[i, j]
            axim = ax.imshow(100 * ts[idx], vmax=vm, vmin=-vm)
            fig.colorbar(axim, ax=ax)
            ax.set_title(str(idx) + ":" + str(dates[idx]))

    return fig, axes


@dataclass
class Stats:
    """Class holding the raster stats returned by `gdalinfo`."""

    min: float
    max: float
    mean: float
    stddev: float
    pct_valid: float


def get_raster_stats(filename: Filename, band: int = 1) -> Stats:
    """Get the (Min, Max, Mean, StdDev, pct_valid) of the 1-band file."""
    try:
        ii = gdal.Info(os.fspath(filename), stats=True, format="json")
    except RuntimeError as e:
        if "No such file or directory" in e.args[0]:
            raise
        elif "no valid pixels found" in e.args[0]:
            return Stats(nan, nan, nan, nan, 0.0)

    s = ii["bands"][band - 1]["metadata"][""]
    return Stats(
        float(s["STATISTICS_MAXIMUM"]),
        float(s["STATISTICS_MINIMUM"]),
        float(s["STATISTICS_MEAN"]),
        float(s["STATISTICS_STDDEV"]),
        float(s["STATISTICS_VALID_PERCENT"]),
    )


def is_valid(filename: Filename) -> tuple[bool, str]:
    """Check if GDAL can open the file and if there are any valid pixels.

    Parameters
    ----------
    filename : Filename
        Path to file.

    Returns
    -------
    bool:
        False if bad file, or no valid pixels.
    str:
        Reason behind a `False` value given by GDAL.
    """
    try:
        get_raster_stats(filename)
    except RuntimeError as e:
        return False, str(e)
    return True, ""


def get_bad_files(
    path: Filename,
    ext: str = ".int",
    pct_valid_threshold: float = 60,
    max_jobs: int = 20,
) -> tuple[list[Path], list[Stats]]:
    """Check all files in `path` for (partially) invalid rasters."""
    files = sorted(Path(path).glob(f"*{ext}"))
    logger.info(f"Searching {len(files)} in {path} with extension {ext}")

    with ThreadPoolExecutor(max_jobs) as exc:
        stats = list(exc.map(get_raster_stats, files))

    bad_files = [
        (f, s) for (f, s) in zip(files, stats) if s.pct_valid < pct_valid_threshold
    ]
    if len(bad_files) == 0:
        return [], []
    out_files, out_stats = list(zip(*bad_files))
    return out_files, out_stats  # type: ignore


def remove_invalid_ifgs(bad_files: list[Filename]):
    """Remove invalid ifgs and their unwrapped counterparts."""
    for ff in bad_files:
        u = str(ff).replace("stitched", "unwrapped").replace(".int", ".unw")
        try:
            Path(u).unlink()
        except FileNotFoundError:
            continue
        Path(ff).unlink()
