#!/usr/bin/env python
"""ASF full download strategy using the existing ASF query logic.

Base taken from the original download.py module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlencode

import requests
from dateutil.parser import parse
from dolphin.workflows.config import YamlModel
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator
from shapely.geometry import box

from .._log import get_log, log_runtime
from .._types import Filename
from ._strategy import DownloadStrategy

logger = get_log(__name__)

DIRNAME = os.path.dirname(os.path.abspath(__file__))


class ASFFullDownload(DownloadStrategy, YamlModel):
    """Download strategy for full SAFE files from ASF."""

    method: Literal["full"] = "full"

    out_dir: Path = Field(
        Path(".") / "data",
        description="Output directory for downloaded files",
        validate_default=True,
    )
    bbox: Optional[tuple] = Field(
        None,
        description=(
            "lower left lon, lat, upper right format e.g."
            " bbox=(-150.2,65.0,-150.1,65.5)"
        ),
    )
    wkt: Optional[str] = Field(
        None,
        description="Well Known Text (WKT) string",
    )
    start: datetime = Field(
        ...,
        description=(
            "Starting time for search. Can be datetime or string (goes to"
            " `dateutil.parse`)"
        ),
    )
    end: datetime = Field(
        default_factory=datetime.now,
        description=(
            "Ending time for search. Can be datetime or string (goes to"
            " `dateutil.parse`)"
        ),
    )
    track: Optional[int] = Field(
        None,
        alias="relativeOrbit",
        description="Path number",
    )
    flight_direction: Optional[Literal["ASCENDING", "DESCENDING"]] = Field(
        None,
        alias="flightDirection",
        description="Direction of satellite during acquisition.",
    )
    frames: Optional[tuple[int, int]] = Field(
        None,
        description="(start, end) range of ASF frames.",
    )
    unzip: bool = Field(
        False,
        description="Unzip downloaded files into .SAFE directories",
    )
    _url: str = PrivateAttr()
    model_config = ConfigDict(extra="forbid")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _parse_date(cls, v):
        if isinstance(v, datetime):
            return v
        elif isinstance(v, date):
            # Convert to datetime
            return datetime.combine(v, datetime.min.time())
        return parse(v)

    @field_validator("out_dir")
    def _is_absolute(cls, v):
        return Path(v).resolve()

    @field_validator("flight_direction")
    @classmethod
    def _accept_prefixes(cls, v):
        if v is None:
            return v
        if v.lower().startswith("a"):
            return "ASCENDING"
        elif v.lower().startswith("d"):
            return "DESCENDING"

    @model_validator(mode="before")
    def _check_search_area(cls, values: Any):
        if isinstance(values, dict):
            if not values.get("wkt"):
                if values.get("bbox") is not None:
                    values["wkt"] = box(*values["bbox"]).wkt
                else:
                    raise ValueError("Must provide a bbox or wkt")

            elif Path(values["wkt"]).exists():
                values["wkt"] = Path(values["wkt"]).read_text().strip()

            # Check that end is after start
            if values.get("start") is not None and values.get("end") is not None:
                if values["end"] < values["start"]:
                    raise ValueError("End must be after start")
        return values

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Form the url for the ASF query.
        self._url = self._form_url()

    def _form_url(self) -> str:
        """Form the url for the ASF query."""
        frame_str = f"{self.frames[0]}-{self.frames[1]}" if self.frames else None
        params = dict(
            # bbox is getting deprecated in favor of intersectsWith
            # https://docs.asf.alaska.edu/api/keywords/#geospatial-parameters
            intersectsWith=self.wkt,
            start=self.start,
            end=self.end,
            processingLevel="SLC",
            relativeOrbit=self.track,
            flightDirection=self.flight_direction,
            maxResults=2000,
            output="geojson",
            platform="S1",  # Currently only supporting S1 right now
            beamMode="IW",
            frame=frame_str,
        )
        params = {k: v for k, v in params.items() if v is not None}
        base_url = "https://api.daac.asf.alaska.edu/services/search/param?{params}"
        return base_url.format(params=urlencode(params))

    def query_results(self) -> dict:
        """Query the ASF API and save the results to a file."""
        return _query_url(self._url)

    @staticmethod
    def _get_urls(results: dict) -> list[str]:
        return [r["properties"]["url"] for r in results["features"]]

    @staticmethod
    def _file_names(results: dict) -> list[str]:
        return [r["properties"]["fileName"] for r in results["features"]]

    def _download_with_aria(self, urls, log_dir: Filename = Path(".")):
        url_filename = self.out_dir / "urls.txt"
        with open(self.out_dir / url_filename, "w") as f:
            for u in urls:
                f.write(u + "\n")

        log_filename = Path(log_dir) / "aria2c.log"
        aria_cmd = f'aria2c -i "{url_filename}" -d "{self.out_dir}" --continue=true'
        logger.info("Downloading with aria2c")
        logger.info(aria_cmd)
        with open(log_filename, "w") as f:
            subprocess.run(aria_cmd, shell=True, stdout=f, stderr=f, text=True)

    def _download_with_wget(self, urls, log_dir: Filename = Path(".")):
        def download_url(idx_url_pair):
            idx, u = idx_url_pair
            log_filename = Path(log_dir) / f"wget_{idx:02d}.log"
            with open(log_filename, "w") as f:
                wget_cmd = f'wget -nc -c "{u}" -P "{self.out_dir}"'
                logger.info(f"({idx} / {len(urls)}): Downloading {u} with wget")
                logger.info(wget_cmd)
                subprocess.run(wget_cmd, shell=True, stdout=f, stderr=f, text=True)

        # Parallelize the download using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(download_url, enumerate(urls)))

    @log_runtime
    def download(self, *, log_dir: Path) -> list[Path]:
        """Download full SAFE files from ASF."""
        # Start by saving data available as geojson
        results = self.query_results()
        urls = self._get_urls(results)

        if not urls:
            raise ValueError("No results found for query")

        # Make the output directory
        logger.info(f"Saving to {self.out_dir}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        file_names = [self.out_dir / f for f in self._file_names(results)]

        # TODO: use aria if available? or just make wget parallel...
        self._download_with_wget(urls, log_dir=log_dir)

        if self.unzip:
            # Change to .SAFE extension
            logger.info("Unzipping files...")
            file_names = unzip_all(self.out_dir, out_dir=self.out_dir)
        return file_names


@lru_cache(maxsize=10)
def _query_url(url: str) -> dict:
    """Query the ASF API and save the results to a file."""
    logger.info("Querying url:")
    print(url, file=sys.stderr)
    resp = requests.get(url)
    resp.raise_for_status()
    results = json.loads(resp.content.decode("utf-8"))
    return results


def _unzip_one(filepath: Filename, pol: str = "vv", out_dir=Path(".")):
    """Unzip one Sentinel-1 zip file."""
    if pol is None:
        pol = ""
    with zipfile.ZipFile(filepath, "r") as zipref:
        # Get the list of files in the zip
        names_to_extract = [
            fp for fp in zipref.namelist() if pol.lower() in str(fp).lower()
        ]
        zipref.extractall(path=out_dir, members=names_to_extract)


def delete_tiffs_within_zip(data_path: Filename, pol: str = "vh"):
    """Delete (in place) the tiff files within a zip file matching `pol`."""
    cmd = f"""find {data_path} -name "S*.zip" | xargs -I{{}} -n1 -P4 zip -d {{}} '*-vh-*.tiff'"""  # noqa
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)


def unzip_one(
    filepath: Filename, pol: str = "vv", out_dir: Filename = Path(".")
) -> Path:
    """Unzip one Sentinel-1 zip file."""
    if pol is None:
        pol = ""

    # unzip all of these
    to_unzip = [pol.lower(), "preview", "support", "manifest.safe"]
    with zipfile.ZipFile(filepath, "r") as zipref:
        # Get the list of files in the zip
        names_to_extract = [
            fp
            for fp in zipref.namelist()
            if any(key in str(fp).lower() for key in to_unzip)
        ]
        zipref.extractall(path=out_dir, members=names_to_extract)
    # Return the path to the unzipped file
    return Path(filepath).with_suffix(".SAFE")


def unzip_all(
    path: Filename = ".",
    pol: str = "vv",
    out_dir: Filename = Path("."),
    delete_zips: bool = False,
    n_workers: int = 4,
) -> list[Path]:
    """Find all .zips and unzip them, skipping overwrites."""
    zip_files = list(Path(path).glob("S1[AB]_*IW*.zip"))
    logger.info(f"Found {len(zip_files)} zip files to unzip")

    existing_safes = list(Path(path).glob("S1[AB]_*IW*.SAFE"))
    logger.info(f"Found {len(existing_safes)} SAFE files already unzipped")

    # Skip if already unzipped
    files_to_unzip = [
        fp for fp in zip_files if fp.stem not in [sf.stem for sf in existing_safes]
    ]
    logger.info(f"Unzipping {len(files_to_unzip)} zip files")
    # Unzip in parallel
    newly_unzipped = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(unzip_one, fp, pol=pol, out_dir=out_dir)
            for fp in files_to_unzip
        ]
        for future in as_completed(futures):
            newly_unzipped.append(future.result())

    if delete_zips:
        for fp in files_to_unzip:
            fp.unlink()
    return newly_unzipped + existing_safes
