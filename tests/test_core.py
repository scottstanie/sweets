"""Lightweight tests for the Workflow config object.

These avoid touching ASF, COMPASS or dolphin — they exercise validation,
YAML round-trip, and the bbox/wkt cross-fill logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweets.core import Workflow
from sweets.download import BurstSearch, NisarGslcSearch, OperaCslcSearch


@pytest.fixture
def bbox() -> tuple[float, float, float, float]:
    return (-102.2, 32.15, -102.1, 32.22)


@pytest.fixture
def search_kwargs() -> dict:
    return {
        "start": "2022-12-15",
        "end": "2022-12-29",
        "track": 78,
    }


class TestWorkflow:
    def test_construct_from_dict(self, tmp_path, bbox, search_kwargs):
        w = Workflow(
            bbox=bbox,
            search={**search_kwargs, "out_dir": "data"},
            n_workers=1,
            threads_per_worker=16,
        )
        outfile = tmp_path / "config.yaml"
        w.to_yaml(outfile, with_comments=True)
        w2 = Workflow.from_yaml(outfile)
        assert w.model_dump() == w2.model_dump()

    def test_construct_from_burst_search_instance(self, bbox, search_kwargs):
        search = BurstSearch(bbox=bbox, **search_kwargs)
        w = Workflow(search=search, bbox=bbox)
        assert w.bbox == bbox
        assert w.search.track == 78

    def test_bbox_wkt_cross_fill(self, tmp_path, search_kwargs):
        wkt_str = "POLYGON((-10.0 30.0,-9.0 30.0,-9.0 31.0,-10.0 31.0,-10.0 30.0))"
        expected_bbox = (-10, 30, -9, 31)

        # bbox in -> bbox stays bbox (no auto-wkt fill anymore — outer wkt is
        # only set if the user explicitly supplied one)
        w = Workflow(bbox=expected_bbox, search=search_kwargs)
        assert w.bbox == expected_bbox

        # wkt string in -> bbox derived from wkt
        w = Workflow(wkt=wkt_str, search=search_kwargs)
        assert w.bbox == expected_bbox
        assert w.wkt == wkt_str
        # And the wkt is propagated down into the search source for
        # downloaders that need a polygon (NisarGslcSearch, OperaCslcSearch).
        assert w.search.wkt == wkt_str

        # wkt path in -> bbox out (file content is read at validation time)
        wkt_file = tmp_path / "aoi.wkt"
        wkt_file.write_text(wkt_str)
        w = Workflow(wkt=str(wkt_file), search=search_kwargs)
        assert w.bbox == expected_bbox

    def test_default_factory_order(self, bbox, search_kwargs):
        # Custom paths are honored.
        dem = Path("dem")
        mask = Path("mask")
        w = Workflow(
            bbox=bbox,
            search=search_kwargs,
            dem_filename=dem,
            water_mask_filename=mask,
        )
        assert w.dem_filename == dem
        assert w.water_mask_filename == mask

        # Defaults are derived from work_dir.
        w = Workflow(bbox=bbox, search=search_kwargs)
        assert w.dem_filename == w.work_dir / "dem.tif"
        assert w.water_mask_filename == w.work_dir / "watermask.tif"
        assert w.log_dir == w.work_dir / "logs"

    def test_missing_aoi_raises(self, search_kwargs):
        with pytest.raises(ValueError, match="bbox.*wkt"):
            Workflow(search=search_kwargs)

    def test_invalid_bbox_raises(self, search_kwargs):
        # Latitude swapped
        with pytest.raises(ValueError, match="Latitude"):
            Workflow(bbox=(-10, 31, -9, 30), search=search_kwargs)
        # Longitude swapped
        with pytest.raises(ValueError, match="Longitude"):
            Workflow(bbox=(-9, 30, -10, 31), search=search_kwargs)

    def test_default_kind_is_safe(self, bbox, search_kwargs):
        """A `search` dict without a `kind` key should default to BurstSearch."""
        w = Workflow(bbox=bbox, search=search_kwargs)
        assert isinstance(w.search, BurstSearch)
        assert w.search.kind == "safe"

    def test_opera_cslc_kind(self, bbox, search_kwargs):
        """`kind: opera-cslc` should produce an OperaCslcSearch source."""
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {"kind": "opera-cslc", **search_kwargs},
            }
        )
        assert isinstance(w.search, OperaCslcSearch)
        assert w.search.kind == "opera-cslc"
        # bbox cross-fill still works
        assert w.search.bbox == bbox

    def test_opera_cslc_yaml_roundtrip(self, tmp_path, bbox, search_kwargs):
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {"kind": "opera-cslc", **search_kwargs},
                "tropo": {"enabled": True},
            }
        )
        out = tmp_path / "config.yaml"
        w.to_yaml(out, with_comments=True)
        w2 = Workflow.from_yaml(out)
        assert isinstance(w2.search, OperaCslcSearch)
        assert w2.tropo.enabled is True

    def test_nisar_gslc_kind(self, bbox):
        """`kind: nisar-gslc` should produce a NisarGslcSearch source."""
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {
                    "kind": "nisar-gslc",
                    "start": "2024-06-01",
                    "end": "2024-08-10",
                    "frequency": "A",
                    "polarizations": ["HH"],
                },
            }
        )
        assert isinstance(w.search, NisarGslcSearch)
        assert w.search.kind == "nisar-gslc"
        assert w.search.frequency == "A"
        assert w.search.polarizations == ["HH"]
        assert w.search.bbox == bbox
        # NISAR subdataset path that gets handed to dolphin
        assert w.search.hdf5_subdataset == "/science/LSAR/GSLC/grids/frequencyA/HH"

    def test_nisar_gslc_yaml_roundtrip(self, tmp_path, bbox):
        # `track`/`frame` are the user-facing names; the opera-utils field
        # aliases (`relative_orbit_number` / `track_frame_number`) also work.
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {
                    "kind": "nisar-gslc",
                    "start": "2024-06-01",
                    "end": "2024-08-10",
                    "track": 13,
                    "frame": 71,
                    "frequency": "A",
                    "polarizations": ["HH"],
                },
            }
        )
        out = tmp_path / "config.yaml"
        w.to_yaml(out, with_comments=True)
        w2 = Workflow.from_yaml(out)
        assert isinstance(w2.search, NisarGslcSearch)
        assert w2.search.track == 13
        assert w2.search.frame == 71
        assert w2.search.frequency == "A"

    def test_nisar_gslc_alias_field_names(self, bbox):
        """opera-utils' canonical names should still validate via aliases."""
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {
                    "kind": "nisar-gslc",
                    "start": "2024-06-01",
                    "end": "2024-08-10",
                    "relative_orbit_number": 13,
                    "track_frame_number": 71,
                },
            }
        )
        assert isinstance(w.search, NisarGslcSearch)
        assert w.search.track == 13
        assert w.search.frame == 71
