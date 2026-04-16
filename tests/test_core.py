"""Lightweight tests for the Workflow config object.

These avoid touching ASF, COMPASS or dolphin — they exercise validation,
YAML round-trip, and the bbox/wkt cross-fill logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweets.core import Workflow
from sweets.download import (
    BurstSearch,
    LocalSafeSearch,
    NisarGslcSearch,
    OperaCslcSearch,
)


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
        # `hdf5_subdataset()` would normally peek at a CMR result or a
        # cached file to learn the actual frequency / pol; here we just
        # check that the user's explicit choices feed straight through
        # the construction (the resolver is exercised in the smoke test).
        assert w.search.frequency == "A"
        assert w.search.polarizations == ["HH"]

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


class TestLocalSafeSearch:
    """LocalSafeSearch: pre-downloaded .SAFE dirs / .zip archives, no download."""

    def test_kind_and_bbox(self, tmp_path, bbox):
        src = LocalSafeSearch(out_dir=tmp_path, bbox=bbox)
        assert src.kind == "local"
        assert src.out_dir == tmp_path.resolve()
        assert src.aoi.bounds == bbox

    def test_requires_aoi(self, tmp_path):
        with pytest.raises(ValueError, match="bbox.*wkt"):
            LocalSafeSearch(out_dir=tmp_path)

    def test_existing_safes_prefers_safe_over_zip(self, tmp_path, bbox):
        """When both .SAFE and .zip are in `out_dir`, .SAFE wins."""
        safe = tmp_path / "S1A_IW_SLC__1SDV_20230101T000000.SAFE"
        safe.mkdir()
        zipf = tmp_path / "S1A_IW_SLC__1SDV_20230101T000000.zip"
        zipf.write_bytes(b"")

        src = LocalSafeSearch(out_dir=tmp_path, bbox=bbox)
        assert src.existing_safes() == [safe]

    def test_existing_safes_picks_up_zip(self, tmp_path, bbox):
        """With only .zip files present, those are returned."""
        zipf = tmp_path / "S1A_IW_SLC__1SDV_20230101T000000.zip"
        zipf.write_bytes(b"")

        src = LocalSafeSearch(out_dir=tmp_path, bbox=bbox)
        assert src.existing_safes() == [zipf]

    def test_existing_safes_empty_dir(self, tmp_path, bbox):
        src = LocalSafeSearch(out_dir=tmp_path, bbox=bbox)
        assert src.existing_safes() == []


class TestWorkflowLocal:
    """Workflow integration for the LocalSafeSearch source."""

    def test_local_kind_via_workflow(self, tmp_path, bbox):
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {"kind": "local", "out_dir": str(tmp_path)},
            }
        )
        assert isinstance(w.search, LocalSafeSearch)
        assert w.search.kind == "local"
        assert w.search.bbox == bbox

    def test_local_yaml_roundtrip(self, tmp_path, bbox):
        data_dir = tmp_path / "safes"
        data_dir.mkdir()
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {"kind": "local", "out_dir": str(data_dir)},
            }
        )
        out = tmp_path / "config.yaml"
        w.to_yaml(out, with_comments=True)
        w2 = Workflow.from_yaml(out)
        assert isinstance(w2.search, LocalSafeSearch)
        assert w2.search.out_dir == data_dir.resolve()
        assert w2.search.bbox == bbox

    def test_local_uses_compass_dem_buffer(self, tmp_path, bbox):
        """LocalSafeSearch should get the wide COMPASS DEM buffer (not the 0.25 deg default)."""
        w = Workflow.model_validate(
            {
                "bbox": bbox,
                "search": {"kind": "local", "out_dir": str(tmp_path)},
            }
        )
        # 1.0 deg buffer — full IW frame fits.
        assert w._dem_bbox == (
            bbox[0] - 1.0,
            bbox[1] - 1.0,
            bbox[2] + 1.0,
            bbox[3] + 1.0,
        )
