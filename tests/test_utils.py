"""Tests for sweets.utils pure helpers.

No network, no real raster files - exercises the conversion functions
(to_wkt, to_bbox, get_overlapping_bounds) and the cache-dir resolver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sweets.utils import get_cache_dir, get_overlapping_bounds, to_bbox, to_wkt


@pytest.fixture
def square_geojson() -> str:
    """Return a unit-square polygon at the origin as a GeoJSON string."""
    return json.dumps(
        {
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            ],
        }
    )


def test_to_wkt_and_back_round_trips(square_geojson: str) -> None:
    wkt_str = to_wkt(square_geojson)
    assert wkt_str.startswith("POLYGON")
    assert to_bbox(wkt_str=wkt_str) == (0.0, 0.0, 1.0, 1.0)


def test_to_bbox_from_geojson(square_geojson: str) -> None:
    assert to_bbox(geojson=square_geojson) == (0.0, 0.0, 1.0, 1.0)


def test_to_bbox_from_wkt() -> None:
    wkt_str = "POLYGON ((-1 -2, 3 -2, 3 4, -1 4, -1 -2))"
    assert to_bbox(wkt_str=wkt_str) == (-1.0, -2.0, 3.0, 4.0)


def test_to_bbox_raises_without_input() -> None:
    with pytest.raises(ValueError, match="geojson or wkt_str"):
        to_bbox()


def test_overlapping_bounds_partial_overlap() -> None:
    assert get_overlapping_bounds((0, 0, 2, 2), (1, 1, 3, 3)) == (1.0, 1.0, 2.0, 2.0)


def test_overlapping_bounds_disjoint_is_nan() -> None:
    # Shapely 2.x returns (nan, nan, nan, nan) for the bounds of an empty
    # geometry; callers relying on intersection nonemptiness need to check.
    import math

    bounds = get_overlapping_bounds((0, 0, 1, 1), (2, 2, 3, 3))
    assert len(bounds) == 4
    assert all(math.isnan(v) for v in bounds)


def test_get_cache_dir_respects_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    p = get_cache_dir()
    assert p == tmp_path / "sweets"
    assert p.is_dir()
