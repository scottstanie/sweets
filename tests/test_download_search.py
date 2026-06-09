"""Network-dependent tests for sweets.download search paths.

These replay recorded HTTP via pytest-recording, so they run offline in
CI after the cassette is committed. Re-record after an opera_utils / ASF
contract change with:

    pixi run pytest tests/test_download_search.py --record-mode=once
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweets.download import LocalSafeSearch, OperaCslcSearch


def _make_safes(d: Path, names: list[str]) -> None:
    for n in names:
        (d / n).mkdir()


_AOI = (-118.394, 33.728, -118.347, 33.762)
_SAFES = [
    "S1A_IW_SLC__1SSV_20250906T135253_20250906T135256_x.SAFE",
    "S1C_IW_SLC__1SSV_20251217T135149_20251217T135151_x.SAFE",  # S1C in window
    "S1D_IW_SLC__1SSV_20260116T135150_20260116T135153_x.SAFE",  # S1D in window
    "S1A_IW_SLC__1SSV_20240103T135300_20240103T135303_x.SAFE",  # before window
    "S1A_IW_SLC__1SSV_20260516T135246_20260516T135246_x.SAFE",  # after window
]


def test_local_safe_search_includes_s1c_and_s1d(tmp_path: Path) -> None:
    """`existing_safes` must not drop Sentinel-1C/D products (S1[AB] -> S1[A-D])."""
    _make_safes(tmp_path, _SAFES)
    s = LocalSafeSearch(out_dir=tmp_path, bbox=_AOI)
    found = s.existing_safes()
    assert len(found) == 5
    assert any("S1C" in p.name for p in found)
    assert any("S1D" in p.name for p in found)


def test_local_safe_search_filters_by_date(tmp_path: Path) -> None:
    """Optional start/end keep only SAFEs whose acquisition is in range."""
    _make_safes(tmp_path, _SAFES)
    s = LocalSafeSearch(
        out_dir=tmp_path, bbox=_AOI, start="2025-09-01", end="2026-03-04"
    )
    found = sorted(p.name[:25] for p in s.existing_safes())
    assert found == [
        "S1A_IW_SLC__1SSV_20250906",
        "S1C_IW_SLC__1SSV_20251217",
        "S1D_IW_SLC__1SSV_20260116",
    ]


def test_local_safe_search_no_dates_uses_all(tmp_path: Path) -> None:
    """With no start/end, every overlapping SAFE is returned (back-compat)."""
    _make_safes(tmp_path, _SAFES)
    assert len(LocalSafeSearch(out_dir=tmp_path, bbox=_AOI).existing_safes()) == 5


@pytest.mark.vcr
def test_opera_cslc_search_resolves_burst_ids(tmp_path: Path) -> None:
    """`_resolve_burst_ids` returns OPERA burst IDs for a known AOI/track.

    Uses the same LA AOI + track 71 as `docs/example_opera_cslc.ipynb`,
    trimmed to a 5-day window so the cassette stays small.
    """
    search = OperaCslcSearch.model_validate(
        {
            "bbox": (-118.3957, 33.7284, -118.3459, 33.772),
            "start": "2025-12-01",
            "end": "2025-12-06",
            "track": 71,
            "out_dir": tmp_path,
        }
    )
    burst_ids = search._resolve_burst_ids()
    assert burst_ids, "ASF returned no OPERA bursts for this AOI/track/dates"
    # OPERA burst IDs look like "t071_151200_iw2": track-prefixed, _iw<n> suffix.
    assert all(bid.startswith("t071_") for bid in burst_ids)
    assert all("_iw" in bid for bid in burst_ids)
