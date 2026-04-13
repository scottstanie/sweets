"""Network-dependent tests for sweets.download search paths.

These replay recorded HTTP via pytest-recording, so they run offline in
CI after the cassette is committed. Re-record after an opera_utils / ASF
contract change with:

    pixi run pytest tests/test_download_search.py --record-mode=once
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sweets.download import OperaCslcSearch


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
