"""Quick smoke-test driver for sweets v0.2.

Edit the bbox / dates below for your own AOI. The default targets a tiny
patch over Pecos, TX, where Sentinel-1 track 78 has good summer 2021
coverage.

Run with::

    python scripts/demo_sweet.py
"""

from __future__ import annotations

from rich import print

from sweets.core import Workflow

if __name__ == "__main__":
    w = Workflow.model_validate(
        {
            "bbox": (-102.96, 31.22, -101.91, 31.56),
            "search": {
                "start": "2021-06-05",
                "end": "2021-08-10",
                "track": 78,
                "out_dir": "data",
            },
            "n_workers": 2,
            "threads_per_worker": 8,
        }
    )
    print(w.run())
