"""Tests for the sweets CLI layer — specifically :class:`ConfigCli`.

The CLI-facing subclass is subtle enough to deserve its own tests:

- a ``mode="wrap"`` validator re-assembles the discriminated-union
  ``search`` field from flat ``--source`` / ``--start`` / ``--track``
  flags before Workflow's own validators run;
- ``dem_filename`` / ``water_mask_filename`` are re-declared as
  ``Optional[Path] = None`` to work around a tyro limitation, but still
  need to fall back to Workflow's ``<work_dir>/...`` defaults after the
  CLI has gone through ``execute()``;
- flat source flags are ``exclude=True`` so they must not leak into the
  serialized YAML — the YAML on disk has to be a byte-equivalent pure
  Workflow.

These tests avoid touching ASF, COMPASS, or dolphin — they exercise
ConfigCli's validation and YAML-dump logic end-to-end, and run
``tyro.cli(ConfigCli, args=...)`` once to catch regressions in the
parser wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tyro
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from sweets._dolphin import DolphinOptions
from sweets._tropo import TropoOptions
from sweets.cli import ConfigCli
from sweets.core import Workflow
from sweets.download import (
    BurstSearch,
    LocalSafeSearch,
    NisarGslcSearch,
    OperaCslcSearch,
)


@pytest.fixture
def bbox() -> tuple[float, float, float, float]:
    return (-104.0, 32.0, -103.0, 33.0)


@pytest.fixture
def safe_kwargs(
    tmp_path: Path, bbox: tuple[float, float, float, float]
) -> dict[str, Any]:
    """Minimum kwargs for a valid `--source safe` ConfigCli instance."""
    return {
        "start": "2023-01-01",
        "end": "2023-02-01",
        "source": "safe",
        "track": 78,
        "bbox": bbox,
        "work_dir": tmp_path,
        "out_dir": tmp_path / "data",
    }


class TestAssembleSearch:
    """The `mode="wrap"` validator that builds `search` from flat flags."""

    def test_safe_source_builds_burst_search(self, safe_kwargs):
        cfg = ConfigCli(**safe_kwargs)
        assert isinstance(cfg.search, BurstSearch)
        assert cfg.search.kind == "safe"
        assert cfg.search.track == 78
        assert cfg.search.polarizations == ["VV"]

    def test_safe_custom_polarizations_and_swaths(self, safe_kwargs):
        cfg = ConfigCli(
            **safe_kwargs,
            polarizations=["VV", "VH"],
            swaths=["IW2"],
        )
        assert isinstance(cfg.search, BurstSearch)
        assert cfg.search.polarizations == ["VV", "VH"]
        assert cfg.search.swaths == ["IW2"]

    def test_opera_cslc_source(self, tmp_path, bbox):
        cfg = ConfigCli(
            start="2023-01-01",
            end="2023-02-01",
            source="opera-cslc",
            bbox=bbox,
            work_dir=tmp_path,
            out_dir=tmp_path / "data",
        )
        assert isinstance(cfg.search, OperaCslcSearch)
        assert cfg.search.kind == "opera-cslc"

    def test_opera_cslc_track_optional_but_preserved(self, tmp_path, bbox):
        cfg = ConfigCli(
            start="2023-01-01",
            end="2023-02-01",
            source="opera-cslc",
            track=71,
            bbox=bbox,
            work_dir=tmp_path,
            out_dir=tmp_path / "data",
        )
        assert isinstance(cfg.search, OperaCslcSearch)
        assert cfg.search.track == 71

    def test_nisar_gslc_source(self, tmp_path, bbox):
        cfg = ConfigCli(
            start="2023-01-01",
            end="2023-02-01",
            source="nisar-gslc",
            track=101,
            frame=71,
            frequency="A",
            polarizations=["HH"],
            bbox=bbox,
            work_dir=tmp_path,
            out_dir=tmp_path / "data",
        )
        assert isinstance(cfg.search, NisarGslcSearch)
        assert cfg.search.track == 101
        assert cfg.search.frame == 71
        assert cfg.search.frequency == "A"
        assert cfg.search.polarizations == ["HH"]

    def test_local_source_skips_dates_and_track(self, tmp_path, bbox):
        """`--source local` should produce a LocalSafeSearch with no date pins."""
        cfg = ConfigCli(
            source="local",
            bbox=bbox,
            work_dir=tmp_path,
            out_dir=tmp_path / "data",
        )
        assert isinstance(cfg.search, LocalSafeSearch)
        assert cfg.search.kind == "local"
        assert cfg.search.out_dir == (tmp_path / "data").resolve()

    def test_non_local_source_requires_dates(self, tmp_path, bbox):
        with pytest.raises(ValidationError, match="--start and --end are required"):
            ConfigCli(
                source="safe",
                track=78,
                bbox=bbox,
                work_dir=tmp_path,
                out_dir=tmp_path / "data",
            )

    def test_safe_requires_track(self, tmp_path, bbox):
        with pytest.raises(
            ValidationError, match="--track is required for --source safe"
        ):
            ConfigCli(
                start="2023-01-01",
                end="2023-02-01",
                source="safe",
                bbox=bbox,
                work_dir=tmp_path,
                out_dir=tmp_path / "data",
            )

    def test_missing_aoi_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="bbox.*wkt"):
            ConfigCli(
                start="2023-01-01",
                end="2023-02-01",
                source="safe",
                track=78,
                work_dir=tmp_path,
                out_dir=tmp_path / "data",
            )


class TestLegacyDoTropo:
    """The `--do-tropo` alias must keep working for README / notebook examples."""

    def test_do_tropo_true_sets_tropo_enabled(self, safe_kwargs):
        cfg = ConfigCli(**safe_kwargs, do_tropo=True)
        assert cfg.tropo.enabled is True

    def test_do_tropo_false_leaves_tropo_disabled(self, safe_kwargs):
        cfg = ConfigCli(**safe_kwargs, do_tropo=False)
        assert cfg.tropo.enabled is False

    def test_nested_tropo_enabled_directly(self, safe_kwargs):
        cfg = ConfigCli(**safe_kwargs, tropo=TropoOptions(enabled=True))
        assert cfg.tropo.enabled is True


class TestNestedOverrides:
    """Nested pydantic sub-configs round-trip through instantiation."""

    def test_custom_dolphin_half_window(self, safe_kwargs):
        cfg = ConfigCli(
            **safe_kwargs,
            dolphin=DolphinOptions(half_window=(20, 10), unwrap_method="spurt"),
        )
        assert cfg.dolphin.half_window == (20, 10)
        assert cfg.dolphin.unwrap_method == "spurt"

    def test_slc_posting_override(self, safe_kwargs):
        cfg = ConfigCli(**safe_kwargs, slc_posting=(10, 2.5))
        assert cfg.slc_posting == (10, 2.5)


class TestExecuteRoundTrip:
    """``execute()`` writes a YAML that reloads as a pure Workflow."""

    def test_reloads_with_user_overrides_preserved(self, safe_kwargs, tmp_path):
        output = tmp_path / "config.yaml"
        cfg = ConfigCli(
            **safe_kwargs,
            slc_posting=(10, 2.5),
            dolphin=DolphinOptions(half_window=(20, 10), unwrap_method="spurt"),
            tropo=TropoOptions(enabled=True),
            output=output,
            with_schema=False,
        )
        cfg.execute()

        reloaded = Workflow.from_yaml(output)
        assert reloaded.slc_posting == (10, 2.5)
        assert reloaded.dolphin.half_window == (20, 10)
        assert reloaded.dolphin.unwrap_method == "spurt"
        assert reloaded.tropo.enabled is True
        assert reloaded.search.kind == "safe"
        assert reloaded.search.track == 78

    def test_default_paths_computed_from_work_dir(self, safe_kwargs, tmp_path):
        """dem_filename / water_mask_filename default to <work_dir>/*.tif."""
        output = tmp_path / "config.yaml"
        cfg = ConfigCli(**safe_kwargs, output=output, with_schema=False)
        cfg.execute()

        reloaded = Workflow.from_yaml(output)
        assert reloaded.dem_filename == tmp_path / "dem.tif"
        assert reloaded.water_mask_filename == tmp_path / "watermask.tif"

    def test_user_path_overrides_preserved(self, safe_kwargs, tmp_path):
        output = tmp_path / "config.yaml"
        my_dem = tmp_path / "my_dem.tif"
        my_mask = tmp_path / "my_mask.tif"
        cfg = ConfigCli(
            **safe_kwargs,
            dem_filename=my_dem,
            water_mask_filename=my_mask,
            output=output,
            with_schema=False,
        )
        cfg.execute()

        reloaded = Workflow.from_yaml(output)
        assert reloaded.dem_filename == my_dem
        assert reloaded.water_mask_filename == my_mask

    def test_flat_cli_fields_not_serialized(self, safe_kwargs, tmp_path):
        """Flat source flags and CLI-only knobs must not appear in the YAML.

        They're CLI input sugar — the canonical state lives on ``search``
        (for source details) or on nothing at all (for ``output``,
        ``with_schema``). If any of them leaked into the dumped YAML,
        round-tripping through ``Workflow.from_yaml`` would either break
        or silently retain CLI-only state across reloads.
        """
        output = tmp_path / "config.yaml"
        cfg = ConfigCli(**safe_kwargs, output=output, with_schema=False)
        cfg.execute()

        raw = yaml.safe_load(output.read_text())
        flat_keys = {
            "start",
            "end",
            "source",
            "track",
            "frame",
            "frequency",
            "polarizations",
            "swaths",
            "out_dir",
            "do_tropo",
            "output",
            "with_schema",
        }
        leaked = flat_keys & set(raw.keys())
        assert not leaked, f"flat CLI fields leaked into YAML top level: {leaked}"
        # Sanity: the canonical state is present on `search` instead.
        # `track` serializes through BurstSearch's `relativeOrbit` alias.
        assert raw["search"]["kind"] == "safe"
        assert raw["search"]["relativeOrbit"] == 78

    def test_schema_sidecar_written(self, safe_kwargs, tmp_path):
        output = tmp_path / "config.yaml"
        cfg = ConfigCli(**safe_kwargs, output=output, with_schema=True)
        cfg.execute()

        sidecar = output.with_suffix(output.suffix + ".schema.json")
        assert sidecar.exists()
        assert output.read_text().startswith("# yaml-language-server: $schema=")


class TestTyroIntegration:
    """End-to-end: run through `tyro.cli` so the argparse wiring is exercised."""

    def test_parses_nested_dolphin_and_tropo_flags(self, tmp_path):
        output = tmp_path / "config.yaml"
        args = [
            "--start",
            "2023-01-01",
            "--end",
            "2023-02-01",
            "--bbox",
            "-104",
            "32",
            "-103",
            "33",
            "--source",
            "opera-cslc",
            "--slc-posting",
            "10",
            "2.5",
            "--dolphin.half-window",
            "20",
            "10",
            "--dolphin.unwrap-method",
            "spurt",
            "--tropo.enabled",
            "--work-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "data"),
            "--output",
            str(output),
            "--no-with-schema",
        ]
        cfg = tyro.cli(ConfigCli, args=args)
        assert cfg.slc_posting == (10, 2.5)
        assert cfg.dolphin.half_window == (20, 10)
        assert cfg.dolphin.unwrap_method == "spurt"
        assert cfg.tropo.enabled is True
        search = cfg.search
        assert isinstance(search, OperaCslcSearch)
        assert search.kind == "opera-cslc"

        cfg.execute()
        assert output.exists()
        reloaded = Workflow.from_yaml(output)
        assert reloaded.dolphin.unwrap_method == "spurt"

    def test_search_field_suppressed_from_help(self, capsys):
        """`tyro.conf.Suppress` should keep `--search` out of the help text."""
        with pytest.raises(SystemExit):
            tyro.cli(ConfigCli, args=["--help"])
        out = capsys.readouterr().out
        # The discriminated-union `search` field must be hidden — if it
        # leaks back into the CLI it'll be spread across subcommand-style
        # choices and break the flat-flag UX.
        assert "--search" not in out
        # Nested groups should still be present.
        assert "--dolphin.half-window" in out
        assert "--tropo.enabled" in out
