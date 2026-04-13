"""Tests for sweets._show_versions."""

from __future__ import annotations

import pytest

from sweets import _show_versions as sv


def test_get_sys_info_has_expected_keys() -> None:
    info = sv._get_sys_info()
    assert set(info.keys()) == {"python", "executable", "machine"}
    for v in info.values():
        assert isinstance(v, str)
        assert v


def test_get_version_returns_real_version_for_numpy() -> None:
    v = sv._get_version("numpy")
    assert v is not None
    assert isinstance(v, str)
    assert v[0].isdigit()


def test_get_version_returns_none_for_missing_module() -> None:
    assert sv._get_version("definitely_not_a_real_module_xyz_123") is None


def test_get_deps_info_returns_dict() -> None:
    deps = sv._get_deps_info()
    assert isinstance(deps, dict)
    for name in ("numpy", "pydantic", "shapely"):
        assert name in deps
        assert deps[name] is not None


def test_show_versions_prints_output(capsys: pytest.CaptureFixture[str]) -> None:
    sv.show_versions()
    out = capsys.readouterr().out
    assert "sweets core packages" in out
    assert "System:" in out
    assert "Python deps:" in out
