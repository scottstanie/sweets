from sweets._dolphin import (
    DolphinOptions,
    _estimate_snaphu_tiles_from_bounds,
    build_displacement_config,
)


def test_estimate_snaphu_tiles_small_bounds():
    bounds = (-102.2, 32.15, -102.1, 32.22)
    assert _estimate_snaphu_tiles_from_bounds(bounds, (6, 12)) == (1, 1)


def test_estimate_snaphu_tiles_thresholds():
    assert _estimate_snaphu_tiles_from_bounds((0.0, 0.0, 0.01, 0.539), (6, 12)) == (
        2,
        2,
    )
    assert _estimate_snaphu_tiles_from_bounds((0.0, 0.0, 0.01, 1.078), (6, 12)) == (
        3,
        3,
    )


def test_estimate_snaphu_tiles_respects_strides():
    bounds = (0.0, 0.0, 0.01, 0.539)
    assert _estimate_snaphu_tiles_from_bounds(bounds, (12, 24)) == (1, 1)


BOUNDS = (-102.2, 32.15, -102.1, 32.22)


def _build(tmp_path, **kwargs):
    cslc_files = [tmp_path / "t087_20221215_iw2.h5", tmp_path / "t087_20221227_iw2.h5"]
    return build_displacement_config(
        cslc_files,
        tmp_path / "work",
        options=DolphinOptions(**kwargs),
        bounds=BOUNDS,
    )


def test_whirlwind_is_the_default_unwrapper(tmp_path):
    cfg = _build(tmp_path)
    assert cfg.unwrap_options.unwrap_method == "whirlwind"


def test_whirlwind_options_are_forwarded(tmp_path):
    cfg = _build(
        tmp_path,
        whirlwind_num_threads=4,
        whirlwind_interpolate=True,
        whirlwind_interp_cutoff=0.3,
        whirlwind_bridge=False,
    )
    ww = cfg.unwrap_options.whirlwind_options
    assert ww.num_threads == 4
    assert ww.interpolate is True
    assert ww.interp_cutoff == 0.3
    assert ww.bridge is False


def test_snaphu_options_only_built_for_snaphu(tmp_path):
    cfg = _build(tmp_path, unwrap_method="snaphu", snaphu_ntiles=(3, 3))
    assert cfg.unwrap_options.unwrap_method == "snaphu"
    assert tuple(cfg.unwrap_options.snaphu_options.ntiles) == (3, 3)
