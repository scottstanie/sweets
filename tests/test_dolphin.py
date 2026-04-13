from sweets._dolphin import _estimate_snaphu_tiles_from_bounds


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
