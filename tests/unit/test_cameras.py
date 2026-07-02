# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Camera quaternion helper — look_at_quat degeneracy handling."""

from __future__ import annotations

import numpy as np

from vitruvian.env.cameras import look_at_quat


def _is_unit_quat(q: list[float]) -> bool:
    return len(q) == 4 and abs(sum(x * x for x in q) - 1.0) < 1e-6


def test_look_at_quat_general() -> None:
    q = look_at_quat([-2.5, 0.0, 1.5], [0.0, 0.0, 0.6])
    assert all(np.isfinite(q)) and _is_unit_quat(q)


def test_look_at_quat_straight_down_no_nan() -> None:
    """Camera directly above the target: fwd == -Z makes the naive
    ``cross(fwd, Z)`` zero-length → a divide-by-zero NaN. The helper picks an
    alternate 'up' so the quaternion stays finite and unit."""
    q = look_at_quat([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    assert all(np.isfinite(q)), f"got non-finite quat {q}"
    assert _is_unit_quat(q)


def test_look_at_quat_straight_up_no_nan() -> None:
    q = look_at_quat([0.0, 0.0, 0.0], [0.0, 0.0, 2.0])
    assert all(np.isfinite(q)) and _is_unit_quat(q)
