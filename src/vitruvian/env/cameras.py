# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Camera constants and quaternion helpers for the G1 sim.

The **head** cam is what the world model sees (robot's POV).
The **chase** and **side** cams are rendered only for demo videos.
"""

from __future__ import annotations

import numpy as np

# Head-cam pose relative to ``torso_link`` (scalar-first MuJoCo quat,
# world frame). Same values used during M4.2 dataset collection, so
# plan-time renders match training distribution.
HEAD_CAM_POS = [0.10, 0.0, 0.18]
HEAD_CAM_QUAT = [-0.5, -0.5, 0.5, 0.5]

# Chase / side cam world-frame offsets. Attached to ``worldbody`` so
# their orientation stays stable as the torso pitches/rolls.
CHASE_CAM_POS = [-2.5, 0.0, 1.5]
SIDE_CAM_POS = [0.0, -2.5, 1.0]

# Both chase and side look at this point (roughly torso COM).
CHASE_CAM_TARGET = [0.0, 0.0, 0.6]


def look_at_quat(
    cam_offset: list[float] | tuple[float, ...] | np.ndarray,
    target_offset: list[float] | tuple[float, ...] | np.ndarray = (0.0, 0.0, 0.0),
) -> list[float]:
    """Return a MuJoCo scalar-first quaternion so a camera at
    ``cam_offset`` looks at ``target_offset``.

    Works with ``TRACKCOM`` cameras whose quat is interpreted in the
    world frame. MuJoCo cameras look along their local -Z.
    """
    from scipy.spatial.transform import Rotation

    fwd = np.array(target_offset, dtype=np.float64) - np.array(
        cam_offset, dtype=np.float64
    )
    fwd /= np.linalg.norm(fwd)
    # Pick a world "up" that isn't parallel to the view direction, so the
    # ``fwd × up`` right-vector is well-defined even when the camera looks
    # straight down/up (fwd == ±Z would make ``cross(fwd, Z)`` zero-length).
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    R = np.column_stack([right, cam_up, -fwd])
    q = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


__all__ = [
    "CHASE_CAM_POS",
    "CHASE_CAM_TARGET",
    "HEAD_CAM_POS",
    "HEAD_CAM_QUAT",
    "SIDE_CAM_POS",
    "look_at_quat",
]
