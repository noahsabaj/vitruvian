#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M0.3 capture — load Unitree G1, apply random actuator commands,
render a short headless animation.

Confirms the full MuJoCo path is wired end-to-end on this host: MJCF
parses, physics steps, actuators take input, offscreen renderer
produces frames, imageio serializes them to a GIF. Saves to
docs/journal/assets/ for the paper trail.

Run
---
    source scripts/env-setup.sh     # unsets LD_LIBRARY_PATH
    uv run python scripts/m03_capture.py
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
OUT_GIF = ROOT / "docs" / "journal" / "assets" / "2026-04-16-g1-ragdoll.gif"
OUT_PNG = ROOT / "docs" / "journal" / "assets" / "2026-04-16-g1-initial.png"

SEED = 0
DURATION = 3.0          # seconds
FPS = 20
RENDER_W, RENDER_H = 480, 360
# Fraction of each actuator's ctrl range to exercise. Low enough that
# the ragdoll flails rather than instantly self-destructs under
# arbitrary torques.
CTRL_SCALE = 0.25


def build_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.2
    cam.azimuth = 110.0
    cam.elevation = -15.0
    cam.lookat[:] = np.array([0.0, 0.0, 0.6])
    return cam


def main() -> None:
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    camera = build_camera()

    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (FPS * dt))))
    total_steps = int(round(DURATION * FPS)) * steps_per_frame
    ctrl_update_every = max(1, steps_per_frame // 2)

    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]
    ctrl_mid = 0.5 * (ctrl_lo + ctrl_hi)
    ctrl_span = 0.5 * (ctrl_hi - ctrl_lo) * CTRL_SCALE

    print(
        f"G1: {model.nu} actuators, "
        f"dt={dt * 1e3:.2f} ms, "
        f"steps/frame={steps_per_frame}, "
        f"total_steps={total_steps}"
    )

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    # Initial frame before any control — confirms the starting pose renders.
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    imageio.imwrite(str(OUT_PNG), renderer.render())
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")

    frames = []
    for i in range(total_steps):
        if i % ctrl_update_every == 0:
            data.ctrl[:] = ctrl_mid + ctrl_span * rng.uniform(-1.0, 1.0, size=model.nu)
        mujoco.mj_step(model, data)
        if i % steps_per_frame == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

    print(f"Captured {len(frames)} frames at {FPS} fps, {RENDER_W}x{RENDER_H}.")
    imageio.mimsave(str(OUT_GIF), frames, fps=FPS, loop=0)
    size_kb = OUT_GIF.stat().st_size // 1024
    print(f"Wrote {OUT_GIF.relative_to(ROOT)} ({size_kb} KB)")
    print(
        f"Final base z = {data.qpos[2]:.3f} m "
        f"(starts ~0.79; gravity + random torques → ragdoll)"
    )


if __name__ == "__main__":
    main()
