#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M2 — add a first-person head-mounted camera to G1 and render its
feed during a policy rollout.

G1 has no separate head body; the "head" mesh is rigid-attached to
torso_link. We attach a camera to torso_link at a reasonable eye-level
forward pose using mujoco.MjSpec (no submodule edits, no duplicated
MJCF). Physics is driven by the unmodified playground env; rendering
uses our augmented MjModel with the new camera, fed each step's qpos /
qvel from the playground state.

M2 exit criterion: a head-camera frame sequence exists on disk, proving
pixels from G1's first-person perspective flow cleanly through the
pipeline. Integrating these pixels into the env observation dict is M3.

Runs
----
    source scripts/env-setup.sh
    uv run python scripts/m2_head_camera.py
    uv run python scripts/m2_head_camera.py --ckpt checkpoints/m1-g1-full/000043253760
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import functools
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp


# brax 0.14.2 still references jax.device_put_replicated (removed in
# JAX 0.10). Same shim we use in m1_train.py / m1_render.py.
def _device_put_replicated_shim(value, devices):
    try:
        n = len(devices)
    except TypeError:
        n = 1

    def _replicate_leaf(x):
        arr = jnp.asarray(x)
        expanded = jnp.expand_dims(arr, 0)
        if n > 1:
            expanded = jnp.broadcast_to(expanded, (n,) + arr.shape)
        return jax.device_put(expanded)

    return jax.tree.map(_replicate_leaf, value)


jax.device_put_replicated = _device_put_replicated_shim  # type: ignore[attr-defined]


import imageio.v2 as imageio
import mujoco
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = "G1JoystickFlatTerrain"
ENV_OVERRIDES = {"njmax": 96}
G1_SCENE = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

# Head-camera placement, expressed in torso_link local frame.
# G1 has no head body; the head mesh is a geom on torso_link at
# pos=(0.0039635, 0, -0.044). Empirically, putting the camera a bit
# forward of and above torso_link origin gives a "looking out through
# the face" view. Fine-tune by inspecting the sample PNG; tweak pos
# here if the framing is off.
HEAD_CAM_POS = [0.10, 0.0, 0.18]

# MuJoCo cameras look down their own -Z axis by default. For a
# forward-looking (+X world) camera with world-up preserved we need a
# right-handed cam basis with cam_z = world -X:
#   cam_x = world -Y
#   cam_y = world +Z
#   cam_z = cam_x × cam_y = world -X   (det(R) = +1)
# Verified by round-trip through mujoco.mju_mat2Quat → mju_quat2Mat.
# MuJoCo quaternion convention is (w, x, y, z).
HEAD_CAM_QUAT = [-0.5, -0.5, 0.5, 0.5]


def build_model_with_head_cam(scene_xml: Path) -> mujoco.MjModel:
    """Load G1's scene MJCF, attach a 'head' camera to torso_link, return
    the compiled model."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    torso = spec.body("torso_link")
    if torso is None:
        raise RuntimeError("torso_link body not found in G1 MJCF")
    cam = torso.add_camera()
    cam.name = "head"
    cam.pos = HEAD_CAM_POS
    cam.quat = HEAD_CAM_QUAT
    return spec.compile()


def load_params(ckpt_path: Path):
    """Pickle or orbax directory, same as m1_render.py."""
    if ckpt_path.is_dir():
        params = brax_checkpoint.load(str(ckpt_path.resolve()))
        print(f"loaded orbax checkpoint dir {ckpt_path.name}")
        return params
    with ckpt_path.open("rb") as f:
        params = pickle.load(f)
    print(f"loaded pickle {ckpt_path.name}")
    return params


def make_policy(params, env):
    cfg = locomotion_params.brax_ppo_config(ENV_NAME)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.network_factory.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_factory.value_hidden_layer_sizes),
        policy_obs_key=cfg.network_factory.policy_obs_key,
        value_obs_key=cfg.network_factory.value_obs_key,
    )
    preprocess_fn = (
        running_statistics.normalize
        if cfg.normalize_observations
        else lambda obs, *_: obs
    )
    net = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    inference_factory = ppo_networks.make_inference_fn(net)
    return jax.jit(inference_factory(params, deterministic=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "m1-g1-full" / "000043253760",
        help="Pickle file or orbax directory with trained PPO params",
    )
    ap.add_argument(
        "--out_png",
        type=Path,
        default=ROOT
        / "docs"
        / "journal"
        / "assets"
        / "2026-04-16-m2-head-cam-sample.png",
    )
    ap.add_argument(
        "--out_gif",
        type=Path,
        default=ROOT
        / "docs"
        / "journal"
        / "assets"
        / "2026-04-16-m2-head-cam-rollout.gif",
    )
    ap.add_argument("--num_frames", type=int, default=200)  # 4 s @ 50 fps
    ap.add_argument("--render_h", type=int, default=240)
    ap.add_argument("--render_w", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)

    # Build the head-cam rendering model.
    print(f"building MjModel with head camera from {G1_SCENE.name} ...")
    mj_model = build_model_with_head_cam(G1_SCENE)
    mj_data = mujoco.MjData(mj_model)
    try:
        cam_id = mj_model.camera("head").id
    except Exception:
        raise RuntimeError("head camera not present after spec.compile()")
    print(f"  cameras on model: {mj_model.ncam}  (head cam id={cam_id})")

    # Playground env drives the physics for the policy rollout.
    env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)
    print(f"  env obs_size: {env.observation_size}")
    print(f"  env action_size: {env.action_size}")

    # Load the trained policy.
    params = load_params(args.ckpt)
    inference_fn = make_policy(params, env)

    # Renderer on the augmented model.
    renderer = mujoco.Renderer(mj_model, height=args.render_h, width=args.render_w)

    # Rollout.
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    state = reset_fn(rng)

    frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    mid_frame: np.ndarray | None = None
    for i in range(args.num_frames):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)
        mjx_data = getattr(state, "data", None) or getattr(
            state, "pipeline_state", None
        )
        if mjx_data is None:
            raise RuntimeError(
                "Cannot locate mjx.Data on state — expected state.data or "
                "state.pipeline_state."
            )
        mj_data.qpos[:] = np.array(mjx_data.qpos)
        mj_data.qvel[:] = np.array(mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam_id)
        frame = renderer.render().copy()
        frames.append(frame)
        if i == args.num_frames // 2:
            mid_frame = frame

    wall = time.perf_counter() - t0
    print(f"rendered {len(frames)} head-cam frames in {wall:.1f}s")

    # Save a representative PNG (mid-rollout).
    assert mid_frame is not None
    imageio.imwrite(str(args.out_png), mid_frame)
    print(
        f"  sample PNG:  {args.out_png.relative_to(ROOT)} "
        f"({args.out_png.stat().st_size // 1024} KB, shape={mid_frame.shape})"
    )

    # Save the full rollout as a GIF.
    imageio.mimsave(str(args.out_gif), frames, fps=50, loop=0)
    print(
        f"  rollout GIF: {args.out_gif.relative_to(ROOT)} "
        f"({args.out_gif.stat().st_size // 1024} KB, {len(frames)} frames)"
    )

    print()
    print("M2 exit criterion: head-camera frames render from G1's first-person view.")
    print("                   Integrating pixels into the env obs dict is M3.")


if __name__ == "__main__":
    main()
