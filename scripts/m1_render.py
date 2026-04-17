#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Render a rollout of a trained G1JoystickFlatTerrain policy to a GIF.

Intentionally separate from m1_train.py so rendering starts with a
fresh VRAM state — end-of-training rendering in the same process OOMs
on the 8 GB 4060 Ti because training's Warp kernels and replay buffers
are still resident.

Runs
----
    source scripts/env-setup.sh
    uv run python scripts/m1_render.py                       # uses defaults
    uv run python scripts/m1_render.py --ckpt checkpoints/my.pkl \
        --out docs/journal/assets/my.gif --num_frames 500
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


# Same jax.device_put_replicated shim as scripts/m1_train.py — brax
# 0.14.2 accesses this attribute during network construction on JAX
# 0.10 where it was removed.
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
# Match the training env config so rollout physics is identical to
# what the policy was optimized against.
ENV_OVERRIDES: dict = {"njmax": 96}


def load_params(ckpt_path: Path):
    """Load PPO params from either a pickle file (end-of-training output of
    our m1_train.py) or a brax/orbax checkpoint directory (mid-training
    snapshot created by save_checkpoint_path). Both return the same
    3-tuple shape brax expects: (normalizer, policy, value)."""
    if ckpt_path.is_dir():
        # Orbax rejects relative paths.
        params = brax_checkpoint.load(str(ckpt_path.resolve()))
        print(f"loaded orbax checkpoint dir {ckpt_path.name}")
        return params
    with ckpt_path.open("rb") as f:
        params = pickle.load(f)
    size_mb = ckpt_path.stat().st_size / 1024 / 1024
    print(f"loaded pickle {ckpt_path.name}: {size_mb:.1f} MB")
    return params


def build_inference_fn(params, env):
    """Reconstruct the brax PPO inference function from pickled params."""
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
        else lambda obs, *_: obs  # identity
    )
    net = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    make_inference_fn = ppo_networks.make_inference_fn(net)
    return jax.jit(make_inference_fn(params, deterministic=True))


def render(ckpt_path: Path, out_path: Path, num_frames: int, seed: int) -> None:
    params = load_params(ckpt_path)

    env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)
    inference_fn = build_inference_fn(params, env)

    mjcf = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf))
    mj_data = mujoco.MjData(mj_model)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.azimuth = 110.0
    cam.elevation = -12.0
    cam.lookat[:] = np.array([0.0, 0.0, 0.7])
    renderer = mujoco.Renderer(mj_model, height=360, width=480)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    state = reset_fn(rng)

    frames = []
    t0 = time.perf_counter()
    for _ in range(num_frames):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)

        mjx_data = getattr(state, "data", None) or getattr(state, "pipeline_state", None)
        if mjx_data is None:
            raise RuntimeError(
                "Cannot locate mjx.Data on state — expected state.data or "
                "state.pipeline_state."
            )
        mj_data.qpos[:] = np.array(mjx_data.qpos)
        mj_data.qvel[:] = np.array(mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=50, loop=0)
    wall = time.perf_counter() - t0
    abs_out = out_path.resolve()
    try:
        display_path = abs_out.relative_to(ROOT)
    except ValueError:
        display_path = abs_out
    print(
        f"rendered {num_frames} frames in {wall:.1f}s -> "
        f"{display_path} ({abs_out.stat().st_size // 1024} KB)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "m1-g1-smoke.pkl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "journal" / "assets" / "2026-04-16-m1-g1-smoke.gif",
    )
    ap.add_argument("--num_frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    render(args.ckpt, args.out, args.num_frames, args.seed)


if __name__ == "__main__":
    main()
