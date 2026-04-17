#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M1 — train a PPO walker on G1JoystickFlatTerrain.

Option A (smoke): 1024 envs, 20 M timesteps, ~13 min on an RTX 4060 Ti.
Logs to wandb (project "vitruvian"), saves final params as a pickle to
checkpoints/, and renders a rollout GIF to docs/journal/assets/.

Runs
----
    source scripts/env-setup.sh
    uv run python scripts/m1_train.py

Optional flags: --num_envs, --num_timesteps, --seed, --run_name,
--no_wandb, --no_render.
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


def _device_put_replicated_shim(value, devices):
    """Shim for jax.device_put_replicated, removed in JAX 0.10 but still
    called by brax 0.14.2. Prepends a leading axis of size len(devices)
    to every leaf so brax's _unpmap(...).squeeze(0) survives. Single-GPU
    is our only case; multi-device would broadcast along axis 0."""
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

import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
ASSETS = ROOT / "docs" / "journal" / "assets"
ENV_NAME = "G1JoystickFlatTerrain"


def make_network_factory(cfg):
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.network_factory.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_factory.value_hidden_layer_sizes),
        policy_obs_key=cfg.network_factory.policy_obs_key,
        value_obs_key=cfg.network_factory.value_obs_key,
    )


def make_progress_fn(start_time: float, use_wandb: bool):
    import wandb

    def progress(num_steps: int, metrics: dict) -> None:
        wall = time.time() - start_time
        loggable = {"train/wall_seconds": wall}
        for k, v in metrics.items():
            try:
                loggable[k] = float(v)
            except (TypeError, ValueError):
                continue
        if use_wandb:
            wandb.log(loggable, step=num_steps)
        sps = num_steps / wall if wall > 0 else 0.0
        reward = metrics.get("eval/episode_reward", None)
        reward_str = f"{float(reward):>7.2f}" if reward is not None else "    ?"
        print(
            f"[{wall:7.1f}s] step {num_steps:>11} "
            f"sps={sps:>7.0f}  eval_reward={reward_str}"
        )

    return progress


def render_rollout(
    params,
    make_inference_fn,
    out_gif: Path,
    num_frames: int = 500,
    seed: int = 0,
) -> None:
    """Roll the trained policy in the MJX/Warp env, render each step via a
    classic MuJoCo Renderer driven off extracted qpos/qvel."""
    import imageio.v2 as imageio
    import mujoco

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

    env = registry.load(ENV_NAME)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(seed)
    state = reset_fn(rng)

    frames = []
    for _ in range(num_frames):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)

        # The mjx.Data lives inside state.data or state.pipeline_state
        # depending on playground version. Handle both.
        mjx_data = getattr(state, "data", None) or getattr(state, "pipeline_state", None)
        if mjx_data is None:
            raise RuntimeError(
                "Cannot locate mjx.Data on playground state — expected "
                "state.data or state.pipeline_state."
            )
        mj_data.qpos[:] = np.array(mjx_data.qpos)
        mj_data.qvel[:] = np.array(mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())

    imageio.mimsave(str(out_gif), frames, fps=50, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=1024)
    ap.add_argument("--num_timesteps", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", type=str, default="m1-g1-smoke")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    cfg = locomotion_params.brax_ppo_config(ENV_NAME)
    cfg.num_envs = args.num_envs
    cfg.num_timesteps = args.num_timesteps

    print(f"--- M1 training — {args.run_name} ---")
    print(f"env:           {ENV_NAME}")
    print(f"num_envs:      {cfg.num_envs}  (default was {8192})")
    print(f"num_timesteps: {cfg.num_timesteps:,}")
    print(f"seed:          {args.seed}")
    print(f"wandb:         {'off' if args.no_wandb else 'on'}")
    print()

    # Per upstream playground train_jax_ppo.py: pass unwrapped envs and
    # let brax call wrap_env_fn at the right point in its wrapping stack.
    env = registry.load(ENV_NAME)
    eval_env = registry.load(ENV_NAME)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project="vitruvian",
            name=args.run_name,
            tags=["m1", "g1", "ppo", "warp", "smoke"],
            config={
                "env_name": ENV_NAME,
                "num_envs": cfg.num_envs,
                "num_timesteps": cfg.num_timesteps,
                "seed": args.seed,
                "learning_rate": float(cfg.learning_rate),
                "entropy_cost": float(cfg.entropy_cost),
                "discounting": float(cfg.discounting),
                "clipping_epsilon": float(cfg.clipping_epsilon),
                "batch_size": int(cfg.batch_size),
                "num_minibatches": int(cfg.num_minibatches),
                "unroll_length": int(cfg.unroll_length),
                "num_updates_per_batch": int(cfg.num_updates_per_batch),
                "episode_length": int(cfg.episode_length),
                "normalize_observations": bool(cfg.normalize_observations),
            },
        )

    start = time.time()
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=cfg.num_timesteps,
        num_envs=cfg.num_envs,
        episode_length=cfg.episode_length,
        unroll_length=cfg.unroll_length,
        num_updates_per_batch=cfg.num_updates_per_batch,
        num_minibatches=cfg.num_minibatches,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        discounting=cfg.discounting,
        reward_scaling=cfg.reward_scaling,
        clipping_epsilon=cfg.clipping_epsilon,
        action_repeat=cfg.action_repeat,
        num_evals=cfg.num_evals,
        network_factory=make_network_factory(cfg),
        max_grad_norm=cfg.max_grad_norm,
        normalize_observations=cfg.normalize_observations,
        num_resets_per_eval=cfg.num_resets_per_eval,
        progress_fn=make_progress_fn(start, use_wandb=use_wandb),
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    total = time.time() - start
    print(f"\n=== training done in {total:.1f}s ({total / 60:.1f} min) ===")

    # Save the final normalizer state + policy params.
    ckpt = CKPT_DIR / f"{args.run_name}.pkl"
    with ckpt.open("wb") as f:
        pickle.dump(params, f)
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    print(f"checkpoint -> {ckpt.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    if use_wandb:
        import wandb

        wandb.summary["wall_seconds_total"] = total
        wandb.summary["ckpt_path"] = str(ckpt.relative_to(ROOT))

    if not args.no_render:
        gif = ASSETS / f"2026-04-16-{args.run_name}.gif"
        print(f"\nrendering 10 s rollout -> {gif.relative_to(ROOT)} ...")
        t0 = time.time()
        render_rollout(params, make_inference_fn, gif, num_frames=500, seed=args.seed)
        print(f"  render wall: {time.time() - t0:.1f}s  size: {gif.stat().st_size // 1024} KB")
        if use_wandb:
            import wandb

            wandb.log({"rollout_gif": wandb.Video(str(gif))})

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
