#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.2 — collect expert rollouts from the M1 PPO walker into the HDF5
format LeWM's ``stable_worldmodel.data.HDF5Dataset`` expects.

Orchestrator-worker design: the main invocation spawns one subprocess
per chunk of ``--chunk_size`` episodes. Each worker collects its
chunk into a temp HDF5 and exits cleanly, releasing all GPU memory.
The orchestrator then merges the per-chunk HDF5s into a single
output file. Subprocess isolation is required because Warp's VRAM
grows across env resets in a single process (we hit this first in
M0.4's batch sweep, and again in M1-full's 48.66 M-step crash).

Schema (matches tworoom.h5 / pusht_expert_train.h5):

    pixels     uint8   (N_total, H, W, 3)
    action     float32 (N_total, 29)
    proprio    float32 (N_total, 103)     # state.obs['state']
    state      float32 (N_total, nq+nv)   # raw qpos || qvel
    ep_len     int32   (N_episodes,)
    ep_offset  int64   (N_episodes,)      # cumulative offsets

Runs
----
    source scripts/env-setup.sh
    uv run python scripts/m4_collect_g1_rollouts.py

    # small smoke:
    uv run python scripts/m4_collect_g1_rollouts.py \
        --n_episodes 3 --episode_len 30 --img_size 96 \
        --out /tmp/g1_smoke.h5
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = "G1JoystickFlatTerrain"
ENV_OVERRIDES = {"njmax": 96}
G1_SCENE = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

# Same head-cam pose as M2 (scripts/m2_head_camera.py).
HEAD_CAM_POS = [0.10, 0.0, 0.18]
HEAD_CAM_QUAT = [-0.5, -0.5, 0.5, 0.5]


# --------------------------------------------------------------------------
# Orchestrator — spawns workers in subprocesses, merges chunks at the end
# --------------------------------------------------------------------------


def orchestrate(args: argparse.Namespace) -> None:
    assert args.chunk_size > 0
    assert args.n_episodes > 0
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    chunk_dir = out.parent / f".{out.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_files: list[Path] = []
    total_wall = 0.0
    failed_chunks = 0

    chunks_needed = (args.n_episodes + args.chunk_size - 1) // args.chunk_size
    print(
        f"Orchestrator: {args.n_episodes} episodes in {chunks_needed} chunks of "
        f"≤ {args.chunk_size}. Each chunk runs in its own subprocess to keep "
        f"Warp / JAX VRAM bounded.\n"
    )

    for chunk_id in range(chunks_needed):
        start_ep = chunk_id * args.chunk_size
        this_chunk = min(args.chunk_size, args.n_episodes - start_ep)
        chunk_file = chunk_dir / f"chunk_{chunk_id:03d}.h5"
        chunk_seed = args.seed + chunk_id * 1000

        print(
            f"[chunk {chunk_id + 1:>3}/{chunks_needed}] "
            f"episodes={this_chunk}  seed={chunk_seed}  out={chunk_file.name}"
        )
        t0 = time.perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--n_episodes",
                str(this_chunk),
                "--episode_len",
                str(args.episode_len),
                "--img_size",
                str(args.img_size),
                "--out",
                str(chunk_file),
                "--seed",
                str(chunk_seed),
                "--ckpt",
                str(args.ckpt),
            ],
            check=False,
        )
        wall = time.perf_counter() - t0
        total_wall += wall
        if result.returncode != 0:
            print(f"  FAILED (rc={result.returncode}) in {wall:.1f}s — skipping chunk")
            failed_chunks += 1
            if chunk_file.exists():
                chunk_file.unlink()
        else:
            print(f"  done in {wall:.1f}s")
            chunk_files.append(chunk_file)

    if not chunk_files:
        raise RuntimeError("All chunks failed; nothing to merge.")

    print(
        f"\nMerging {len(chunk_files)} chunk(s) "
        f"({failed_chunks} failed) into {out} ..."
    )
    merge_hdf5_chunks(chunk_files, out)
    for f in chunk_files:
        f.unlink()
    if not any(chunk_dir.iterdir()):
        chunk_dir.rmdir()

    size_mb = out.stat().st_size / 1024 / 1024
    print(
        f"M4.2 complete: orchestration wall {total_wall:.1f}s  "
        f"({args.n_episodes - failed_chunks * args.chunk_size} expected-good episodes)  "
        f"→ {out.name} ({size_mb:.1f} MB)"
    )


def merge_hdf5_chunks(chunk_files: list[Path], out: Path) -> None:
    """Concatenate per-chunk HDF5 files into a single dataset. Rewrites
    ep_offset as the global cumulative offset across all chunks."""
    import h5py
    import numpy as np

    # Read all chunks; concat arrays.
    pixels_list, action_list, proprio_list, state_list = [], [], [], []
    ep_len_list, attrs = [], {}
    for cf in chunk_files:
        with h5py.File(cf, "r") as f:
            pixels_list.append(f["pixels"][:])
            action_list.append(f["action"][:])
            proprio_list.append(f["proprio"][:])
            state_list.append(f["state"][:])
            ep_len_list.append(f["ep_len"][:])
            # Copy attrs from the first chunk; they match across chunks.
            if not attrs:
                attrs = dict(f.attrs)

    pixels = np.concatenate(pixels_list, axis=0)
    action = np.concatenate(action_list, axis=0)
    proprio = np.concatenate(proprio_list, axis=0)
    state = np.concatenate(state_list, axis=0)
    ep_len = np.concatenate(ep_len_list, axis=0).astype(np.int32)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)

    total_steps = int(ep_len.sum())
    assert pixels.shape[0] == total_steps, (
        f"pixels rows {pixels.shape[0]} != total ep_len {total_steps}"
    )

    with h5py.File(out, "w") as f:
        f.create_dataset(
            "pixels", data=pixels, compression="gzip", compression_opts=4, chunks=True
        )
        f.create_dataset("action", data=action)
        f.create_dataset("proprio", data=proprio)
        f.create_dataset("state", data=state)
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        for k, v in attrs.items():
            f.attrs[k] = v
        # Update attrs that summarize the merged file.
        f.attrs["total_steps"] = total_steps
        f.attrs["n_episodes"] = len(ep_len)


# --------------------------------------------------------------------------
# Worker — collects one chunk and writes it to the provided --out path
# --------------------------------------------------------------------------


def worker(args: argparse.Namespace) -> None:
    # Heavy imports deferred until we know we're the worker so the
    # orchestrator process doesn't pay the JAX/Warp startup cost.
    import functools

    import jax
    import jax.numpy as jnp

    # brax 0.14.2 × JAX 0.10 shim (same as other M*.py scripts).
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

    import h5py
    import mujoco
    import numpy as np
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    # --- build head-cam model ---
    spec = mujoco.MjSpec.from_file(str(G1_SCENE))
    torso = spec.body("torso_link")
    cam = torso.add_camera()
    cam.name = "head"
    cam.pos = HEAD_CAM_POS
    cam.quat = HEAD_CAM_QUAT
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    cam_id = mj_model.camera("head").id
    renderer = mujoco.Renderer(mj_model, height=args.img_size, width=args.img_size)

    # --- env + policy ---
    env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)
    proprio_dim = int(env.observation_size["state"][0])
    qpos_dim = mj_model.nq
    qvel_dim = mj_model.nv

    if args.ckpt.is_dir():
        params = brax_checkpoint.load(str(args.ckpt.resolve()))
    else:
        with args.ckpt.open("rb") as f:
            params = pickle.load(f)

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
    inference_fn = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Warm up JIT once so timings are accurate.
    rng = jax.random.PRNGKey(args.seed)
    rng, warm_key = jax.random.split(rng)
    warm = reset_fn(warm_key)
    rng, act_key = jax.random.split(rng)
    warm_action, _ = inference_fn(warm.obs, act_key)
    warm = step_fn(warm, warm_action)
    jax.block_until_ready(warm.obs["state"])

    # --- collect episodes ---
    pixels_eps: list[np.ndarray] = []
    actions_eps: list[np.ndarray] = []
    proprio_eps: list[np.ndarray] = []
    state_eps: list[np.ndarray] = []
    ep_lens: list[int] = []
    total_steps = 0
    t0 = time.perf_counter()

    for ep in range(args.n_episodes):
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)

        ep_pixels = np.zeros(
            (args.episode_len, args.img_size, args.img_size, 3), dtype=np.uint8
        )
        ep_actions = np.zeros((args.episode_len, env.action_size), dtype=np.float32)
        ep_proprio = np.zeros((args.episode_len, proprio_dim), dtype=np.float32)
        ep_state = np.zeros((args.episode_len, qpos_dim + qvel_dim), dtype=np.float32)

        actual_len = 0
        for t in range(args.episode_len):
            mjx_data = getattr(state, "data", None) or getattr(
                state, "pipeline_state", None
            )
            if mjx_data is None:
                raise RuntimeError("No mjx.Data on playground state")
            qpos = np.asarray(mjx_data.qpos)
            qvel = np.asarray(mjx_data.qvel)
            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = qvel
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam_id)
            ep_pixels[t] = renderer.render()

            rng, act_key = jax.random.split(rng)
            action, _ = inference_fn(state.obs, act_key)
            ep_actions[t] = np.asarray(action)
            ep_proprio[t] = np.asarray(state.obs["state"])
            ep_state[t, :qpos_dim] = qpos
            ep_state[t, qpos_dim:] = qvel
            actual_len = t + 1

            if bool(state.done):
                break
            state = step_fn(state, action)

        pixels_eps.append(ep_pixels[:actual_len])
        actions_eps.append(ep_actions[:actual_len])
        proprio_eps.append(ep_proprio[:actual_len])
        state_eps.append(ep_state[:actual_len])
        ep_lens.append(actual_len)
        total_steps += actual_len
        print(
            f"  [worker] ep {ep + 1:>3}/{args.n_episodes}  "
            f"len={actual_len:>3}  total={total_steps:>6}"
        )

    wall = time.perf_counter() - t0
    rate = total_steps / wall if wall > 0 else 0
    print(
        f"  [worker] {total_steps} steps in {wall:.1f}s ({rate:.1f} steps/s). "
        f"Early terms: {sum(1 for L in ep_lens if L < args.episode_len)}/{len(ep_lens)}"
    )

    # --- write HDF5 ---
    pixels = np.concatenate(pixels_eps, axis=0)
    actions = np.concatenate(actions_eps, axis=0)
    proprio = np.concatenate(proprio_eps, axis=0)
    state_arr = np.concatenate(state_eps, axis=0)
    ep_len_arr = np.array(ep_lens, dtype=np.int32)
    ep_offset_arr = np.concatenate(
        [[0], np.cumsum(ep_len_arr)[:-1]]
    ).astype(np.int64)

    with h5py.File(args.out, "w") as f:
        f.create_dataset(
            "pixels", data=pixels, compression="gzip", compression_opts=4, chunks=True
        )
        f.create_dataset("action", data=actions)
        f.create_dataset("proprio", data=proprio)
        f.create_dataset("state", data=state_arr)
        f.create_dataset("ep_len", data=ep_len_arr)
        f.create_dataset("ep_offset", data=ep_offset_arr)
        f.attrs["env_name"] = ENV_NAME
        f.attrs["ckpt_path"] = str(args.ckpt)
        f.attrs["img_size"] = args.img_size
        f.attrs["ctrl_dt"] = 0.02
        f.attrs["seed"] = args.seed

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"  [worker] wrote {args.out.name} ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "m1-g1-full" / "000043253760",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".stable_worldmodel" / "g1_joystick_expert.h5",
    )
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--episode_len", type=int, default=200)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help=(
            "Episodes per subprocess. We hit Warp VRAM OOM around episode 20 in "
            "a single process, so chunk into smaller units to keep each "
            "subprocess below the leak threshold."
        ),
    )
    ap.add_argument(
        "--worker",
        action="store_true",
        help="Internal: this invocation is a worker process. Users don't set this.",
    )
    args = ap.parse_args()

    if args.worker:
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
