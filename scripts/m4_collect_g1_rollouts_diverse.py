#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.5 — collect a DIVERSE G1 expert dataset by sweeping the joystick
command grid. Derivative of M4.2's ``m4_collect_g1_rollouts.py`` with
two behavioral changes:

    1. Each worker runs a FIXED command (vel_x, vel_y, yaw_rate)
       throughout its episodes. The command is pinned via
       ``state.replace(info={**state.info, "command": ...})`` after
       every step, overriding the env's random resampling.
    2. The orchestrator enumerates a 3×3×3 command grid and allocates
       ~50 episodes per combo → 1350 episodes / 270k transitions
       (14× the original M4.2 dataset).

Also rewrites ``merge_hdf5_chunks`` to stream chunk-to-output without
holding the full 40 GB pixel array in host RAM — M4.2's merge would
OOM a 32 GB host on the new dataset size.

HDF5 schema (identical to M4.2 so everything downstream still works):

    pixels     uint8   (N_total, H, W, 3)
    action     float32 (N_total, 29)
    proprio    float32 (N_total, 103)
    state      float32 (N_total, nq+nv)
    ep_len     int32   (N_episodes,)
    ep_offset  int64   (N_episodes,)
    commands   float32 (N_episodes, 3)   # NEW: pinned cmd per episode

Runs
----
    # full dataset (~1-3 h wall; 27 combos × 50 eps × 200 steps):
    uv run python scripts/m4_collect_g1_rollouts_diverse.py

    # smoke:
    uv run python scripts/m4_collect_g1_rollouts_diverse.py \\
        --n_eps_per_combo 2 --episode_len 30 --out /tmp/g1_diverse_smoke.h5
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import pickle
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = "G1JoystickFlatTerrain"
ENV_OVERRIDES = {"njmax": 96}
G1_SCENE = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

# Head-cam pose — identical to M4.2 and m4c so embeddings compare.
HEAD_CAM_POS = [0.10, 0.0, 0.18]
HEAD_CAM_QUAT = [-0.5, -0.5, 0.5, 0.5]

# Command grid: 3^3 = 27 combinations.
VEL_X_GRID = [-0.5, 0.0, 0.5]
VEL_Y_GRID = [-0.3, 0.0, 0.3]
YAW_GRID = [-0.3, 0.0, 0.3]


def iter_command_grid() -> list[tuple[float, float, float]]:
    return [
        (vx, vy, yr) for vx, vy, yr in product(VEL_X_GRID, VEL_Y_GRID, YAW_GRID)
    ]


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def orchestrate(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    commands = iter_command_grid()
    total_episodes = len(commands) * args.n_eps_per_combo
    chunks_per_combo = (args.n_eps_per_combo + args.chunk_size - 1) // args.chunk_size

    chunk_dir = out.parent / f".{out.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Orchestrator: {len(commands)} commands × {args.n_eps_per_combo} "
        f"episodes = {total_episodes} total episodes "
        f"({chunks_per_combo} chunks/combo × {len(commands)} combos "
        f"= {chunks_per_combo * len(commands)} subprocesses).\n"
    )

    chunk_files: list[Path] = []
    failed_chunks = 0
    total_wall = 0.0
    global_chunk_id = 0

    for cmd_idx, (vx, vy, yr) in enumerate(commands):
        for chunk_idx in range(chunks_per_combo):
            start_ep = chunk_idx * args.chunk_size
            this_chunk = min(args.chunk_size, args.n_eps_per_combo - start_ep)
            if this_chunk <= 0:
                continue

            chunk_file = chunk_dir / (
                f"chunk_{global_chunk_id:04d}_cmd{cmd_idx:02d}"
                f"_vx{vx:+.1f}_vy{vy:+.1f}_yr{yr:+.1f}.h5"
            )
            chunk_seed = args.seed + global_chunk_id * 1000

            print(
                f"[chunk {global_chunk_id + 1:>4}/"
                f"{chunks_per_combo * len(commands)}] "
                f"cmd=({vx:+.2f},{vy:+.2f},{yr:+.2f})  "
                f"eps={this_chunk}  seed={chunk_seed}"
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
                    "--vel_x",
                    str(vx),
                    "--vel_y",
                    str(vy),
                    "--yaw_rate",
                    str(yr),
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
            global_chunk_id += 1

    if not chunk_files:
        raise RuntimeError("All chunks failed; nothing to merge.")

    print(
        f"\nMerging {len(chunk_files)} chunk(s) ({failed_chunks} failed) "
        f"into {out} via streaming writes ..."
    )
    merge_hdf5_chunks_streaming(chunk_files, out)
    if args.keep_chunks:
        print(f"Chunk files retained at {chunk_dir}")
    else:
        for f in chunk_files:
            f.unlink()
        if not any(chunk_dir.iterdir()):
            chunk_dir.rmdir()

    size_gb = out.stat().st_size / 1024**3
    print(
        f"\nM4.5 diverse collection complete: wall {total_wall:.1f}s  "
        f"→ {out.name} ({size_gb:.2f} GB)"
    )


def merge_hdf5_chunks_streaming(chunk_files: list[Path], out: Path) -> None:
    """Stream chunk HDF5 files into a single output file without ever
    holding the full pixel array in RAM. Two passes:
      1. Read per-chunk `ep_len` to compute total row count and layout.
      2. Pre-allocate output datasets, then copy rows chunk-by-chunk.

    This is the difference between M4.2's 2 GB dataset (fits in RAM) and
    M4.5's 40 GB (does not).
    """
    import h5py
    import numpy as np

    # Pass 1: shapes + metadata.
    total_rows = 0
    episodes_per_chunk: list[int] = []
    ep_len_list: list[np.ndarray] = []
    commands_per_chunk: list[np.ndarray] = []
    first_attrs: dict = {}
    shape_probe: dict = {}

    for cf in chunk_files:
        with h5py.File(cf, "r") as f:
            chunk_rows = int(f["pixels"].shape[0])
            total_rows += chunk_rows
            episodes_per_chunk.append(int(f["ep_len"].shape[0]))
            ep_len_list.append(f["ep_len"][:].astype(np.int32))
            if "commands" in f:
                commands_per_chunk.append(f["commands"][:].astype(np.float32))
            else:
                # back-fill zeros if a legacy chunk lacks commands
                commands_per_chunk.append(
                    np.zeros((int(f["ep_len"].shape[0]), 3), dtype=np.float32)
                )
            if not shape_probe:
                shape_probe = {
                    "pixels": f["pixels"].shape[1:],
                    "pixels_dtype": f["pixels"].dtype,
                    "action_dim": int(f["action"].shape[1]),
                    "proprio_dim": int(f["proprio"].shape[1]),
                    "state_dim": int(f["state"].shape[1]),
                }
                first_attrs = dict(f.attrs)

    ep_len_all = np.concatenate(ep_len_list, axis=0).astype(np.int32)
    ep_offset_all = np.concatenate([[0], np.cumsum(ep_len_all)[:-1]]).astype(np.int64)
    commands_all = np.concatenate(commands_per_chunk, axis=0).astype(np.float32)
    n_episodes_total = int(ep_len_all.shape[0])

    # Pass 2: preallocate output datasets, stream copies.
    with h5py.File(out, "w") as fout:
        pixels_ds = fout.create_dataset(
            "pixels",
            shape=(total_rows, *shape_probe["pixels"]),
            dtype=shape_probe["pixels_dtype"],
            compression="gzip",
            compression_opts=4,
            chunks=(min(256, total_rows), *shape_probe["pixels"]),
        )
        action_ds = fout.create_dataset(
            "action", shape=(total_rows, shape_probe["action_dim"]), dtype="float32"
        )
        proprio_ds = fout.create_dataset(
            "proprio", shape=(total_rows, shape_probe["proprio_dim"]), dtype="float32"
        )
        state_ds = fout.create_dataset(
            "state", shape=(total_rows, shape_probe["state_dim"]), dtype="float32"
        )
        fout.create_dataset("ep_len", data=ep_len_all)
        fout.create_dataset("ep_offset", data=ep_offset_all)
        fout.create_dataset("commands", data=commands_all)
        for k, v in first_attrs.items():
            fout.attrs[k] = v
        fout.attrs["total_steps"] = total_rows
        fout.attrs["n_episodes"] = n_episodes_total
        fout.attrs["collection_mode"] = "diverse-command-grid"

        cursor = 0
        for cf_idx, cf in enumerate(chunk_files):
            with h5py.File(cf, "r") as fin:
                rows = int(fin["pixels"].shape[0])
                # Copy in-place using h5py dataset slicing. Each slab is
                # already chunked; host RAM footprint is bounded by HDF5's
                # own chunk cache, not the slab size.
                pixels_ds[cursor : cursor + rows] = fin["pixels"][:]
                action_ds[cursor : cursor + rows] = fin["action"][:]
                proprio_ds[cursor : cursor + rows] = fin["proprio"][:]
                state_ds[cursor : cursor + rows] = fin["state"][:]
            cursor += rows
            if (cf_idx + 1) % 10 == 0:
                print(
                    f"  merged {cf_idx + 1}/{len(chunk_files)} chunks "
                    f"({cursor} / {total_rows} rows)"
                )


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def worker(args: argparse.Namespace) -> None:
    import functools

    import jax
    import jax.numpy as jnp

    # brax 0.14.2 × JAX 0.10 shim.
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

    # --- head-cam model ---
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

    pinned_cmd = jnp.asarray([args.vel_x, args.vel_y, args.yaw_rate], dtype=jnp.float32)

    def _pin_cmd(state):
        return state.replace(info={**state.info, "command": pinned_cmd})

    # Warm up JIT.
    rng = jax.random.PRNGKey(args.seed)
    rng, warm_key = jax.random.split(rng)
    warm = _pin_cmd(reset_fn(warm_key))
    rng, act_key = jax.random.split(rng)
    warm_action, _ = inference_fn(warm.obs, act_key)
    warm = _pin_cmd(step_fn(warm, warm_action))
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
        state = _pin_cmd(reset_fn(reset_key))

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
            state = _pin_cmd(step_fn(state, action))

        pixels_eps.append(ep_pixels[:actual_len])
        actions_eps.append(ep_actions[:actual_len])
        proprio_eps.append(ep_proprio[:actual_len])
        state_eps.append(ep_state[:actual_len])
        ep_lens.append(actual_len)
        total_steps += actual_len

    wall = time.perf_counter() - t0
    rate = total_steps / wall if wall > 0 else 0
    print(
        f"  [worker cmd=({args.vel_x:+.1f},{args.vel_y:+.1f},{args.yaw_rate:+.1f})] "
        f"{total_steps} steps in {wall:.1f}s ({rate:.1f} steps/s). "
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
    commands_arr = np.tile(
        np.asarray([args.vel_x, args.vel_y, args.yaw_rate], dtype=np.float32),
        (len(ep_lens), 1),
    )

    with h5py.File(args.out, "w") as f:
        f.create_dataset(
            "pixels", data=pixels, compression="gzip", compression_opts=4, chunks=True
        )
        f.create_dataset("action", data=actions)
        f.create_dataset("proprio", data=proprio)
        f.create_dataset("state", data=state_arr)
        f.create_dataset("ep_len", data=ep_len_arr)
        f.create_dataset("ep_offset", data=ep_offset_arr)
        f.create_dataset("commands", data=commands_arr)
        f.attrs["env_name"] = ENV_NAME
        f.attrs["ckpt_path"] = str(args.ckpt)
        f.attrs["img_size"] = args.img_size
        f.attrs["ctrl_dt"] = 0.02
        f.attrs["seed"] = args.seed
        f.attrs["vel_x"] = args.vel_x
        f.attrs["vel_y"] = args.vel_y
        f.attrs["yaw_rate"] = args.yaw_rate


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
        default=Path.home() / ".stable_worldmodel" / "g1_diverse_v1.h5",
    )
    ap.add_argument(
        "--n_eps_per_combo",
        type=int,
        default=50,
        help="Episodes per (vel_x, vel_y, yaw_rate) combination. 27 combos × 50 "
        "= 1350 total episodes at default.",
    )
    ap.add_argument("--episode_len", type=int, default=200)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Episodes per subprocess. Warp VRAM creep past ~20 eps/process "
        "forces chunking.",
    )
    ap.add_argument(
        "--keep_chunks",
        action="store_true",
        help="Retain per-chunk HDF5 files after merging (default: delete).",
    )
    # Worker-only flags.
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--n_episodes", type=int, default=0, help="(worker) episodes in this chunk")
    ap.add_argument("--vel_x", type=float, default=0.0)
    ap.add_argument("--vel_y", type=float, default=0.0)
    ap.add_argument("--yaw_rate", type=float, default=0.0)
    args = ap.parse_args()

    if args.worker:
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
