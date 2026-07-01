# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""G1 rollout collector.

Subsumes the legacy ``scripts/m4_collect_g1_rollouts*.py`` orchestrator
+ worker pair. One HDF5 dataset is built per :class:`CollectionConfig`.

**Why subprocesses.** M4.2 confirmed Warp's CUDA allocator creeps past
~20 episodes per process, eventually OOMing on an 8 GB GPU. The fix
that's stuck: run each small chunk of episodes in a fresh subprocess so
Warp's VRAM releases between chunks. The
``subprocess.run([sys.executable, "-m", "vitruvian.data.collection",
"--worker", ...])`` contract keeps this isolation; a single-process
fallback is available for CI-sized configs.

**Why streaming merge.** The diverse dataset (270k transitions @
224×224 uint8) is ~40 GB. M4.2's merge concatenated everything in host
RAM and OOM'd a 32 GB machine at 14× scale. Two-pass streaming merge
(read shapes → preallocate → copy chunk-by-chunk via h5py slabs)
bounds host RAM to HDF5's own chunk cache.

HDF5 schema (identical to legacy so downstream caches/datasets don't
care which collector wrote the file)::

    pixels     uint8   (N_total, H, W, 3)
    action     float32 (N_total, 29)
    proprio    float32 (N_total, 103)
    state      float32 (N_total, nq+nv)
    ep_len     int32   (N_episodes,)
    ep_offset  int64   (N_episodes,)
    commands   float32 (N_episodes, 3)    # per-episode pinned cmd

Invoke via the CLI (``vit-collect configs/collect/diverse.yaml``) or
directly::

    from vitruvian.data import diverse_config, run_collection
    run_collection(diverse_config(Path("~/.stable_worldmodel/g1_diverse_v2.h5")))
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandSpec:
    """Single ``(vel_x, vel_y, yaw_rate)`` locomotion command."""

    vel_x: float
    vel_y: float
    yaw_rate: float


DEFAULT_DIVERSE_GRID: tuple[CommandSpec, ...] = tuple(
    CommandSpec(vx, vy, yr)
    for vx, vy, yr in product((-0.5, 0.0, 0.5), (-0.3, 0.0, 0.3), (-0.3, 0.0, 0.3))
)
NARROW_COMMAND: CommandSpec = CommandSpec(0.5, 0.0, 0.0)

DEFAULT_G1_POLICY_CKPT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "m1-g1-full"
    / "000043253760"
)


@dataclass
class CollectionConfig:
    """Full config for a single collection run.

    Attributes:
        out_h5: Destination HDF5 path.
        episodes_per_command: Rollouts per command in the grid.
        episode_steps: Primitive steps per rollout.
        commands: Grid of commands to sweep. For the legacy "narrow"
            single-command dataset, pass a one-element tuple.
        chunk_size: Episodes collected per subprocess worker (Warp VRAM
            isolation).
        seed: Base seed; per-chunk seed is ``seed + global_chunk_id *
            1000``.
        policy_ckpt: Path to the frozen PPO policy whose rollouts fill
            the dataset.
        img_size: Pixel render resolution.
        keep_chunks: If True, retain the per-chunk HDF5 files after
            merge (default deletes).
    """

    out_h5: Path
    episodes_per_command: int = 50
    episode_steps: int = 200
    commands: Sequence[CommandSpec] = field(
        default_factory=lambda: DEFAULT_DIVERSE_GRID
    )
    chunk_size: int = 10
    seed: int = 0
    policy_ckpt: Path = field(default_factory=lambda: DEFAULT_G1_POLICY_CKPT)
    img_size: int = 224
    keep_chunks: bool = False


def narrow_config(out_h5: Path, **overrides) -> CollectionConfig:
    """Config mirroring the original M4.2 single-command dataset."""
    return CollectionConfig(
        out_h5=out_h5,
        commands=(NARROW_COMMAND,),
        episodes_per_command=overrides.pop("episodes_per_command", 100),
        **overrides,
    )


def diverse_config(out_h5: Path, **overrides) -> CollectionConfig:
    """Config mirroring the M4.5 3×3×3 diverse grid (270k transitions)."""
    return CollectionConfig(
        out_h5=out_h5,
        commands=DEFAULT_DIVERSE_GRID,
        **overrides,
    )


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def run_collection(
    cfg: CollectionConfig,
    *,
    single_process: bool = False,
    allow_partial: bool = False,
    max_workers: int = 1,
) -> None:
    """Run the full collection described by ``cfg``.

    With ``single_process=False`` (default), each chunk of
    ``chunk_size`` episodes runs in a fresh subprocess so Warp's CUDA
    allocator releases between chunks. Use ``single_process=True`` only
    when the total episode count is small enough to fit in a single
    process (empirically < 20 eps on an 8 GB GPU).

    ``max_workers`` (default 1 = the historical serial behavior) runs up
    to that many chunk subprocesses CONCURRENTLY. Each chunk is already a
    GPU-isolated subprocess (Warp's allocator resets between chunks), so
    concurrency is bounded by GPU memory, not correctness — set it high
    on a big card (~16–24 on 96 GB), low on 8 GB (~2–3). This is the main
    collection speed lever: collection is otherwise a single serial
    per-step render loop, so wall-time scales ~1/workers. Ignored when
    ``single_process=True`` (in-process runs can't be GPU-isolated).

    With ``allow_partial=False`` (default), any chunk subprocess
    returning a non-zero exit code causes ``run_collection`` to raise
    after the merge step — the run still produces an HDF5 from the
    surviving chunks, but the failure is not silently swallowed.
    Pass ``allow_partial=True`` to accept a partial dataset without
    raising (useful when you're rerunning and only want to complete
    whatever works on this pass).
    """
    out = Path(cfg.out_h5).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    total_episodes = len(cfg.commands) * cfg.episodes_per_command
    chunks_per_combo = (
        cfg.episodes_per_command + cfg.chunk_size - 1
    ) // cfg.chunk_size

    print(
        f"[collect] {len(cfg.commands)} command(s) × "
        f"{cfg.episodes_per_command} episodes × "
        f"{cfg.episode_steps} steps = {total_episodes} episodes "
        f"→ {out}  ({'in-process' if single_process else 'subprocess'})"
    )

    chunk_dir = out.parent / f".{out.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Build the full chunk work-list, then execute it (serially, or up to
    # ``max_workers`` GPU-isolated subprocesses at once).
    specs: list[tuple[int, int, CommandSpec, Path, int, int]] = []
    gid = 0
    for cmd_idx, cmd in enumerate(cfg.commands):
        for chunk_idx in range(chunks_per_combo):
            start_ep = chunk_idx * cfg.chunk_size
            this_chunk = min(
                cfg.chunk_size, cfg.episodes_per_command - start_ep
            )
            if this_chunk <= 0:
                continue
            chunk_file = chunk_dir / (
                f"chunk_{gid:04d}_cmd{cmd_idx:02d}"
                f"_vx{cmd.vel_x:+.1f}_vy{cmd.vel_y:+.1f}"
                f"_yr{cmd.yaw_rate:+.1f}.h5"
            )
            specs.append(
                (gid, cmd_idx, cmd, chunk_file, cfg.seed + gid * 1000, this_chunk)
            )
            gid += 1

    n_chunks = len(specs)
    chunk_files: list[Path] = []
    failed_commands: list[tuple[int, CommandSpec]] = []
    failed = 0
    workers = 1 if single_process else max(1, int(max_workers))
    print(
        f"[collect] {n_chunks} chunk(s) × {cfg.chunk_size} eps  "
        f"({workers} concurrent worker(s))"
    )
    t_start = time.perf_counter()

    def _run_one(spec: tuple) -> tuple[tuple, int]:
        _gid, _cmd_idx, cmd, chunk_file, chunk_seed, this_chunk = spec
        if single_process:
            collect_chunk(
                n_episodes=this_chunk, episode_steps=cfg.episode_steps,
                img_size=cfg.img_size, out=chunk_file, seed=chunk_seed,
                policy_ckpt=cfg.policy_ckpt, command=cmd,
            )
            return spec, 0
        rc = _spawn_worker(
            n_episodes=this_chunk, episode_steps=cfg.episode_steps,
            img_size=cfg.img_size, out=chunk_file, seed=chunk_seed,
            policy_ckpt=cfg.policy_ckpt, command=cmd,
        )
        return spec, rc

    def _handle(spec: tuple, rc: int) -> None:
        nonlocal failed
        _gid, _cmd_idx, cmd, chunk_file, _seed, _n = spec
        if rc != 0:
            print(f"  [chunk {_gid + 1}/{n_chunks}] FAILED (rc={rc}) — skipping")
            failed += 1
            failed_commands.append((_gid, cmd))
            if chunk_file.exists():
                chunk_file.unlink()
        else:
            print(f"  [chunk {_gid + 1}/{n_chunks}] done")
            chunk_files.append(chunk_file)

    if workers == 1:
        for spec in specs:
            _handle(*_run_one(spec))
    else:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, s) for s in specs]
            for fut in concurrent.futures.as_completed(futs):
                _handle(*fut.result())
    total_wall = time.perf_counter() - t_start

    if not chunk_files:
        raise RuntimeError("All chunks failed; nothing to merge.")

    print(
        f"[merge] {len(chunk_files)} chunk(s) ({failed} failed) "
        f"→ {out} via streaming writes"
    )
    merge_hdf5_chunks_streaming(chunk_files, out)

    if not cfg.keep_chunks:
        for f in chunk_files:
            f.unlink()
        if chunk_dir.exists() and not any(chunk_dir.iterdir()):
            chunk_dir.rmdir()

    size_gb = out.stat().st_size / 1024**3
    print(
        f"[collect] done: wall {total_wall:.1f}s  "
        f"→ {out.name} ({size_gb:.2f} GB)"
    )

    if failed > 0 and not allow_partial:
        cmd_summary = ", ".join(
            f"[chunk {cid}: vx={c.vel_x:+.1f},vy={c.vel_y:+.1f},yr={c.yaw_rate:+.1f}]"
            for cid, c in failed_commands
        )
        raise RuntimeError(
            f"collect: {failed}/{failed + len(chunk_files)} chunks failed. "
            f"Dataset at {out} is partial. Failed: {cmd_summary}. "
            f"Rerun the collection to retry, or pass allow_partial=True to "
            f"accept this as a partial dataset."
        )


def _spawn_worker(
    *,
    n_episodes: int,
    episode_steps: int,
    img_size: int,
    out: Path,
    seed: int,
    policy_ckpt: Path,
    command: CommandSpec,
) -> int:
    """Launch a worker subprocess via ``python -m vitruvian.data.collection
    --worker ...``. Returns the exit code."""
    args = [
        sys.executable,
        "-m",
        "vitruvian.data.collection",
        "--worker",
        "--n-episodes",
        str(n_episodes),
        "--episode-steps",
        str(episode_steps),
        "--img-size",
        str(img_size),
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--policy-ckpt",
        str(policy_ckpt),
        "--vel-x",
        str(command.vel_x),
        "--vel-y",
        str(command.vel_y),
        "--yaw-rate",
        str(command.yaw_rate),
    ]
    return subprocess.run(args, check=False).returncode


# --------------------------------------------------------------------------
# Streaming merge
# --------------------------------------------------------------------------


def merge_hdf5_chunks_streaming(
    chunk_files: list[Path], out: Path
) -> None:
    """Two-pass streaming merge. Pass 1: read per-chunk shapes/metadata.
    Pass 2: preallocate output datasets, copy one whole chunk file at a
    time into the output slab.

    Host RAM stays bounded by the largest single chunk file (each chunk
    is read via ``fin["pixels"][:]`` and copied straight into the output
    dataset), not by the full merged size — essential for the 40 GB
    diverse set, which would OOM a 32 GB box if concatenated in RAM.
    Keep ``chunk_size`` modest so a chunk file fits comfortably.
    """
    import h5py
    import numpy as np

    total_rows = 0
    ep_len_list: list[np.ndarray] = []
    commands_list: list[np.ndarray] = []
    shape_probe: dict = {}
    first_attrs: dict = {}

    for cf in chunk_files:
        with h5py.File(cf, "r") as f:
            total_rows += int(f["pixels"].shape[0])
            ep_len_list.append(f["ep_len"][:].astype(np.int32))
            if "commands" in f:
                commands_list.append(f["commands"][:].astype(np.float32))
            else:
                commands_list.append(
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
    ep_offset_all = np.concatenate(
        [[0], np.cumsum(ep_len_all)[:-1]]
    ).astype(np.int64)
    commands_all = np.concatenate(commands_list, axis=0).astype(np.float32)

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
            "action",
            shape=(total_rows, shape_probe["action_dim"]),
            dtype="float32",
        )
        proprio_ds = fout.create_dataset(
            "proprio",
            shape=(total_rows, shape_probe["proprio_dim"]),
            dtype="float32",
        )
        state_ds = fout.create_dataset(
            "state",
            shape=(total_rows, shape_probe["state_dim"]),
            dtype="float32",
        )
        fout.create_dataset("ep_len", data=ep_len_all)
        fout.create_dataset("ep_offset", data=ep_offset_all)
        fout.create_dataset("commands", data=commands_all)
        for k, v in first_attrs.items():
            fout.attrs[k] = v
        fout.attrs["total_steps"] = total_rows
        fout.attrs["n_episodes"] = int(ep_len_all.shape[0])
        fout.attrs["collection_mode"] = (
            "diverse-command-grid"
            if len(commands_list) > 1 or commands_all.shape[0] > 1
            else "narrow"
        )

        cursor = 0
        for cf_idx, cf in enumerate(chunk_files):
            with h5py.File(cf, "r") as fin:
                rows = int(fin["pixels"].shape[0])
                pixels_ds[cursor : cursor + rows] = fin["pixels"][:]
                action_ds[cursor : cursor + rows] = fin["action"][:]
                proprio_ds[cursor : cursor + rows] = fin["proprio"][:]
                state_ds[cursor : cursor + rows] = fin["state"][:]
            cursor += rows
            if (cf_idx + 1) % 10 == 0:
                print(
                    f"  merged {cf_idx + 1}/{len(chunk_files)} chunks "
                    f"({cursor}/{total_rows} rows)"
                )


# --------------------------------------------------------------------------
# Worker — runs inside a subprocess (or in-process when single_process=True)
# --------------------------------------------------------------------------


def collect_chunk(
    *,
    n_episodes: int,
    episode_steps: int,
    img_size: int,
    out: Path,
    seed: int,
    policy_ckpt: Path,
    command: CommandSpec,
) -> None:
    """Collect ``n_episodes`` rollouts into ``out`` HDF5. Pins the
    joystick command to ``command`` after every reset + step so the env
    doesn't randomize away from the grid point we want."""
    import functools
    import pickle

    import h5py
    import mujoco
    import numpy as np

    from vitruvian.env import install_jax_brax_shim
    from vitruvian.env.cameras import HEAD_CAM_POS, HEAD_CAM_QUAT

    install_jax_brax_shim()

    import jax
    import jax.numpy as jnp
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    repo_root = Path(__file__).resolve().parents[3]
    g1_scene = (
        repo_root / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
    )

    spec = mujoco.MjSpec.from_file(str(g1_scene))
    torso = spec.body("torso_link")
    cam = torso.add_camera()
    cam.name = "head"
    cam.pos = HEAD_CAM_POS
    cam.quat = HEAD_CAM_QUAT
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    cam_id = mj_model.camera("head").id
    renderer = mujoco.Renderer(mj_model, height=img_size, width=img_size)

    env_name = "G1JoystickFlatTerrain"
    env = registry.load(env_name, config_overrides={"njmax": 96})
    proprio_dim = int(env.observation_size["state"][0])
    qpos_dim = mj_model.nq
    qvel_dim = mj_model.nv

    if policy_ckpt.is_dir():
        params = brax_checkpoint.load(str(policy_ckpt.resolve()))
    else:
        with policy_ckpt.open("rb") as f:
            params = pickle.load(f)

    ppo_cfg = locomotion_params.brax_ppo_config(env_name)
    factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(
            ppo_cfg.network_factory.policy_hidden_layer_sizes
        ),
        value_hidden_layer_sizes=tuple(
            ppo_cfg.network_factory.value_hidden_layer_sizes
        ),
        policy_obs_key=ppo_cfg.network_factory.policy_obs_key,
        value_obs_key=ppo_cfg.network_factory.value_obs_key,
    )
    preprocess_fn = (
        running_statistics.normalize
        if ppo_cfg.normalize_observations
        else lambda obs, *_: obs
    )
    net = factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_fn,
    )
    inference_fn = jax.jit(
        ppo_networks.make_inference_fn(net)(params, deterministic=True)
    )
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    pinned_cmd = jnp.asarray(
        [command.vel_x, command.vel_y, command.yaw_rate], dtype=jnp.float32
    )

    def _pin(state):
        return state.replace(info={**state.info, "command": pinned_cmd})

    rng = jax.random.PRNGKey(seed)
    rng, warm_key = jax.random.split(rng)
    warm = _pin(reset_fn(warm_key))
    rng, act_key = jax.random.split(rng)
    warm_action, _ = inference_fn(warm.obs, act_key)
    warm = _pin(step_fn(warm, warm_action))
    jax.block_until_ready(warm.obs["state"])

    pixels_eps: list[np.ndarray] = []
    actions_eps: list[np.ndarray] = []
    proprio_eps: list[np.ndarray] = []
    state_eps: list[np.ndarray] = []
    ep_lens: list[int] = []
    total_steps = 0
    t0 = time.perf_counter()

    for _ep in range(n_episodes):
        rng, reset_key = jax.random.split(rng)
        state = _pin(reset_fn(reset_key))

        ep_pixels = np.zeros(
            (episode_steps, img_size, img_size, 3), dtype=np.uint8
        )
        ep_actions = np.zeros((episode_steps, env.action_size), dtype=np.float32)
        ep_proprio = np.zeros((episode_steps, proprio_dim), dtype=np.float32)
        ep_state = np.zeros(
            (episode_steps, qpos_dim + qvel_dim), dtype=np.float32
        )

        actual_len = 0
        for t in range(episode_steps):
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
            state = _pin(step_fn(state, action))

        pixels_eps.append(ep_pixels[:actual_len])
        actions_eps.append(ep_actions[:actual_len])
        proprio_eps.append(ep_proprio[:actual_len])
        state_eps.append(ep_state[:actual_len])
        ep_lens.append(actual_len)
        total_steps += actual_len

    wall = time.perf_counter() - t0
    rate = total_steps / wall if wall > 0 else 0
    print(
        f"  [worker cmd=({command.vel_x:+.1f},{command.vel_y:+.1f},"
        f"{command.yaw_rate:+.1f})] "
        f"{total_steps} steps in {wall:.1f}s ({rate:.1f} steps/s). "
        f"Early terms: "
        f"{sum(1 for L in ep_lens if L < episode_steps)}/{len(ep_lens)}"
    )

    pixels = np.concatenate(pixels_eps, axis=0)
    actions = np.concatenate(actions_eps, axis=0)
    proprio = np.concatenate(proprio_eps, axis=0)
    state_arr = np.concatenate(state_eps, axis=0)
    ep_len_arr = np.array(ep_lens, dtype=np.int32)
    ep_offset_arr = np.concatenate(
        [[0], np.cumsum(ep_len_arr)[:-1]]
    ).astype(np.int64)
    commands_arr = np.tile(
        np.asarray(
            [command.vel_x, command.vel_y, command.yaw_rate], dtype=np.float32
        ),
        (len(ep_lens), 1),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset(
            "pixels",
            data=pixels,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )
        f.create_dataset("action", data=actions)
        f.create_dataset("proprio", data=proprio)
        f.create_dataset("state", data=state_arr)
        f.create_dataset("ep_len", data=ep_len_arr)
        f.create_dataset("ep_offset", data=ep_offset_arr)
        f.create_dataset("commands", data=commands_arr)
        f.attrs["env_name"] = env_name
        f.attrs["policy_ckpt"] = str(policy_ckpt)
        f.attrs["img_size"] = img_size
        f.attrs["ctrl_dt"] = 0.02
        f.attrs["seed"] = seed
        f.attrs["vel_x"] = command.vel_x
        f.attrs["vel_y"] = command.vel_y
        f.attrs["yaw_rate"] = command.yaw_rate


# --------------------------------------------------------------------------
# python -m vitruvian.data.collection --worker ...
# --------------------------------------------------------------------------


def _worker_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--n-episodes", type=int, required=True)
    ap.add_argument("--episode-steps", type=int, required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--policy-ckpt", type=Path, required=True)
    ap.add_argument("--vel-x", type=float, required=True)
    ap.add_argument("--vel-y", type=float, required=True)
    ap.add_argument("--yaw-rate", type=float, required=True)
    args = ap.parse_args()

    collect_chunk(
        n_episodes=args.n_episodes,
        episode_steps=args.episode_steps,
        img_size=args.img_size,
        out=args.out,
        seed=args.seed,
        policy_ckpt=args.policy_ckpt,
        command=CommandSpec(args.vel_x, args.vel_y, args.yaw_rate),
    )


if __name__ == "__main__":
    _worker_main()


__all__ = [
    "CollectionConfig",
    "CommandSpec",
    "DEFAULT_DIVERSE_GRID",
    "NARROW_COMMAND",
    "collect_chunk",
    "diverse_config",
    "merge_hdf5_chunks_streaming",
    "narrow_config",
    "run_collection",
]
