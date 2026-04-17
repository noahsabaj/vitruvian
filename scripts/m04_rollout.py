#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M0.4 rollout — instantiate the mujoco_playground G1JoystickFlatTerrain
environment, roll out a random policy, and measure parallel-env
throughput on the RTX 4060 Ti.

Runs (orchestrator mode — default)
----------------------------------
    source scripts/env-setup.sh
    uv run python scripts/m04_rollout.py

    The orchestrator spawns one subprocess per batch size (64, 256, 512,
    1024, 2048) so each test gets a clean VRAM state. Writes a summary
    to docs/journal/assets/2026-04-16-m04-bench.txt.

Runs (bench worker — called from orchestrator)
----------------------------------------------
    uv run python scripts/m04_rollout.py --bench N

    Load the env, warm up, time 200 random-action steps at batch size
    N, print one line of `NAME N STEPS WALL KENVSTEPS_PER_S`.
"""

# Share the 8 GB card between JAX and Warp: disable JAX pre-allocation
# so both allocators can grow on demand. Must be set before importing
# jax. (No-op in orchestrator mode since we don't import jax there.)
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "journal" / "assets" / "2026-04-16-m04-bench.txt"
BATCHES = [64, 256, 512, 1024, 2048]


def bench_one(n_envs: int) -> None:
    """Worker: load env, warm up, time 200 steps at batch size n_envs,
    print one summary line and exit. Called as a subprocess."""
    import jax
    import jax.numpy as jnp
    from mujoco_playground import registry

    env = registry.load("G1JoystickFlatTerrain")
    vreset = jax.jit(jax.vmap(env.reset))
    vstep = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(42), n_envs)
    states = vreset(keys)
    jax.block_until_ready(states.obs["state"])

    # Warm up one step (triggers kernel compile for this batch shape)
    states = vstep(states, jnp.zeros((n_envs, env.action_size)))
    jax.block_until_ready(states.obs["state"])

    n_steps = 200
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(7)
    for _ in range(n_steps):
        key, sk = jax.random.split(key)
        actions = jax.random.uniform(
            sk, (n_envs, env.action_size), minval=-1.0, maxval=1.0
        )
        states = vstep(states, actions)
    jax.block_until_ready(states.obs["state"])
    dt = time.perf_counter() - t0

    env_steps = n_envs * n_steps
    sps = env_steps / dt
    print(
        f"RESULT n_envs={n_envs} steps={n_steps} wall={dt:.3f} "
        f"env_steps={env_steps} kenvsteps_per_s={sps / 1000:.2f}"
    )


def orchestrate() -> None:
    """Top-level: spawn one subprocess per batch size, collect + log."""
    OUT.parent.mkdir(parents=True, exist_ok=True)

    header_lines: list[str] = []

    def log(s: str) -> None:
        print(s)
        header_lines.append(s)

    # ---- One-time env probe + single-env sanity check (orchestrator side) ----
    log("--- load G1JoystickFlatTerrain ---")
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        check=False,
        capture_output=True,
        text=True,
    )
    # Strip kernel-compile / warp-init noise, keep the clean summary lines
    for line in probe.stdout.splitlines():
        if line.startswith("PROBE "):
            log(line[len("PROBE ") :])
    if probe.returncode != 0:
        print("probe stderr tail:", probe.stderr.splitlines()[-5:], file=sys.stderr)
        raise SystemExit(probe.returncode)

    # ---- Parallel throughput: one subprocess per batch size ----
    log("")
    log("--- parallel throughput (each batch in a fresh subprocess) ---")
    results: list[str] = []
    for n in BATCHES:
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--bench", str(n)],
            check=False,
            capture_output=True,
            text=True,
        )
        result_line = next(
            (line for line in r.stdout.splitlines() if line.startswith("RESULT ")),
            None,
        )
        if r.returncode != 0 or result_line is None:
            msg = f"  {n:>5} envs: FAILED (exit={r.returncode}, OOM likely)"
            log(msg)
            results.append(msg)
            # Higher batches will also fail — stop scanning.
            break
        kvs = dict(x.split("=", 1) for x in result_line[len("RESULT ") :].split())
        line = (
            f"  {n:>5} envs x {kvs['steps']} steps = {kvs['env_steps']:>9} env-steps "
            f"in {float(kvs['wall']):>6.2f}s -> {float(kvs['kenvsteps_per_s']):>9.1f} kEnvSteps/s"
        )
        log(line)
        results.append(line)

    with OUT.open("w") as f:
        for line in header_lines:
            f.write(line + "\n")
    log("")
    log(f"bench log -> {OUT.relative_to(ROOT)}")


def probe() -> None:
    """Worker: print env shape / dt / sanity-rollout stats, then exit."""
    import jax
    import jax.numpy as jnp
    from mujoco_playground import registry

    env = registry.load("G1JoystickFlatTerrain")
    print(f"PROBE env class:       {type(env).__name__}")
    print(f"PROBE obs_size:        {env.observation_size}")
    print(f"PROBE action_size:     {env.action_size}")
    print(f"PROBE control dt:      {env.dt}s ({1 / env.dt:.0f} Hz)")

    key = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(key)
    obs_fingerprint = ", ".join(f"{k}:{v.shape}" for k, v in state.obs.items())
    print(f"PROBE reset obs:       {obs_fingerprint}")
    print(f"PROBE reset reward:    {float(state.reward):.3f}  done: {bool(state.done)}")
    state = step_fn(state, jnp.zeros(env.action_size))
    jax.block_until_ready(state.obs["state"])
    print(f"PROBE one-step reward: {float(state.reward):.3f}  done: {bool(state.done)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=int, default=None)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.bench is not None:
        bench_one(args.bench)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
