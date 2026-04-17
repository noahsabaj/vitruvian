#!/usr/bin/env python
"""Diagnostic — run M1 PPO policy alone on G1 playground, no planner.

If G1 walks: the env + policy are fine, all failures are in the planner.
If G1 falls: env/initial-state is broken and no planner can rescue us.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "le-wm"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
import jax.numpy as jnp

from m4c_hierarchical_plan import build_env_and_policy, get_torso_xyz


def main() -> None:
    ckpt = Path("checkpoints/m1-g1-full/000043253760")
    ctx = build_env_and_policy(ckpt, device="cuda", seed=0)
    policy = ctx["policy"]
    if policy is None:
        print("[fatal] no policy loaded")
        sys.exit(1)

    start = get_torso_xyz(ctx)
    print(f"[start] xyz = {start}")

    total_steps = 500  # 10 seconds
    upright = 0
    rng = ctx["rng"]
    for t in range(total_steps):
        rng, sub = jax.random.split(rng)
        action, _ = policy(ctx["state"].obs, sub)
        ctx["state"] = ctx["step_fn"](
            ctx["state"], jnp.asarray(action, dtype=jnp.float32)
        )
        xyz = get_torso_xyz(ctx)
        if xyz[2] > 0.5:
            upright += 1
        if t % 50 == 49:
            dxy = float(np.linalg.norm(xyz[:2] - start[:2]))
            print(
                f"[t={t + 1:3d}] z={xyz[2]:.3f} m  dxy={dxy:.3f} m  "
                f"upright-frac={upright / (t + 1):.2f}"
            )
        if xyz[2] < 0.3:
            print(f"[fell] at t={t + 1}, z={xyz[2]:.3f}")
            break

    final = get_torso_xyz(ctx)
    dxy = float(np.linalg.norm(final[:2] - start[:2]))
    print()
    print(
        f"--- Summary ---\n"
        f"final z:       {final[2]:.3f} m\n"
        f"dxy:           {dxy:.3f} m\n"
        f"upright-frac:  {upright / total_steps:.2f}\n"
        f"verdict:       {'WALKS' if final[2] > 0.5 and upright / total_steps > 0.8 else 'FALLS'}"
    )


if __name__ == "__main__":
    main()
