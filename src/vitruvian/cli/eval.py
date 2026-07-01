# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""``vit-eval`` — eval-matrix driver.

Single-process runner — replaces the 40-row bash loop that re-paid ~8s
Python + JAX + model-load startup per config. See
:func:`vitruvian.cli.eval.main` for the run loop.

Configs are expected to carry a ``runs`` list of ``RunConfig`` dicts;
each run fires a full walk with fresh env reset but shared
env/JEPA/policy state.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from vitruvian.env import (
    build_env_and_policy,
    get_torso_xyz,
    load_goal_pixel,
    render_head_cam,
    rollout_policy_warm_start,
)
from vitruvian.models import PlannerBackbone, load_jepa
from vitruvian.planning import (
    EncoderHistory,
    MPPIPlanner,
    MSECost,
    encode_goal,
)
from vitruvian.utils import bf16_autocast, compile_model, load_config


@dataclass(frozen=True)
class RunConfig:
    scenario: str
    h5_path: str
    goal_ep: int
    sigma: float
    seed: int
    n_macros: int = 10
    horizon: int = 50
    num_samples: int = 64


@dataclass
class RunResult:
    cfg: dict
    mean_cos: float
    cos_range: float
    max_cos: float
    final_torso_xyz: list[float]
    walk_completed: bool
    wall_s: float
    per_macro: list[dict] = field(default_factory=list)


def _run_one(
    cfg: RunConfig,
    env_ctx: dict,
    jepa,
    planner_backbone,
) -> RunResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinned_cmd = env_ctx.get("pinned_cmd")

    # Fresh env reset for this run. Pin the eval command into the state so
    # the rollout actually runs under ``vel_cmd`` (the env otherwise keeps
    # the random reset command).
    rng = jax.random.PRNGKey(cfg.seed)
    rng, rkey = jax.random.split(rng)
    state = env_ctx["reset_fn"](rkey)
    if pinned_cmd is not None:
        state = state.replace(info={**state.info, "command": pinned_cmd})
    env_ctx["state"] = state
    env_ctx["rng"] = rng

    with bf16_autocast(), torch.no_grad():
        goal_pixel = torch.from_numpy(
            load_goal_pixel(Path(cfg.h5_path), 0, cfg.goal_ep)
        )
        # Plain visual goal embedding — the world-model target is
        # visual-only, so the planner cost and this cosine diagnostic
        # both live in that one space.
        goal_emb = encode_goal(planner_backbone, goal_pixel).to(device)

    planner = MPPIPlanner(
        jepa=jepa,
        backbone=planner_backbone,
        subgoal_emb=goal_emb,
        cost_fn=MSECost(),
        action_dim=29,
        horizon=cfg.horizon,
        num_samples=cfg.num_samples,
        noise_sigma=cfg.sigma,
        iterations=3,
        history_size=3,
        device=device,
    )
    HS = planner.history_size
    history = EncoderHistory(size=HS, encoder=planner_backbone.encode)
    action_hist: list[np.ndarray] = []
    rng_key = env_ctx["rng"]

    def _capture_frame() -> None:
        pix = render_head_cam(env_ctx)
        pix_chw = torch.from_numpy(pix).permute(2, 0, 1).contiguous()
        history.push(pix_chw)

    cosines: list[float] = []
    per_macro: list[dict] = []
    walk_completed = True
    t0 = time.perf_counter()

    for macro_idx in range(cfg.n_macros):
        # Seed macro 0; later macros inherit a consecutive HS-frame window
        # from the previous macro's tail captures below.
        if len(history) == 0:
            _capture_frame()

        warm_U, rng_key = rollout_policy_warm_start(
            env_ctx, planner.horizon, rng_key
        )
        ah = (
            np.stack(action_hist[-HS:], axis=0)
            if action_hist
            else np.zeros((1, 29), dtype=np.float32)
        )
        with bf16_autocast():
            U = planner.plan(
                pixel_history=None,
                action_history=torch.from_numpy(ah),
                warm_start_U=(
                    torch.from_numpy(warm_U) if warm_U is not None else None
                ),
                encoded_history=history.latest_window(),
            )
        U_np = U.detach().cpu().numpy()

        # Cosine between the current visual latent and the goal — a
        # logged diagnostic in the same visual space.
        cur_flat = history.latest().flatten().float()
        goal_flat = goal_emb.flatten().float()
        cos = float(
            (cur_flat * goal_flat).sum()
            / (cur_flat.norm() * goal_flat.norm() + 1e-9)
        )
        cosines.append(cos)
        per_macro.append({"macro": macro_idx, "cos": cos})

        for step_i in range(planner.horizon):
            action = jnp.asarray(U_np[step_i], dtype=jnp.float32)
            env_ctx["state"] = env_ctx["step_fn"](env_ctx["state"], action)
            if pinned_cmd is not None:
                env_ctx["state"] = env_ctx["state"].replace(
                    info={**env_ctx["state"].info, "command": pinned_cmd}
                )
            action_hist.append(U_np[step_i])
            if step_i >= planner.horizon - HS:
                _capture_frame()

        # Safety — if the robot falls mid-walk (torso z < 0.3m) abort.
        z = float(get_torso_xyz(env_ctx)[2])
        if z < 0.3:
            walk_completed = False
            break

    wall = time.perf_counter() - t0
    return RunResult(
        cfg=asdict(cfg),
        mean_cos=float(np.mean(cosines)) if cosines else float("nan"),
        cos_range=float(np.ptp(cosines)) if cosines else 0.0,
        max_cos=float(np.max(cosines)) if cosines else float("nan"),
        final_torso_xyz=list(map(float, get_torso_xyz(env_ctx))),
        walk_completed=walk_completed,
        wall_s=wall,
        per_macro=per_macro,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an eval matrix.")
    ap.add_argument("config", type=Path, help="YAML config path")
    ap.add_argument("--override", "-o", action="append", default=[])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--policy-ckpt", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_root = Path(__file__).resolve().parents[3]

    ckpt = args.ckpt or Path(cfg["ckpt"]).expanduser()
    jepa = load_jepa(ckpt, device=device)
    if hasattr(jepa.backbone, "load_eagerly"):
        jepa.backbone.load_eagerly()
    if not args.no_compile:
        jepa.predictor = compile_model(jepa.predictor, mode="reduce-overhead")

    planner_backbone = PlannerBackbone(jepa)

    policy_ckpt = args.policy_ckpt or (
        Path(cfg["policy_ckpt"]).expanduser()
        if cfg.get("policy_ckpt")
        else None
    )
    env_ctx = build_env_and_policy(
        ckpt_policy=policy_ckpt,
        device=device,
        seed=0,
        repo_root=repo_root,
    )

    vel = cfg.get("vel_cmd")
    if vel is not None:
        env_ctx["pinned_cmd"] = jnp.asarray(
            [float(v) for v in vel], dtype=jnp.float32
        )

    runs_cfg = [RunConfig(**r) for r in cfg["runs"]]
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir is not None
        else Path(cfg.get("out_dir", "/tmp/vitruvian/eval")).expanduser()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "summary.jsonl"
    tsv = out_dir / "summary.tsv"

    with jsonl.open("w") as jf, tsv.open("w") as tf:
        tf.write("scenario\th5\tgoal_ep\tsigma\tseed\tmean_cos\tcos_range\tmax_cos\twalk_completed\twall_s\n")
        for i, r_cfg in enumerate(runs_cfg, 1):
            print(f"[{i}/{len(runs_cfg)}] {r_cfg}")
            res = _run_one(r_cfg, env_ctx, jepa, planner_backbone)
            jf.write(json.dumps(asdict(res)) + "\n")
            tf.write(
                f"{r_cfg.scenario}\t{r_cfg.h5_path}\t{r_cfg.goal_ep}\t"
                f"{r_cfg.sigma}\t{r_cfg.seed}\t{res.mean_cos:.4f}\t"
                f"{res.cos_range:.4f}\t{res.max_cos:.4f}\t"
                f"{int(res.walk_completed)}\t{res.wall_s:.1f}\n"
            )

    print(f"[eval] wrote {jsonl} and {tsv}")


if __name__ == "__main__":
    main()
