# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""``vit-plan`` — MPPI planning entry point.

Replaces the 868 LOC ``scripts/m4c_hierarchical_plan.py``. The heavy
lifting lives in the library; this module is argparse + run-loop.
"""

from __future__ import annotations

import argparse
import time
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
    render_multi_cam,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan with a JEPA world model.")
    ap.add_argument("config", type=Path, help="YAML config path")
    ap.add_argument("--override", "-o", action="append", default=[])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--policy-ckpt", type=Path, default=None)
    ap.add_argument("--h5", type=Path, default=None)
    ap.add_argument("--goal-ep", type=int, default=None)
    ap.add_argument("--goal-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-out", type=Path, default=None)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_root = Path(__file__).resolve().parents[3]

    ckpt = args.ckpt or Path(cfg["ckpt"]).expanduser()
    jepa = load_jepa(ckpt, device=device)
    # Backbone weights required at plan time — hydrate the lazy path if
    # the checkpoint used ``lazy=True``.
    if hasattr(jepa.backbone, "load_eagerly"):
        jepa.backbone.load_eagerly()
    if not args.no_compile:
        jepa.predictor = compile_model(jepa.predictor, mode="reduce-overhead")

    planner_backbone = PlannerBackbone(jepa)

    # Build env + optional PPO policy.
    policy_ckpt = args.policy_ckpt or (
        Path(cfg["policy_ckpt"]).expanduser()
        if cfg.get("policy_ckpt")
        else None
    )
    env_ctx = build_env_and_policy(
        ckpt_policy=policy_ckpt,
        device=device,
        seed=args.seed,
        repo_root=repo_root,
    )

    vel = cfg.get("vel_cmd")
    if vel is not None:
        env_ctx["pinned_cmd"] = jnp.asarray(
            [float(v) for v in vel], dtype=jnp.float32
        )
        env_ctx["state"] = env_ctx["state"].replace(
            info={**env_ctx["state"].info, "command": env_ctx["pinned_cmd"]}
        )

    # Build goal embedding — a plain visual embedding. The world model's
    # target is visual-only (proprio is train-time conditioning), so the
    # goal and the MPPI cost live in the same clean visual space.
    h5 = args.h5 or Path(cfg["h5"]).expanduser()
    goal_ep = args.goal_ep if args.goal_ep is not None else int(cfg["goal_ep"])
    goal_pixel = torch.from_numpy(load_goal_pixel(h5, args.goal_idx, goal_ep))
    with bf16_autocast(), torch.no_grad():
        goal_emb = encode_goal(planner_backbone, goal_pixel).to(device)

    # Build planner.
    plan_cfg = cfg.get("planner", {})
    planner = MPPIPlanner(
        jepa=jepa,
        backbone=planner_backbone,
        subgoal_emb=goal_emb,
        cost_fn=MSECost(),
        action_dim=int(plan_cfg.get("action_dim", 29)),
        horizon=int(plan_cfg.get("horizon", 50)),
        num_samples=int(plan_cfg.get("num_samples", 64)),
        noise_sigma=float(plan_cfg.get("noise_sigma", 0.02)),
        lambda_=float(plan_cfg.get("lambda", 0.0025)),
        iterations=int(plan_cfg.get("iterations", 3)),
        history_size=int(plan_cfg.get("history_size", 3)),
        device=device,
    )

    # Encode-once history. The world model is per-step, so the history
    # must be the last HS *consecutive* frames (matching training), not
    # one stale frame per macro.
    HS = planner.history_size
    history = EncoderHistory(size=HS, encoder=planner_backbone.encode)
    action_hist: list[np.ndarray] = []

    def _capture_frame() -> None:
        """Encode + retain the current head-cam frame (visual-only)."""
        pix = render_head_cam(env_ctx)
        pix_chw = torch.from_numpy(pix).permute(2, 0, 1).contiguous()
        history.push(pix_chw)

    n_macros = int(cfg.get("n_macros", 10))
    frames: list[np.ndarray] = []
    rng_key = env_ctx["rng"]
    t0 = time.perf_counter()

    for macro_idx in range(n_macros):
        # Macro 0 has no prior frames — seed with the current one. Later
        # macros inherit a full HS-frame window from the previous macro's
        # tail captures below.
        if len(history) == 0:
            _capture_frame()

        warm_U, rng_key = rollout_policy_warm_start(
            env_ctx, planner.horizon, rng_key
        )
        ah = (
            np.stack(action_hist[-HS:], axis=0)
            if action_hist
            else np.zeros((1, planner.action_dim), dtype=np.float32)
        )
        ah_t = torch.from_numpy(ah)

        with bf16_autocast():
            U = planner.plan(
                pixel_history=None,
                action_history=ah_t,
                warm_start_U=(
                    torch.from_numpy(warm_U) if warm_U is not None else None
                ),
                encoded_history=history.latest_window(),
            )
        U_np = U.detach().cpu().numpy()

        for step_i in range(planner.horizon):
            action = jnp.asarray(U_np[step_i], dtype=jnp.float32)
            env_ctx["state"] = env_ctx["step_fn"](env_ctx["state"], action)
            if env_ctx.get("pinned_cmd") is not None:
                env_ctx["state"] = env_ctx["state"].replace(
                    info={
                        **env_ctx["state"].info,
                        "command": env_ctx["pinned_cmd"],
                    }
                )
            action_hist.append(U_np[step_i])
            # Capture the last HS frames of this macro so the next plan
            # sees a consecutive HS-frame history.
            if step_i >= planner.horizon - HS:
                _capture_frame()
            if args.video_out is not None:
                frames.append(render_multi_cam(env_ctx))

        xyz = get_torso_xyz(env_ctx)
        print(
            f"[macro {macro_idx + 1:02d}/{n_macros}]  "
            f"best_cost={planner.best_cost:.4f}  "
            f"torso_xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})"
        )

    print(f"=== plan done in {time.perf_counter() - t0:.1f}s ===")

    if args.video_out is not None and frames:
        import mediapy

        mediapy.write_video(str(args.video_out), frames, fps=50)
        print(f"[video] wrote {args.video_out}")


if __name__ == "__main__":
    main()
