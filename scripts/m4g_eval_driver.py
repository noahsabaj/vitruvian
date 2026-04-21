#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.7 — Single-process eval driver.

Replaces the M4.5/M4.6 bash-loop evaluation matrices
(``m4e_eval_matrix.sh`` × 40 + 10 control runs) with one Python
process that:

  1. Builds the env + loads DINOv3 + loads the JEPA checkpoint + loads
     the optional VF head ONCE. JAX JIT and DINOv3 weights are reused
     across every configuration in the matrix.
  2. Iterates ``(scenario, sigma, vf, h5_path, goal_ep)`` configurations,
     running a full 10-macro walk per config with fresh env resets.
  3. Writes a structured JSONL stream to
     ``<out-dir>/summary.jsonl`` — one record per run, easy to
     post-process without regex-scraping log files.

Wall-time savings over the bash driver:
  * 40 × ~8 s Python/JAX/model-load startup cost → eliminated (~5 min).
  * Shared JAX ``step_fn``/``reset_fn`` JIT graphs reused across runs.
  * Shared DINOv3 weights and compiled predictor reused.

Usage:
  uv run python scripts/m4g_eval_driver.py \\
      --encoder dinov3-v5 \\
      --ckpt-jepa-v5 ~/.vitruvian/m4f_v5/best.pt \\
      --policy-ckpt checkpoints/m1-g1-full/000043253760 \\
      --vel-cmd 0.5,0,0 \\
      --out-dir /tmp/vitruvian/m4g_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "le-wm"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse env + helper utilities from the interactive planner so we don't
# duplicate the JAX/PPO/playground wiring.
from m4c_hierarchical_plan import (  # noqa: E402
    build_env_and_policy,
    encode_goal,
    get_torso_xyz,
    load_goal_pixel,
    rollout_policy_warm_start,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vitruvian.hwm.compile_utils import compile_model  # noqa: E402
from vitruvian.hwm.encoder_history import EncoderHistory  # noqa: E402
from vitruvian.hwm.planners import LowLevelPlanner  # noqa: E402


# --------------------------------------------------------------------------
# Config + result dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    scenario: str            # "orig" or "diverse"
    h5_path: str             # goal source HDF5
    goal_ep: int
    sigma: float             # MPPI noise_sigma
    vf: bool                 # VF head enabled?
    seed: int                # env reset seed


@dataclass
class RunResult:
    scenario: str
    h5_path: str
    goal_ep: int
    sigma: float
    vf: bool
    seed: int
    macros_run: int
    macros_total: int
    upright: int
    final_z: float
    dxy: float
    mean_cos: float
    max_cos: float
    min_cos: float
    cos_range: float
    wall_s: float
    per_macro_cos: list[float] = field(default_factory=list)
    fell: bool = False

    def to_jsonable(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Core run loop — factored out so the driver can call it per config
# --------------------------------------------------------------------------


def _pin_command(state, pinned_cmd):
    return state.replace(info={**state.info, "command": pinned_cmd})


def run_one(
    cfg: RunConfig,
    *,
    env_ctx: dict,
    backbone,
    jepa_for_rollout,
    backbone_for_planner,
    vf_head,  # optional loaded ValueHead (or None)
    step_skip: int,
    total_macros: int,
    l1_num_samples: int,
    l1_iterations: int,
    pinned_cmd,
    warm_start_policy: bool,
    fall_z_threshold: float = 0.3,
    upright_z_threshold: float = 0.5,
) -> RunResult:
    """Run one 10-macro walk for a single configuration.

    All heavy objects (env_ctx, models) are reused across calls; this
    function only resets the env and constructs a fresh
    ``LowLevelPlanner`` for the config's goal+sigma+vf.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()

    # 1. Reset env with this config's seed.
    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_key = jax.random.split(rng)
    state = env_ctx["reset_fn"](reset_key)
    if pinned_cmd is not None:
        state = _pin_command(state, pinned_cmd)
    env_ctx["state"] = state
    env_ctx["rng"] = rng
    env_ctx["pinned_cmd"] = pinned_cmd

    # 2. Load goal for this config.
    goal_pix = load_goal_pixel(
        "expert", Path(cfg.h5_path), idx=0, ep_idx=cfg.goal_ep
    )
    goal_emb = encode_goal(backbone, goal_pix.to(device))

    # 3. Build planner fresh per config — holds subgoal + sigma.
    planner = LowLevelPlanner(
        lewm_jepa=jepa_for_rollout,
        backbone=backbone_for_planner,
        subgoal_emb=goal_emb,
        horizon=step_skip,
        num_samples=l1_num_samples,
        noise_sigma=cfg.sigma,
        iterations=l1_iterations,
        history_size=3,
        device=device,
        value_head=vf_head if cfg.vf else None,
    )

    # 4. Walk loop with EncoderHistory (M4.7 encode-once-per-frame).
    HIST_SIZE = 3
    history = EncoderHistory(size=HIST_SIZE, encoder=backbone.encode)
    action_hist: list[np.ndarray] = []
    start_xyz = get_torso_xyz(env_ctx)
    per_macro_cos: list[float] = []
    upright_count = 0
    macros_run = 0
    fell = False

    # Import render_head_cam lazily.
    from m4c_hierarchical_plan import render_head_cam

    for macro_idx in range(total_macros):
        # Render current head-cam.
        pix = render_head_cam(env_ctx)
        pix_chw = (
            torch.from_numpy(pix).permute(2, 0, 1).float().to(device) / 255.0
        )
        curr_emb = history.push(pix_chw)

        # Cos / cost metrics (shape-agnostic).
        if curr_emb.ndim == 1:
            cos_val = float(
                torch.nn.functional.cosine_similarity(
                    curr_emb.unsqueeze(0), goal_emb.unsqueeze(0), dim=-1
                )
            )
        else:
            cos_val = float(
                torch.nn.functional.cosine_similarity(
                    curr_emb, goal_emb, dim=-1
                ).mean()
            )
        per_macro_cos.append(cos_val)

        # Warm-start rollout.
        warm_start_U_t = None
        if warm_start_policy:
            warm_U_np, env_ctx["rng"] = rollout_policy_warm_start(
                env_ctx, step_skip, env_ctx["rng"]
            )
            if warm_U_np is not None:
                warm_start_U_t = torch.from_numpy(warm_U_np).to(device)

        # Plan.
        encoded_window = history.latest_window()
        ah = (
            torch.from_numpy(np.stack(action_hist, axis=0)).to(device)
            if action_hist
            else torch.zeros(0, 29, device=device)
        )
        U = planner.plan(
            pixel_history=None,
            action_history=ah,
            warm_start_U=warm_start_U_t,
            encoded_history=encoded_window,
        )
        primitives = U.detach().cpu().numpy()

        # Step env with primitives.
        for p in primitives:
            env_ctx["state"] = env_ctx["step_fn"](
                env_ctx["state"], jnp.asarray(p, dtype=jnp.float32)
            )
            if pinned_cmd is not None:
                env_ctx["state"] = _pin_command(env_ctx["state"], pinned_cmd)
            action_hist.append(p.copy())
        if len(action_hist) > HIST_SIZE:
            action_hist = action_hist[-HIST_SIZE:]

        xyz = get_torso_xyz(env_ctx)
        if xyz[2] > upright_z_threshold:
            upright_count += 1
        macros_run += 1

        if xyz[2] < fall_z_threshold:
            fell = True
            break

    # 5. Finalize metrics.
    final_xyz = get_torso_xyz(env_ctx)
    dxy = float(np.linalg.norm(final_xyz[:2] - start_xyz[:2]))
    mean_cos = float(np.mean(per_macro_cos)) if per_macro_cos else 0.0
    max_cos = float(np.max(per_macro_cos)) if per_macro_cos else 0.0
    min_cos = float(np.min(per_macro_cos)) if per_macro_cos else 0.0
    cos_range = max_cos - min_cos

    return RunResult(
        scenario=cfg.scenario,
        h5_path=cfg.h5_path,
        goal_ep=cfg.goal_ep,
        sigma=cfg.sigma,
        vf=cfg.vf,
        seed=cfg.seed,
        macros_run=macros_run,
        macros_total=total_macros,
        upright=upright_count,
        final_z=float(final_xyz[2]),
        dxy=dxy,
        mean_cos=mean_cos,
        max_cos=max_cos,
        min_cos=min_cos,
        cos_range=cos_range,
        wall_s=time.perf_counter() - t_start,
        per_macro_cos=per_macro_cos,
        fell=fell,
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build_default_matrix(orig_h5: Path, diverse_h5: Path) -> list[RunConfig]:
    """10 goals × (σ∈{0, 0.02, 0.05}) × (vf off only for first pass)
    — the M4.7 Stage C baseline matrix (30 runs)."""
    configs: list[RunConfig] = []
    orig_eps = [0, 20, 40, 60, 80]
    diverse_eps = [0, 10, 20, 30, 40]
    sigmas = [0.0, 0.02, 0.05]
    vfs = [False]  # VF deferred in M4.7 first pass

    for sigma in sigmas:
        for vf in vfs:
            for ep in orig_eps:
                configs.append(
                    RunConfig(
                        scenario="orig",
                        h5_path=str(orig_h5),
                        goal_ep=ep,
                        sigma=sigma,
                        vf=vf,
                        seed=0,
                    )
                )
            for ep in diverse_eps:
                configs.append(
                    RunConfig(
                        scenario="diverse",
                        h5_path=str(diverse_h5),
                        goal_ep=ep,
                        sigma=sigma,
                        vf=vf,
                        seed=0,
                    )
                )
    return configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder",
        choices=["lewm-v3", "dinov3-v4", "dinov3-v5"],
        default="dinov3-v5",
    )
    ap.add_argument("--ckpt-lewm", type=Path, default=Path.home() / ".stable_worldmodel/lewm_g1_v3_weights.ckpt")
    ap.add_argument("--ckpt-hl", type=Path, default=Path.home() / ".vitruvian/m4b_hl_v3/best.pt")
    ap.add_argument("--ckpt-jepa-v4", type=Path, default=Path.home() / ".vitruvian/m4e_v4/best.pt")
    ap.add_argument("--ckpt-jepa-v5", type=Path, default=Path.home() / ".vitruvian/m4f_v5/best.pt")
    ap.add_argument(
        "--policy-ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "m1-g1-full" / "000043253760",
    )
    ap.add_argument(
        "--orig-h5",
        type=Path,
        default=Path.home() / ".stable_worldmodel/g1_joystick_expert.h5",
    )
    ap.add_argument(
        "--diverse-h5",
        type=Path,
        default=Path.home() / ".stable_worldmodel/g1_diverse_v1.h5",
    )
    ap.add_argument("--vf-ckpt", type=Path, default=None)
    ap.add_argument(
        "--vel-cmd",
        type=str,
        default="0.5,0,0",
        help="Pinned G1 joystick command 'vx,vy,yaw' (default forward walk).",
    )
    ap.add_argument("--total-macros", type=int, default=10)
    ap.add_argument("--l1-num-samples", type=int, default=64)
    ap.add_argument("--l1-iterations", type=int, default=3)
    ap.add_argument("--warm-start-policy", action="store_true", default=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/vitruvian/m4g_eval"),
    )
    ap.add_argument(
        "--configs-json",
        type=Path,
        default=None,
        help="Optional JSON file listing RunConfig dicts; overrides the "
        "default 30-run matrix.",
    )
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- M4.7 single-process eval driver ---")
    print(f"encoder:      {args.encoder}")
    print(f"policy-ckpt:  {args.policy_ckpt}")
    print(f"vel-cmd:      {args.vel_cmd}")
    print(f"out-dir:      {args.out_dir}")

    # ---- 1. Build env + policy once ----
    print(f"[env]  building G1JoystickFlatTerrain + PPO policy...")
    env_ctx = build_env_and_policy(args.policy_ckpt, device, seed=0)
    # Warm JIT.
    env_ctx["state"] = env_ctx["step_fn"](
        env_ctx["state"], jnp.zeros(29, dtype=jnp.float32)
    )
    _ = get_torso_xyz(env_ctx)

    # ---- 2. Load model(s) once ----
    backbone = None
    jepa_for_rollout = None
    backbone_for_planner = None
    step_skip = 50

    if args.encoder == "lewm-v3":
        from m4c_hierarchical_plan import load_high_level_model

        model, cfg = load_high_level_model(args.ckpt_lewm, args.ckpt_hl, device)
        step_skip = int(cfg.get("step_skip", 50))
        backbone = model.backbone
        jepa_for_rollout = model.backbone.jepa
        backbone_for_planner = model.backbone
    elif args.encoder == "dinov3-v4":
        from vitruvian.hwm.jepa_v4 import load_jepa_v4_from_checkpoint

        print(f"[encoder]  loading JEPAv4 from {args.ckpt_jepa_v4}")
        jepa_v4 = load_jepa_v4_from_checkpoint(str(args.ckpt_jepa_v4), device=device)
        backbone = jepa_v4.backbone
        jepa_for_rollout = jepa_v4
        backbone_for_planner = jepa_v4.backbone
    else:  # dinov3-v5
        from vitruvian.hwm.jepa_v5 import (
            JEPAv5PlannerBackbone,
            load_jepa_v5_from_checkpoint,
        )

        print(f"[encoder]  loading JEPAv5 from {args.ckpt_jepa_v5}")
        jepa_v5 = load_jepa_v5_from_checkpoint(
            str(args.ckpt_jepa_v5), device=device
        )
        v5_planner_bb = JEPAv5PlannerBackbone(jepa_v5).to(device)
        backbone = v5_planner_bb
        jepa_for_rollout = jepa_v5
        backbone_for_planner = v5_planner_bb

    # Optional: compile predictor for faster MPPI rollout.
    if not args.no_compile:
        try:
            jepa_for_rollout.predictor = compile_model(
                jepa_for_rollout.predictor, mode="reduce-overhead", dynamic=True
            )
            print(f"[compile] jepa.predictor wrapped with torch.compile")
        except Exception as e:
            print(f"[compile] skipped ({type(e).__name__}: {e})")

    # ---- 3. Load VF head (once) if provided ----
    vf_head = None
    if args.vf_ckpt is not None and args.vf_ckpt.exists():
        from m4d_train_vf import ValueHead  # type: ignore

        vf_ckpt = torch.load(
            args.vf_ckpt, map_location=device, weights_only=False
        )
        vf_head = ValueHead(
            emb_dim=int(vf_ckpt["emb_dim"]),
            hidden=int(vf_ckpt["hidden"]),
            out_dim=int(vf_ckpt["out_dim"]),
        ).to(device)
        vf_head.load_state_dict(vf_ckpt["vf_state"])
        vf_head.eval()
        for p in vf_head.parameters():
            p.requires_grad_(False)
        print(f"[vf]   loaded value head from {args.vf_ckpt}")

    # ---- 4. Pinned command ----
    pinned_cmd = None
    if args.vel_cmd is not None:
        pinned_cmd = jnp.asarray(
            [float(x) for x in args.vel_cmd.split(",")], dtype=jnp.float32
        )

    # ---- 5. Build config matrix ----
    if args.configs_json is not None and args.configs_json.exists():
        with args.configs_json.open() as f:
            configs = [RunConfig(**d) for d in json.load(f)]
    else:
        configs = build_default_matrix(args.orig_h5, args.diverse_h5)
    print(f"[matrix]  {len(configs)} runs scheduled")

    # ---- 6. Iterate ----
    results_path = args.out_dir / "summary.jsonl"
    summary_tsv = args.out_dir / "summary.tsv"
    with results_path.open("w") as f_json, summary_tsv.open("w") as f_tsv:
        f_tsv.write(
            "idx\tscenario\tsigma\tvf\tgoal_ep\twalks\tupright\tdxy\tmean_cos\tmax_cos\tcos_range\twall_s\tfell\n"
        )
        total_wall = 0.0
        for i, cfg in enumerate(configs):
            print(
                f"[{i + 1:>3}/{len(configs)}]  {cfg.scenario} σ={cfg.sigma} "
                f"vf={cfg.vf} ep={cfg.goal_ep}"
            )
            r = run_one(
                cfg,
                env_ctx=env_ctx,
                backbone=backbone,
                jepa_for_rollout=jepa_for_rollout,
                backbone_for_planner=backbone_for_planner,
                vf_head=vf_head,
                step_skip=step_skip,
                total_macros=args.total_macros,
                l1_num_samples=args.l1_num_samples,
                l1_iterations=args.l1_iterations,
                pinned_cmd=pinned_cmd,
                warm_start_policy=args.warm_start_policy,
            )
            total_wall += r.wall_s
            f_json.write(json.dumps(r.to_jsonable()) + "\n")
            f_json.flush()
            f_tsv.write(
                f"{i}\t{r.scenario}\t{r.sigma}\t{r.vf}\t{r.goal_ep}\t"
                f"{r.macros_run}/{r.macros_total}\t{r.upright}\t"
                f"{r.dxy:.2f}\t{r.mean_cos:+.3f}\t{r.max_cos:+.3f}\t"
                f"{r.cos_range:.3f}\t{r.wall_s:.1f}\t{r.fell}\n"
            )
            f_tsv.flush()
            print(
                f"    {r.macros_run}/{r.macros_total} macros  "
                f"upright={r.upright}  dxy={r.dxy:.2f}  "
                f"mean_cos={r.mean_cos:+.3f}  range={r.cos_range:.3f}  "
                f"wall={r.wall_s:.1f}s"
            )

    print(f"\n=== eval matrix complete ({total_wall:.0f}s total) ===")
    print(f"results:  {results_path}")
    print(f"summary:  {summary_tsv}")


if __name__ == "__main__":
    main()
