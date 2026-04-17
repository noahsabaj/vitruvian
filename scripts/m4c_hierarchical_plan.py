#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.4c — hierarchical HWM-on-LeWM planning on the G1 playground env.

Loads the frozen LeWM encoder + trained HL predictor (from M4.4b),
builds the ``G1JoystickFlatTerrain`` env with a head-camera, then
runs a receding-horizon MPPI planner:

    for each macro (1 s = 50 primitive steps):
        pixels   = render head-cam frame
        curr_emb = LeWM.encode(pixels)
        macro_latent = HighLevelPlanner.plan(curr_emb, goal_emb)[0]
        primitives   = MacroNNRetriever.retrieve_first(macro_latent)
        for p in primitives (50 steps):
            env.step(p)
        log cost, cosine to goal, torso height, Δx-y

See docs/decisions/008-hwm-planning-layer.md for the architectural
choices. Runs in the main vitruvian venv.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "le-wm"))

# M1 / M4.2 shim for brax on JAX 0.10 — same pattern as m1_train.py.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


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

from vitruvian.hwm.action_codec import MacroActionEncoder  # noqa: E402
from vitruvian.hwm.backbone_adapter import (  # noqa: E402
    load_lewm_jepa_from_checkpoint,
    LeWMBackboneAdapter,
)
from vitruvian.hwm.goal_builder import MacroNNRetriever  # noqa: E402
from vitruvian.hwm.high_level import HighLevelModel  # noqa: E402
from vitruvian.hwm.planners import (  # noqa: E402
    HierarchicalPlanner,
    HighLevelPlanner,
    LowLevelPlanner,
    encode_goal,
)


STABLEWM_HOME = Path.home() / ".stable_worldmodel"
DEFAULT_CKPT_LEWM = STABLEWM_HOME / "lewm_g1_seed_weights.ckpt"
DEFAULT_CKPT_HL = Path.home() / ".vitruvian" / "m4b_hl_v1" / "latest.pt"
DEFAULT_H5 = STABLEWM_HOME / "g1_joystick_expert.h5"

ENV_NAME = "G1JoystickFlatTerrain"
ENV_OVERRIDES = {"njmax": 96}
G1_SCENE = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
HEAD_CAM_POS = [0.10, 0.0, 0.18]
HEAD_CAM_QUAT = [-0.5, -0.5, 0.5, 0.5]


def load_high_level_model(
    ckpt_lewm: Path,
    ckpt_hl: Path,
    device: str,
) -> tuple[HighLevelModel, dict]:
    """Reconstruct HighLevelModel with LeWM frozen + trained HL weights."""
    print(f"[lewm] loading frozen JEPA from {ckpt_lewm}")
    jepa = load_lewm_jepa_from_checkpoint(
        str(ckpt_lewm),
        lewm_repo_path=str(ROOT / "external" / "le-wm"),
        device=device,
    )

    print(f"[hl]   loading HL checkpoint from {ckpt_hl}")
    payload = torch.load(str(ckpt_hl), map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    n_macros_train = int(cfg.get("n_macros", 2))

    model = HighLevelModel(
        lewm_jepa=jepa,
        action_dim=29,
        step_skip=int(cfg.get("step_skip", 50)),
        macro_act_dim=int(cfg.get("macro_act_dim", 32)),
        hidden=int(cfg.get("hl_hidden", 256)),
        n_layers=int(cfg.get("hl_layers", 4)),
        n_heads=int(cfg.get("hl_heads", 4)),
        max_seq_len=max(n_macros_train, 4),
        freeze_backbone=True,
    ).to(device)
    model.action_encoder.load_state_dict(payload["action_encoder"])
    model.predictor.load_state_dict(payload["predictor"])
    model.eval()
    print(f"[hl]   params: {model.n_trainable():,} (trainable)")
    return model, cfg


def build_env_and_policy(ckpt_policy: Path | None, device: str, seed: int):
    """Build the G1 playground env with head-cam and (optionally) a
    warm-up policy we can reset to. The planner does not use the
    policy at step time — actions come from the NN retriever — but
    having a policy makes it cheap to compute a known-good initial
    state distribution if we ever want it.
    """
    import functools
    import pickle

    import mujoco
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from mujoco_playground import registry
    from mujoco_playground.config import locomotion_params

    spec = mujoco.MjSpec.from_file(str(G1_SCENE))
    torso = spec.body("torso_link")
    cam = torso.add_camera()
    cam.name = "head"
    cam.pos = HEAD_CAM_POS
    cam.quat = HEAD_CAM_QUAT
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    cam_id = mj_model.camera("head").id
    renderer = mujoco.Renderer(mj_model, height=224, width=224)

    env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)

    policy = None
    if ckpt_policy is not None and ckpt_policy.exists():
        print(f"[env]  loading initial-state policy from {ckpt_policy}")
        if ckpt_policy.is_dir():
            params = brax_checkpoint.load(str(ckpt_policy.resolve()))
        else:
            with ckpt_policy.open("rb") as f:
                params = pickle.load(f)
        cfg = locomotion_params.brax_ppo_config(ENV_NAME)
        factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=tuple(
                cfg.network_factory.policy_hidden_layer_sizes
            ),
            value_hidden_layer_sizes=tuple(
                cfg.network_factory.value_hidden_layer_sizes
            ),
            policy_obs_key=cfg.network_factory.policy_obs_key,
            value_obs_key=cfg.network_factory.value_obs_key,
        )
        preprocess_fn = (
            running_statistics.normalize
            if cfg.normalize_observations
            else lambda obs, *_: obs
        )
        net = factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=preprocess_fn,
        )
        policy = jax.jit(
            ppo_networks.make_inference_fn(net)(params, deterministic=True)
        )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    rng, rkey = jax.random.split(rng)
    state = reset_fn(rkey)

    return {
        "env": env,
        "state": state,
        "rng": rng,
        "reset_fn": reset_fn,
        "step_fn": step_fn,
        "mj_model": mj_model,
        "mj_data": mj_data,
        "cam_id": cam_id,
        "renderer": renderer,
        "policy": policy,
    }


def render_head_cam(env_ctx: dict) -> np.ndarray:
    """Pull mjx qpos/qvel onto the CPU-side mj_data and render head-cam."""
    state = env_ctx["state"]
    mjx_data = getattr(state, "data", None) or getattr(
        state, "pipeline_state", None
    )
    mj_data = env_ctx["mj_data"]
    mj_model = env_ctx["mj_model"]
    mj_data.qpos[:] = np.asarray(mjx_data.qpos)
    mj_data.qvel[:] = np.asarray(mjx_data.qvel)
    import mujoco

    mujoco.mj_forward(mj_model, mj_data)
    env_ctx["renderer"].update_scene(mj_data, camera=env_ctx["cam_id"])
    return env_ctx["renderer"].render().copy()  # (224, 224, 3) uint8


def rollout_policy_warm_start(
    env_ctx: dict, horizon: int, rng_key
) -> tuple[np.ndarray, object] | tuple[None, object]:
    """Roll the loaded PPO policy on a SHADOW copy of the env state to
    produce a nominal action sequence of length ``horizon``. The real
    env state is never mutated.

    Returns ``(warm_U, next_rng_key)`` where warm_U has shape
    (horizon, action_dim) dtype float32, or ``(None, rng_key)`` if no
    policy is loaded.
    """
    policy = env_ctx.get("policy")
    if policy is None:
        return None, rng_key
    shadow_state = env_ctx["state"]
    acts: list[np.ndarray] = []
    for _ in range(horizon):
        rng_key, sub = jax.random.split(rng_key)
        action, _ = policy(shadow_state.obs, sub)
        acts.append(np.asarray(action, dtype=np.float32))
        shadow_state = env_ctx["step_fn"](shadow_state, action)
    return np.stack(acts, axis=0), rng_key


def get_torso_xyz(env_ctx: dict) -> np.ndarray:
    state = env_ctx["state"]
    mjx_data = getattr(state, "data", None) or getattr(
        state, "pipeline_state", None
    )
    # qpos[0:3] = free-joint position of the root body.
    return np.asarray(mjx_data.qpos[:3])


def load_goal_pixel(
    goal_source: str, h5_path: Path, idx: int, ep_idx: int
) -> torch.Tensor:
    """Return goal as a uint8 (224, 224, 3) tensor."""
    if goal_source == "expert":
        # Use a mid-episode frame from the expert H5 as a crude goal.
        with h5py.File(h5_path, "r") as f:
            off = int(f["ep_offset"][ep_idx])
            L = int(f["ep_len"][ep_idx])
            t = min(off + L - 1, off + L // 2 + idx)
            pixel = f["pixels"][t]  # (224, 224, 3) uint8
        return torch.from_numpy(np.asarray(pixel))
    raise ValueError(f"unknown goal source: {goal_source!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-lewm", type=Path, default=DEFAULT_CKPT_LEWM)
    ap.add_argument("--ckpt-hl", type=Path, default=DEFAULT_CKPT_HL)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument(
        "--goal-source",
        choices=["expert"],
        default="expert",
        help="expert = pick a frame from the M4.2 expert H5.",
    )
    ap.add_argument("--goal-ep-idx", type=int, default=0)
    ap.add_argument("--goal-idx", type=int, default=0)
    ap.add_argument("--policy-ckpt", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--noise-sigma", type=float, default=10.0)
    ap.add_argument("--lambda-", dest="lambda_", type=float, default=0.0025)
    ap.add_argument("--horizon-macros", type=int, default=2)
    ap.add_argument(
        "--total-macros",
        type=int,
        default=10,
        help="How many macros (1 s each) to roll out.",
    )
    ap.add_argument(
        "--decoder",
        choices=["nn", "mppi", "flat"],
        default="mppi",
        help="Macro→primitive decoder. 'nn' = nearest-neighbour expert "
        "retrieval (M4.4c Stage 1 plumbing path). 'mppi' = true two-level "
        "MPPI per the HWM paper (L1 plans primitives via LeWM predictor). "
        "'flat' = bypass HL, run one-level MPPI on primitives directly "
        "against the goal embedding (PLDM flat-MPC baseline).",
    )
    ap.add_argument("--l1-num-samples", type=int, default=500)
    ap.add_argument("--l1-noise-sigma", type=float, default=0.3)
    ap.add_argument("--l1-iterations", type=int, default=3)
    ap.add_argument(
        "--warm-start-policy",
        action="store_true",
        help="Use the loaded PPO policy as MPPI nominal prior at each "
        "macro. Requires --policy-ckpt. Puts MPPI's sample cloud inside "
        "the world-model training distribution (PLDM expert-prior trick).",
    )
    ap.add_argument(
        "--render-mp4",
        type=Path,
        default=None,
        help="Optional output path — render chase-cam video of run.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- M4.4c hierarchical planning ---")
    print(f"device:          {device}")
    print(f"ckpt-lewm:       {args.ckpt_lewm}")
    print(f"ckpt-hl:         {args.ckpt_hl}")
    print(f"goal-source:     {args.goal_source}  ep={args.goal_ep_idx}")
    print(f"num-samples:     {args.num_samples}  noise-σ: {args.noise_sigma}")
    print(f"horizon:         {args.horizon_macros} macros")
    print(f"total-macros:    {args.total_macros} ({args.total_macros} s)")
    print()

    # --- Env (built first, before the networks, so Warp can allocate
    # the MJX physics CUDA graph while VRAM is still free. On the 8 GB
    # 4060 Ti the opposite order triggers wp_cuda_graph_create_exec
    # OOM on the first env.step).
    env_ctx = build_env_and_policy(args.policy_ckpt, device, args.seed)
    # Force the MJX step graph to compile by taking one no-op step and
    # materializing the state, so Warp allocates its graph memory before
    # the torch models take over the remaining VRAM.
    _warm_action = jnp.zeros((29,), dtype=jnp.float32)
    env_ctx["state"] = env_ctx["step_fn"](env_ctx["state"], _warm_action)
    _ = get_torso_xyz(env_ctx)

    # --- Model ---
    model, cfg = load_high_level_model(args.ckpt_lewm, args.ckpt_hl, device)
    step_skip = int(cfg.get("step_skip", 50))
    backbone = model.backbone

    # --- Goal ---
    goal_pix = load_goal_pixel(
        args.goal_source, args.h5, args.goal_idx, args.goal_ep_idx
    )
    goal_emb = encode_goal(backbone, goal_pix.to(device))
    print(
        f"[goal] source={args.goal_source} ep={args.goal_ep_idx} "
        f"idx={args.goal_idx}  emb shape: {tuple(goal_emb.shape)}  "
        f"norm: {float(goal_emb.norm()):.2f}"
    )

    # --- Planner ---
    hl_planner = HighLevelPlanner(
        hl_model=model,
        goal_emb=goal_emb,
        horizon=args.horizon_macros,
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
        lambda_=args.lambda_,
        device=device,
    )

    flat_planner = None
    if args.decoder == "nn":
        nn_ret = MacroNNRetriever(
            h5_path=args.h5,
            action_encoder=model.action_encoder,
            device=device,
        )
        hier_planner = None
        print(f"[decoder=nn] indexed {len(nn_ret)} expert macros")
    elif args.decoder == "flat":
        nn_ret = None
        hier_planner = None
        flat_planner = LowLevelPlanner(
            lewm_jepa=model.backbone.jepa,
            backbone=model.backbone,
            subgoal_emb=goal_emb,
            horizon=step_skip,
            num_samples=args.l1_num_samples,
            noise_sigma=args.l1_noise_sigma,
            iterations=args.l1_iterations,
            history_size=3,
            device=device,
        )
        print(
            f"[decoder=flat] MPPI-on-primitives: {args.l1_num_samples} "
            f"samples × {args.l1_iterations} iters × {step_skip}-step "
            f"horizon (HL bypassed)"
        )
    else:  # mppi — two-level
        nn_ret = None
        hier_planner = HierarchicalPlanner(
            hl_planner=hl_planner,
            lewm_jepa=model.backbone.jepa,
            backbone=model.backbone,
            horizon_primitives=step_skip,
            num_samples_l1=args.l1_num_samples,
            noise_sigma_l1=args.l1_noise_sigma,
            iterations_l1=args.l1_iterations,
            history_size=3,
            device=device,
        )
        print(
            f"[decoder=mppi] L1: {args.l1_num_samples} samples × "
            f"{args.l1_iterations} iters × {step_skip}-step horizon"
        )

    # --- Env was already built above; just re-bind locals here. ---
    env = env_ctx["env"]
    step_fn = env_ctx["step_fn"]

    # --- Optional video ---
    video_frames: list[np.ndarray] = []

    # --- Run ---
    log_rows: list[dict] = []
    start_xyz = get_torso_xyz(env_ctx)
    # Rolling history for L1 MPPI (history_size most recent frames + actions).
    HIST_SIZE = 3
    pixel_hist: list[torch.Tensor] = []
    action_hist: list[np.ndarray] = []
    t0 = time.perf_counter()
    for macro_idx in range(args.total_macros):
        # 1. Capture current head-cam frame → encode.
        pix = render_head_cam(env_ctx)  # (224, 224, 3) uint8
        pix_t = torch.from_numpy(pix).permute(2, 0, 1).float().unsqueeze(
            0
        ).unsqueeze(0).to(device) / 255.0
        curr_emb = backbone.encode(pix_t).squeeze(0).squeeze(0)

        # Update pixel history with the latest frame.
        pixel_hist.append(pix_t[0, 0])
        if len(pixel_hist) > HIST_SIZE:
            pixel_hist = pixel_hist[-HIST_SIZE:]

        # 2. Plan macros + decode to primitives.
        cost = float(((curr_emb - goal_emb) ** 2).sum())
        cos = float(
            torch.nn.functional.cosine_similarity(
                curr_emb, goal_emb, dim=-1
            )
        )
        if args.decoder == "nn":
            plan = hl_planner.plan(curr_emb, shift_nominal=macro_idx > 0)
            macro_latent = plan[0]
            primitives = nn_ret.retrieve_first(macro_latent)
            l1_cost = None
        elif args.decoder == "flat":
            ph = torch.stack(pixel_hist, dim=0)
            ah = (
                torch.from_numpy(np.stack(action_hist, axis=0)).to(device)
                if action_hist
                else torch.zeros(0, 29, device=device)
            )
            warm_start_U_t = None
            if args.warm_start_policy:
                warm_U_np, env_ctx["rng"] = rollout_policy_warm_start(
                    env_ctx, step_skip, env_ctx["rng"]
                )
                if warm_U_np is not None:
                    warm_start_U_t = torch.from_numpy(warm_U_np).to(device)
            U = flat_planner.plan(ph, ah, warm_start_U=warm_start_U_t)
            primitives = U  # (horizon, action_dim)
            macro_latent = torch.zeros(1, device=device)  # not used
            l1_cost = flat_planner.best_cost
        else:
            ph = torch.stack(pixel_hist, dim=0)
            ah = (
                torch.from_numpy(np.stack(action_hist, axis=0)).to(device)
                if action_hist
                else torch.zeros(0, 29, device=device)
            )
            warm_start_U_t = None
            if args.warm_start_policy:
                warm_U_np, env_ctx["rng"] = rollout_policy_warm_start(
                    env_ctx, step_skip, env_ctx["rng"]
                )
                if warm_U_np is not None:
                    warm_start_U_t = torch.from_numpy(warm_U_np).to(device)
            primitives, diag = hier_planner.plan_primitives(
                curr_emb,
                ph,
                ah,
                shift_nominal=macro_idx > 0,
                warm_start_U=warm_start_U_t,
            )
            macro_latent = diag["macro_plan"][0].to(device)
            l1_cost = diag["l1_best_cost"]

        # 3. Step env with primitives.
        prim_np = primitives.detach().cpu().numpy()
        for p in prim_np:
            action_jax = jnp.asarray(p, dtype=jnp.float32)
            env_ctx["state"] = step_fn(env_ctx["state"], action_jax)
            action_hist.append(p.copy())
        if len(action_hist) > HIST_SIZE:
            action_hist = action_hist[-HIST_SIZE:]

        xyz = get_torso_xyz(env_ctx)
        dxy = float(np.linalg.norm(xyz[:2] - start_xyz[:2]))
        z = float(xyz[2])
        row = {
            "macro": macro_idx,
            "cost": cost,
            "cos_to_goal": cos,
            "torso_z": z,
            "dxy_from_start": dxy,
            "macro_latent_norm": float(macro_latent.norm()),
            "l1_cost": l1_cost,
        }
        log_rows.append(row)
        l1_str = f"  l1={l1_cost:7.2f}" if l1_cost is not None else ""
        print(
            f"  macro {macro_idx:>2}  cost={cost:9.3f}  "
            f"cos={cos:+.3f}  z={z:.3f} m  dxy={dxy:.3f} m  "
            f"|l|={row['macro_latent_norm']:.2f}{l1_str}"
        )

        # 5. Early-termination check — if G1 fell, stop.
        if z < 0.3:
            print(f"  [terminate] torso z < 0.3 m at macro {macro_idx} — fell.")
            break

        if args.render_mp4 is not None:
            video_frames.append(pix)

    wall = time.perf_counter() - t0
    print(f"\n--- Summary ({wall:.1f}s) ---")
    final_xyz = get_torso_xyz(env_ctx)
    dxy_total = float(np.linalg.norm(final_xyz[:2] - start_xyz[:2]))
    final_z = float(final_xyz[2])
    n_macros_run = len(log_rows)
    n_upright = sum(1 for r in log_rows if r["torso_z"] > 0.5)
    mean_cos = (
        float(np.mean([r["cos_to_goal"] for r in log_rows]))
        if log_rows
        else 0.0
    )
    print(
        f"macros run: {n_macros_run}/{args.total_macros}  "
        f"upright: {n_upright}  final z: {final_z:.3f} m"
    )
    print(f"Δxy from start: {dxy_total:.3f} m  mean cos: {mean_cos:+.3f}")

    if args.render_mp4 is not None and video_frames:
        try:
            import mediapy

            args.render_mp4.parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(str(args.render_mp4), video_frames, fps=1)
            print(f"[video] wrote {args.render_mp4}")
        except Exception as e:
            print(f"[video] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
