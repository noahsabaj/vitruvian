#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M3 — forward G1 head-cam frames through DINOv3 (frozen) and save
per-frame latents.

First piece of the graduated architecture (see ADR 003): a frozen
pretrained visual encoder that consumes head-cam pixels and emits a
compact feature vector. This script produces the reference tensors
Dreamer 4 (M4) will eventually consume as part of the observation.

Loading path: **local torch.hub** from the DINOv3 reference repo
(already cached by the user under
`~/.cache/torch/hub/facebookresearch_dinov3_main`) + a local `.pth`
weights file. We deliberately avoid `huggingface-hub` here so the
script works offline and without HF's gated-repo handshake.

Model: `dinov3_vitb16` pretrained on LVD-1689M (86 M params, ViT-B/16,
hash `73cec8be`). Chosen over the Small variant for feature quality
and over Large for first-integration speed; revisit in M4.

Runs
----
    source scripts/env-setup.sh
    uv run python scripts/m3_dinov3_latents.py

    # override checkpoint / output / frame count / weights path:
    uv run python scripts/m3_dinov3_latents.py \
        --ckpt checkpoints/m1-g1-full/000043253760 \
        --num_frames 200 \
        --weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
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


# brax 0.14.2 × JAX 0.10 shim (same as m1_train.py).
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


import mujoco
import numpy as np
import torch
import torchvision.transforms as T
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = "G1JoystickFlatTerrain"
ENV_OVERRIDES = {"njmax": 96}
G1_SCENE = ROOT / "external" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

# DINOv3 local paths — avoids HF gated-repo handshake.
DINOV3_HUB_DIR = Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main"
DEFAULT_DINOV3_WEIGHTS = (
    Path.home()
    / "ai-workshop/jepa/vl-jepa/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
DINOV3_ENTRYPOINT = "dinov3_vitb16"

# DINOv3 uses ImageNet-style normalization at 224x224.
DINOV3_MEAN = [0.485, 0.456, 0.406]
DINOV3_STD = [0.229, 0.224, 0.225]
DINOV3_INPUT = 224

# Same head-cam pose as M2 (see scripts/m2_head_camera.py).
HEAD_CAM_POS = [0.10, 0.0, 0.18]
HEAD_CAM_QUAT = [-0.5, -0.5, 0.5, 0.5]


def build_model_with_head_cam(scene_xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    torso = spec.body("torso_link")
    if torso is None:
        raise RuntimeError("torso_link body not found in G1 MJCF")
    cam = torso.add_camera()
    cam.name = "head"
    cam.pos = HEAD_CAM_POS
    cam.quat = HEAD_CAM_QUAT
    return spec.compile()


def load_params(ckpt_path: Path):
    if ckpt_path.is_dir():
        return brax_checkpoint.load(str(ckpt_path.resolve()))
    with ckpt_path.open("rb") as f:
        return pickle.load(f)


def make_policy(params, env):
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
    return jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "checkpoints" / "m1-g1-full" / "000043253760",
    )
    ap.add_argument(
        "--out_npz",
        type=Path,
        default=ROOT
        / "docs"
        / "journal"
        / "assets"
        / "2026-04-16-m3-dinov3-latents.npz",
    )
    ap.add_argument("--num_frames", type=int, default=200)
    ap.add_argument("--render_h", type=int, default=240)
    ap.add_argument("--render_w", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_DINOV3_WEIGHTS,
        help="Path to the local DINOv3 .pth weights file",
    )
    ap.add_argument(
        "--hub_dir",
        type=Path,
        default=DINOV3_HUB_DIR,
        help="Path to the cached facebookresearch_dinov3_main hub directory",
    )
    args = ap.parse_args()

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load DINOv3 from local hub + weights ----
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"DINOv3 weights file not found: {args.weights}. "
            "Download a fresh copy from Meta or point --weights to an existing "
            "dinov3_vitb16_pretrain_lvd1689m*.pth."
        )
    if not args.hub_dir.is_dir():
        raise FileNotFoundError(
            f"DINOv3 hub dir not found: {args.hub_dir}. "
            "Expected torch.hub cache at "
            "~/.cache/torch/hub/facebookresearch_dinov3_main/."
        )

    print(f"loading DINOv3 {DINOV3_ENTRYPOINT} from")
    print(f"  hub:     {args.hub_dir}")
    print(f"  weights: {args.weights}")
    t0 = time.perf_counter()
    dinov3 = torch.hub.load(
        str(args.hub_dir),
        DINOV3_ENTRYPOINT,
        source="local",
        weights=str(args.weights),
    )
    dinov3 = dinov3.eval().to("cuda")
    load_wall = time.perf_counter() - t0
    n_params = sum(p.numel() for p in dinov3.parameters())
    print(f"  loaded in {load_wall:.1f}s  ({n_params:,} params)")

    # Simple image preprocessing pipeline: resize to 224x224, ToTensor,
    # ImageNet normalize. The reference DINOv3 repo uses this exact
    # recipe for its eval scripts.
    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Resize(
                (DINOV3_INPUT, DINOV3_INPUT),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD),
        ]
    )

    # ---- Sanity forward pass on a random image ----
    print("sanity forward pass...")
    dummy = np.random.randint(
        0, 255, (args.render_h, args.render_w, 3), dtype=np.uint8
    )
    x = preprocess(dummy).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        feats = dinov3.forward_features(x)
    if isinstance(feats, dict):
        print("  forward_features returned dict with keys:")
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v).__name__}")
    else:
        print(f"  forward_features returned tensor: shape={tuple(feats.shape)}")

    # ---- Head-cam rendering setup (same pattern as m2_head_camera.py) ----
    mj_model = build_model_with_head_cam(G1_SCENE)
    mj_data = mujoco.MjData(mj_model)
    cam_id = mj_model.camera("head").id
    renderer = mujoco.Renderer(mj_model, height=args.render_h, width=args.render_w)

    # ---- Env + trained policy ----
    env = registry.load(ENV_NAME, config_overrides=ENV_OVERRIDES)
    params = load_params(args.ckpt)
    inference_fn = make_policy(params, env)

    # ---- Rollout, collect frames ----
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    state = reset_fn(rng)

    frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    for _ in range(args.num_frames):
        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, action)
        mjx_data = getattr(state, "data", None) or getattr(
            state, "pipeline_state", None
        )
        if mjx_data is None:
            raise RuntimeError(
                "Cannot locate mjx.Data on state — expected state.data or "
                "state.pipeline_state."
            )
        mj_data.qpos[:] = np.array(mjx_data.qpos)
        mj_data.qvel[:] = np.array(mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam_id)
        frames.append(renderer.render().copy())
    rollout_wall = time.perf_counter() - t0
    print(f"rendered {len(frames)} head-cam frames in {rollout_wall:.1f}s")

    # ---- DINOv3 encoding in batches ----
    print(f"encoding through DINOv3 in batches of {args.batch_size}...")
    cls_list: list[np.ndarray] = []
    patches_mean_list: list[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(frames), args.batch_size):
            batch = frames[i : i + args.batch_size]
            # Stack after preprocessing each frame; preprocess expects HWC uint8.
            batch_tensor = torch.stack([preprocess(f) for f in batch]).to("cuda")
            feats = dinov3.forward_features(batch_tensor)
            # DINOv3 feature dict keys: `x_norm_clstoken` (CLS),
            # `x_norm_patchtokens` (patch tokens), and `x_norm_regtokens`
            # (the 4 storage / register tokens). We keep CLS and mean-pool
            # the patch tokens for a second complementary view.
            cls = feats["x_norm_clstoken"]  # (B, D)
            patches = feats["x_norm_patchtokens"]  # (B, N_patches, D)
            patches_mean = patches.mean(dim=1)  # (B, D)
            cls_list.append(cls.float().cpu().numpy())
            patches_mean_list.append(patches_mean.float().cpu().numpy())
    enc_wall = time.perf_counter() - t0
    cls_arr = np.concatenate(cls_list, axis=0)
    patches_arr = np.concatenate(patches_mean_list, axis=0)
    print(f"  encoded in {enc_wall:.1f}s  ({len(frames) / enc_wall:.1f} frames/s)")
    print(f"  cls_latents:     shape={cls_arr.shape} dtype={cls_arr.dtype}")
    print(f"  patch_mean_lat:  shape={patches_arr.shape} dtype={patches_arr.dtype}")

    # ---- Save ----
    np.savez(
        str(args.out_npz),
        cls=cls_arr.astype(np.float32),
        patches_mean=patches_arr.astype(np.float32),
        model_entrypoint=np.array(DINOV3_ENTRYPOINT, dtype=object),
        weights_path=np.array(str(args.weights), dtype=object),
        num_frames=np.int64(len(frames)),
        render_shape=np.array([args.render_h, args.render_w], dtype=np.int32),
        ckpt_path=np.array(str(args.ckpt), dtype=object),
        seed=np.int64(args.seed),
    )
    size_kb = args.out_npz.stat().st_size // 1024
    print(f"saved {args.out_npz.relative_to(ROOT)} ({size_kb} KB)")

    # Quick stats on the encoded latents for sanity.
    print()
    print("latent stats (cls, across frames):")
    print(
        f"  mean={cls_arr.mean():.3f}  std={cls_arr.std():.3f}  "
        f"min={cls_arr.min():.3f}  max={cls_arr.max():.3f}"
    )
    print("  per-frame L2 norm (first 5):", np.linalg.norm(cls_arr, axis=1)[:5])
    print()
    print("M3 exit criterion: head-cam frames → DINOv3 → per-frame latents saved. ✓")


if __name__ == "__main__":
    main()
