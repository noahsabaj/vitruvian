#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.3½ — load the trained LeWM-G1 checkpoint and verify it produces
meaningful latent rollouts on fresh G1 head-cam data.

This is a bridge between M4.3 (training loss decreasing → JEPA is
learning) and M4.4 (CEM-planned G1 walking → the actual policy
result). It doesn't plan; it doesn't evaluate reward; it only asks:
"does the trained world model produce stable, action-discriminative
latent predictions on held-out frames?"

Pass criteria:

1. Checkpoint loads into the Module/JEPA hierarchy without size
   mismatch.
2. Encoding N fresh G1 head-cam frames produces a (N, D) latent
   tensor with finite, non-degenerate values.
3. Rolling out the same initial latent under TWO different random
   action sequences produces distinguishable predicted latents
   (cosine distance between the two predicted trajectories > ε).

If all three pass, the trained JEPA is usable. CEM planning can
proceed in a future session; the blocker is cost-function design,
not model functionality.

Runs
----
    /home/nsabaj/ai-workshop/vitruvian/external/le-wm/.venv/bin/python \
        scripts/m4_lewm_inference_probe.py \
        --ckpt ~/.stable_worldmodel/lewm_g1_v1_weights.ckpt

    # Defaults to loading lewm_g1_v1_weights.ckpt.

This script runs *inside* the LeWM venv (Python 3.10 + stable-pretraining),
not the main vitruvian venv. The shebang assumes you invoke it by an
explicit interpreter path.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms.v2 as T


STABLEWM_HOME = Path.home() / ".stable_worldmodel"
DEFAULT_CKPT = STABLEWM_HOME / "lewm_g1_v1_weights.ckpt"
DEFAULT_DATASET = STABLEWM_HOME / "g1_joystick_expert.h5"


def find_cached_history_size_and_num_preds(ckpt: dict) -> tuple[int, int]:
    """LeWM stores history_size and num_preds inside the Hydra config on
    the checkpoint; pull them out so the rollout call uses the values
    the model was trained with."""
    hp = ckpt.get("hyper_parameters", {})
    # hp can be wrapped in OmegaConf-serialized dicts
    wm = hp.get("wm", None) if isinstance(hp, dict) else None
    if isinstance(wm, dict):
        return int(wm.get("history_size", 3)), int(wm.get("num_preds", 3))
    # Reasonable defaults for LeWM's lejepa recipe.
    return 3, 3


def load_jepa_module(ckpt_path: Path) -> tuple[torch.nn.Module, int, int]:
    """Instantiate the stable_pretraining.Module + jepa.JEPA structure
    matching the trained checkpoint, then load weights."""
    # Import inside the function so the script at least errors early
    # if this isn't the LeWM venv.
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import stable_pretraining as spt

    # Hydra re-init: load the train config the checkpoint came from.
    # LeWM stores hyper_parameters inside the Lightning ckpt — we
    # rebuild the Module by instantiating it from the config snippet
    # in the checkpoint.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    hp = ckpt["hyper_parameters"]
    # hp can be a dict or DictConfig; normalize to DictConfig for hydra.
    cfg = OmegaConf.create(hp) if isinstance(hp, dict) else hp

    # stable_pretraining.Module is the top-level class LeWM uses.
    module = hydra.utils.instantiate(cfg, _recursive_=False)
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.eval()

    history_size, num_preds = find_cached_history_size_and_num_preds(ckpt)
    return module, history_size, num_preds


def preprocess_pixels(raw: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (T, 3, H, W) float32 normalized."""
    import stable_pretraining as spt

    xform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(size=img_size),
        ]
    )
    # Apply per-frame then stack.
    frames = torch.stack([xform(frame) for frame in raw])
    return frames


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance between two same-shaped tensors."""
    af = a.flatten()
    bf = b.flatten()
    return float(1.0 - torch.nn.functional.cosine_similarity(af, bf, dim=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--ep_idx", type=int, default=0, help="Which episode to probe")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_samples", type=int, default=8)
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt}\n"
            "Train LeWM on G1 first (scripts invoke train.py with data=g1)."
        )

    # --- Load checkpoint + module ---
    print(f"loading trained LeWM checkpoint: {args.ckpt}")
    t0 = time.perf_counter()
    module, history_size, num_preds = load_jepa_module(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)
    print(
        f"  loaded in {time.perf_counter() - t0:.1f}s on {device}  "
        f"(history_size={history_size}, num_preds={num_preds})"
    )
    jepa = module.model  # the wrapped jepa.JEPA

    # --- Pull a fresh trajectory from the expert dataset ---
    print(f"\nloading episode {args.ep_idx} from {args.dataset.name}")
    with h5py.File(args.dataset, "r") as f:
        ep_len = int(f["ep_len"][args.ep_idx])
        ep_off = int(f["ep_offset"][args.ep_idx])
        # Grab history_size + num_preds + 2 margin frames for rollout.
        n_frames = history_size + num_preds + 2
        n_frames = min(n_frames, ep_len)
        pixels_raw = f["pixels"][ep_off : ep_off + n_frames]  # (T, H, W, 3)
        action_raw = f["action"][ep_off : ep_off + n_frames]  # (T, 29)
    print(
        f"  pulled {n_frames} frames, pixels {pixels_raw.shape} "
        f"{pixels_raw.dtype}, actions {action_raw.shape}"
    )

    # --- Preprocess + encode (check 1 + 2) ---
    pixels = preprocess_pixels(pixels_raw, img_size=args.img_size).to(device)
    actions = torch.from_numpy(action_raw).float().to(device)

    print("\n[CHECK 1/3] encoding fresh frames through JEPA encoder...")
    with torch.inference_mode():
        info = {
            "pixels": pixels.unsqueeze(0),  # (1, T, 3, H, W)
            "action": actions.unsqueeze(0),  # (1, T, 29)
        }
        encoded = jepa.encode(info)
    emb = encoded["emb"]  # (1, T, D)
    print(f"  emb shape: {tuple(emb.shape)}  dtype: {emb.dtype}")
    print(
        f"  finite: {bool(torch.all(torch.isfinite(emb)))}  "
        f"mean={float(emb.mean()):.3f}  std={float(emb.std()):.3f}  "
        f"norm(per-frame)={emb.norm(dim=-1).flatten().tolist()}"
    )
    assert torch.all(torch.isfinite(emb)), "encoder produced non-finite latents"
    assert emb.std() > 0.01, "encoder latents are degenerate (low variance)"
    print("  PASS")

    # --- Rollout under two different action sequences (check 3) ---
    print(f"\n[CHECK 3/3] rolling out {num_preds} steps under 2 distinct action plans ...")
    torch.manual_seed(0)
    # Shape (B=1, S=2, T=history+num_preds, action_dim=29).
    # First action slot copies history; sample future actions.
    hist_act = actions[:history_size].unsqueeze(0).unsqueeze(0)  # (1,1,H,29)
    hist_act_2 = hist_act.expand(-1, 2, -1, -1).clone()

    future_act_a = torch.randn(
        1, 1, num_preds, actions.shape[-1], device=device
    ) * 0.3
    future_act_b = torch.randn(
        1, 1, num_preds, actions.shape[-1], device=device
    ) * 0.3
    future_stacked = torch.cat([future_act_a, future_act_b], dim=1)  # (1,2,T,29)
    action_sequence = torch.cat([hist_act_2, future_stacked], dim=2)

    hist_pixels = pixels[:history_size].unsqueeze(0).unsqueeze(0).expand(
        -1, 2, -1, -1, -1, -1
    )  # (1, 2, H, 3, H, W)
    rollout_info = {"pixels": hist_pixels, "action": action_sequence}

    with torch.inference_mode():
        preds = jepa.rollout(rollout_info, action_sequence, history_size=history_size)

    # preds shape depends on implementation; the key thing is the two
    # samples disagree once the futures diverge.
    print(f"  preds keys: {list(preds.keys()) if isinstance(preds, dict) else 'tensor'}")
    if isinstance(preds, dict):
        preds_emb = preds.get("preds") or preds.get("emb")
    else:
        preds_emb = preds
    print(f"  preds_emb shape: {tuple(preds_emb.shape)}")

    # Split across the sample axis.
    a_pred = preds_emb[0, 0]
    b_pred = preds_emb[0, 1]
    d = cosine_distance(a_pred, b_pred)
    print(
        f"  cosine_distance(a_pred, b_pred) = {d:.4f}  "
        f"(> 0 means different actions produce different futures)"
    )
    assert d > 1e-4, "rollouts don't discriminate actions — model isn't action-aware"
    print("  PASS")

    print("\n✓ M4.3½ sanity probe: trained LeWM is usable for CEM planning.")


if __name__ == "__main__":
    main()
