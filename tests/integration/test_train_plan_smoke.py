# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""End-to-end integration smoke.

Builds a v5-shaped JEPA against mocked backbones → runs one training
step with the real loss → saves + reloads via ``load_jepa`` → runs one
MPPI plan. All CPU, well under 60s.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vitruvian.data import G1EmbSeqDataset, G1PatchSeqDataset
from vitruvian.models import PlannerBackbone, build_jepa, load_jepa
from vitruvian.planning import EncoderHistory, MPPIPlanner, MSECost, encode_goal
from vitruvian.training import JEPATrainer, TrainerConfig, prediction_loss


def _cfg() -> dict:
    return {
        "backbone": {"name": "dinov3-patch", "kwargs": {}},
        "predictor": {
            "name": "patch-ar",
            "kwargs": {
                "num_frames": 3, "depth": 2, "heads": 4, "mlp_dim": 64,
                "input_dim": 32, "hidden_dim": 32, "output_dim": 32,
                "dim_head": 16, "dropout": 0.0, "adaln_rank": 16,
            },
        },
        "action_encoder": {
            "kwargs": {"action_dim": 29, "frameskip": 1, "smoothed_dim": 10,
                        "emb_dim": 32}
        },
        "proprio_encoder": {
            "kwargs": {"in_dim": 103, "hidden": 16, "out_dim": 32,
                        "norm_fn": None}
        },
        "patch_projector": {"kwargs": {"in_dim": 768, "out_dim": 32}},
    }


def test_train_plan_roundtrip(
    tmp_path: Path, synthetic_h5: Path, registry_with_fakes,
    fake_patch_backbone,
) -> None:
    # 1. Build model + tiny precomputed patch cache.
    m = build_jepa(_cfg())
    n_total = 50  # synthetic_h5 has 2 × 25
    patches = torch.randn(n_total, 49, 768, dtype=torch.float16)
    ds = G1PatchSeqDataset(synthetic_h5, patches, seq_len=6)
    # train + val split by index
    split = int(0.8 * len(ds))
    train_idx = list(range(split))
    val_idx = list(range(split, len(ds)))
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    loss_fn = partial(
        prediction_loss,
        history_size=3, num_preds=3,
        rollout_weight=1.0, reg_weight=1.0,
    )
    trainer_cfg = TrainerConfig(
        epochs=1, batch_size=4, lr=1e-3, lr_floor=1e-4, warmup_steps=1,
        bf16=False, compile_predictor=False, num_workers=0,
        val_frac=0.2, seed=0,
    )
    trainer = JEPATrainer(
        jepa=m,
        loss_fn=loss_fn,
        cfg=trainer_cfg,
        out_dir=tmp_path / "ckpt",
        run_name="smoke",
        jepa_config=_cfg(),
    )
    # Force CPU — ``.to('cpu')`` is a no-op since everything was built on CPU.
    trainer.fit(train_loader, val_loader, device="cpu")

    # 2. Checkpoint round-trips via load_jepa.
    latest = tmp_path / "ckpt" / "latest.pt"
    assert latest.exists()
    loaded = load_jepa(latest, device="cpu")
    assert hasattr(loaded, "predict")
    if hasattr(loaded.backbone, "load_eagerly"):
        loaded.backbone.load_eagerly()

    # 3. Build MPPI planner + encode-once history. One macro, 4-step horizon.
    pb = PlannerBackbone(loaded)
    goal_pixel = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
    goal_emb = encode_goal(pb, goal_pixel)

    planner = MPPIPlanner(
        jepa=loaded,
        backbone=pb,
        subgoal_emb=goal_emb,
        cost_fn=MSECost(),
        action_dim=29,
        horizon=4,
        num_samples=8,
        noise_sigma=0.02,
        iterations=1,
        history_size=3,
        device="cpu",
    )
    history = EncoderHistory(size=3, encoder=pb.encode)
    frame = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
    history.push(frame)
    U = planner.plan(
        pixel_history=None,
        action_history=torch.zeros(1, 29),
        encoded_history=history.latest_window(),
    )
    assert U.shape == (4, 29)
    assert torch.isfinite(U).all()
