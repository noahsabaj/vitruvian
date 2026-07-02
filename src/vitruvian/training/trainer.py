# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""``JEPATrainer`` — the shared training loop for v4 / v5 / unified JEPA.

Handles: BF16 autocast, ``torch.compile`` on the predictor, fused
AdamW, cosine LR with warmup, per-epoch checkpoint save with compile
key-prefix normalization, and basic epoch-level metric printing.

Subsumes the four near-identical ``scripts/m4*train*.py`` loops.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vitruvian.utils.compile_utils import bf16_autocast, compile_model


@dataclass
class TrainerConfig:
    """Knobs for the shared trainer loop.

    All fields have sensible defaults matching the M4.6 / M4.7 recipes.
    """

    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    lr_floor: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    bf16: bool = True
    compile_predictor: bool = True
    num_workers: int = 4
    val_frac: float = 0.05
    seed: int = 0


LossFn = Callable[[nn.Module, dict[str, Any]], dict[str, Any]]


def cosine_lr_factor(
    step: int, total: int, warmup: int, peak: float, floor: float
) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (floor + (peak - floor) * cos) / peak


def _normalize_state_key(k: str) -> str:
    """Strip compile wrappers so loaders see eager-module keys."""
    return k.replace("_orig_mod.", "")


class JEPATrainer:
    """Shared train loop for any JEPA-shaped composer.

    Args:
        jepa: The model. Trainable submodules are anything with
            ``requires_grad``; the backbone is typically frozen.
        loss_fn: Callable ``(jepa, batch) -> dict``. Must return a
            dict containing at minimum ``"loss"``; any other key
            whose value is a scalar tensor is averaged across the
            epoch and logged.
        cfg: :class:`TrainerConfig`.
        out_dir: Directory for epoch/best/latest checkpoints.
        run_name: Logged + saved to every checkpoint.
        jepa_config: The config dict that ``build_jepa`` can rebuild
            this model from. Saved into every checkpoint alongside the
            state dict so ``load_jepa`` works.
        frozen_key_prefixes: Prefixes whose state_dict keys are NOT
            persisted (DINOv3 backbone is reloaded from HF, so
            ``("backbone.dinov3.",)`` is the default).
    """

    def __init__(
        self,
        jepa: nn.Module,
        loss_fn: LossFn,
        cfg: TrainerConfig,
        out_dir: Path,
        *,
        run_name: str = "jepa",
        jepa_config: dict[str, Any] | None = None,
        frozen_key_prefixes: tuple[str, ...] = ("backbone.dinov3.",),
    ) -> None:
        # Any: the JEPA composer exposes .backbone.dinov3, .predictor,
        # etc. as submodules that mypy narrows to Tensor|Module under
        # nn.Module's dynamic-attr access. The dynamic-attr duck-typing
        # is the whole point of the shared trainer.
        self.jepa: Any = jepa
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.run_name = run_name
        self.jepa_config = dict(jepa_config) if jepa_config is not None else {}
        self.frozen_key_prefixes = tuple(frozen_key_prefixes)

    def _trainable(self) -> list[nn.Parameter]:
        return [p for p in self.jepa.parameters() if p.requires_grad]

    def _build_optimizer(self) -> AdamW:
        return AdamW(
            self._trainable(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            fused=True,
        )

    def _maybe_compile(self) -> None:
        if self.cfg.compile_predictor and hasattr(self.jepa, "predictor"):
            self.jepa.predictor = compile_model(
                self.jepa.predictor, mode="reduce-overhead"
            )

    def _autocast(self) -> Any:
        return bf16_autocast() if self.cfg.bf16 else _noop_cm()

    def _collect_trainable_state(self) -> dict[str, torch.Tensor]:
        sd = self.jepa.state_dict()
        out: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if any(k.startswith(pref) for pref in self.frozen_key_prefixes):
                continue
            out[_normalize_state_key(k)] = v.detach().cpu()
        return out

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        *,
        suffixes: tuple[str, ...] = ("latest", "epoch_{epoch:03d}"),
    ) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "config": self.jepa_config,
            "state_dict": self._collect_trainable_state(),
            "epoch": epoch,
            "metrics": metrics,
            "run_name": self.run_name,
            # arch_version 2: visual-only prediction target + proprio/action
            # as predictor conditioning (M5). Pre-2 checkpoints trained a
            # proprio-fused target and are semantically incompatible.
            "arch_version": 2,
        }
        for suffix in suffixes:
            path = self.out_dir / (suffix.format(epoch=epoch) + ".pt")
            torch.save(ckpt, path)

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        device: str = "cuda",
    ) -> dict[str, float]:
        """Run the train loop. Returns best-val metrics."""
        torch.manual_seed(self.cfg.seed)

        # The frozen DINOv3 backbone is pinned to eval() by the backbone's
        # own train()/eval() override (see DINOv3*Backbone.train), which
        # nn.Module propagation triggers on every jepa.train()/eval() below —
        # no manual dinov3.eval() calls needed here.
        jepa = self.jepa.to(device)

        opt = self._build_optimizer()
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.cfg.epochs
        warmup = min(self.cfg.warmup_steps, total_steps // 10)
        self._maybe_compile()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        best_score_key = "val_pred_loss+rollout"
        step = 0
        t0 = time.perf_counter()

        for epoch in range(1, self.cfg.epochs + 1):
            jepa.train()  # backbone.train() override keeps frozen DINOv3 in eval

            tr_sums: dict[str, float] = {}
            tr_n = 0
            for batch in train_loader:
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                }
                lr_mult = cosine_lr_factor(
                    step, total_steps, warmup, self.cfg.lr, self.cfg.lr_floor
                )
                for pg in opt.param_groups:
                    pg["lr"] = self.cfg.lr * lr_mult

                with self._autocast():
                    info = self.loss_fn(jepa, batch)
                loss = info["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                for k, v in info.items():
                    if k == "loss" or not torch.is_tensor(v):
                        continue
                    tr_sums[k] = tr_sums.get(k, 0.0) + float(v.detach())
                tr_n += 1
                step += 1

            tr_metrics = {
                f"train_{k}": v / max(1, tr_n) for k, v in tr_sums.items()
            }

            jepa.eval()
            va_sums: dict[str, float] = {}
            va_n = 0
            with torch.no_grad(), self._autocast():
                for batch in val_loader:
                    batch = {
                        k: v.to(device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    info = self.loss_fn(jepa, batch)
                    for k, v in info.items():
                        if k == "loss" or not torch.is_tensor(v):
                            continue
                        va_sums[k] = va_sums.get(k, 0.0) + float(v.detach())
                    va_n += 1
            va_metrics = {
                f"val_{k}": v / max(1, va_n) for k, v in va_sums.items()
            }

            elapsed = time.perf_counter() - t0
            cur_lr = opt.param_groups[0]["lr"]
            metrics = {**tr_metrics, **va_metrics, "lr": cur_lr, "elapsed": elapsed}
            print(
                f"[epoch {epoch:>2}/{self.cfg.epochs}]  "
                + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                             if k not in ("elapsed",))
                + f"  ({elapsed:.1f}s)"
            )

            suffixes: list[str] = ["latest", "epoch_{epoch:03d}"]
            # Select "best" on the full prediction objective (1-step +
            # multi-horizon rollout), not just the 1-step ``val_pred_loss``.
            # For Fast-LeWM the 1-step term is a single horizon of 16; scoring
            # on it alone would ignore the long horizons that are the whole
            # point of the dense prefix loss. ``val_rollout_loss`` is 0 when the
            # rollout term is off, so this reduces to ``val_pred_loss`` for AR
            # runs with ``rollout_weight=0``.
            val_primary = va_metrics.get(
                "val_pred_loss", float("inf")
            ) + va_metrics.get("val_rollout_loss", 0.0)
            if val_primary < best_val:
                best_val = val_primary
                suffixes.append("best")
            self.save_checkpoint(epoch, metrics, suffixes=tuple(suffixes))

        print(
            f"=== training done in {time.perf_counter() - t0:.1f}s; "
            f"best {best_score_key}: {best_val:.4f} ==="
        )
        return {best_score_key: best_val}


class _noop_cm:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


__all__ = ["JEPATrainer", "TrainerConfig", "cosine_lr_factor"]
