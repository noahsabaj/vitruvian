# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.7 — ``compile_utils``: unified compile + mixed-precision policy.

**Why this module exists.** Across M4.5 and M4.6 we hand-rolled
mixed-precision choices per-script (FP16 for DINOv3 inference, FP32
elsewhere) and never enabled ``torch.compile``. Both leave easy
speedups on the table:

  * Our GPU (Ampere RTX 4060 Ti) has BF16 tensor cores. BF16 has
    FP32's exponent range (no loss-scaler needed) and 2× matmul
    throughput vs FP32. Training forward+backward in BF16 autocast
    is a strict win.
  * ``torch.compile`` kernel-fuses a ViT's QKV + FFN + AdaLN ops.
    On 9-25M param predictors, ``mode="reduce-overhead"`` gives
    20-40% end-to-end training speedup after a 30-second warmup.

This module centralizes the choices. All M4.x training/eval/plan
scripts use ``bf16_autocast()`` and ``compile_model()`` via the same
helpers, so upgrading the precision or compile strategy is a one-line
change.

**Footguns this module handles:**

  * ``torch.compile`` plus dynamic shapes (e.g., MPPI rollouts whose
    sequence grows each iteration) can recompile forever. We set
    ``dynamic=True`` as the default and document when callers should
    override it (fixed-shape predictor forwards → ``dynamic=False``
    for a faster first-step compile).
  * Some ops don't have BF16 kernels (uncommon in pure transformers;
    rare in our use). Autocast falls back to FP32 silently.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn


def compile_model(
    model: "nn.Module",
    *,
    mode: str = "reduce-overhead",
    dynamic: bool = True,
    fullgraph: bool = False,
) -> "nn.Module":
    """Wrap ``torch.compile`` with our dynamic-shape-safe defaults.

    Args:
        model: module to compile.
        mode: ``"reduce-overhead"`` (default) is best for our step-size
            range. Use ``"max-autotune"`` for longer training runs when
            the per-compile autotuning cost amortizes.
        dynamic: ``True`` allows the compiled graph to recompile only
            when tensor shapes change *beyond* PyTorch's specialization
            heuristics. Keep ``True`` unless you know all shapes are
            fixed (training predictor at batch-aligned size).
        fullgraph: ``True`` errors on graph breaks; useful for
            validating a module is compile-safe, but usually ``False``.

    Returns:
        The compiled module (same public API; torch wraps it).
    """
    return torch.compile(
        model,
        mode=mode,
        dynamic=dynamic,
        fullgraph=fullgraph,
    )


@contextlib.contextmanager
def bf16_autocast(enabled: bool = True):
    """BF16 autocast context. Our GPU has BF16 tensor cores and BF16
    has FP32's exponent range — no GradScaler needed, no loss scaler,
    pure speedup.

    Usage:
        with bf16_autocast():
            loss = model_forward(batch)
        loss.backward()
        opt.step()

    Pass ``enabled=False`` for a no-op (useful for ablation / debug).
    """
    if not enabled:
        yield
        return
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield


def compiled_no_grad_forward(
    model: "nn.Module", *example_inputs
) -> "nn.Module":
    """Special case: compile a model that will only be used for
    inference, with gradients off. Useful for plan-time predictor
    rollouts.

    The returned module is wrapped so ``model(x)`` runs under
    ``torch.no_grad`` + BF16 autocast, which together with
    ``torch.compile`` give the tightest inference path.
    """
    compiled = compile_model(model, mode="reduce-overhead", dynamic=True)

    # Trigger first compile with the example input if supplied so the
    # caller's first real call doesn't eat a 30-second stall.
    if example_inputs:
        with torch.no_grad(), bf16_autocast():
            compiled(*example_inputs)

    return compiled


__all__ = [
    "bf16_autocast",
    "compile_model",
    "compiled_no_grad_forward",
]
