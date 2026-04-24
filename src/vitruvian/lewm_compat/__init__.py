# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Vendored subset of LeWM (external/le-wm) classes.

Lets Vitruvian train and load JEPA checkpoints (v3 legacy + v4 + v5)
without keeping ``external/le-wm/`` on ``sys.path``. Upstream
attribution is in the repo-root ``NOTICE`` file.
"""

from vitruvian.lewm_compat.jepa import JEPA, detach_clone
from vitruvian.lewm_compat.module import (
    MLP,
    ARPredictor,
    Attention,
    Block,
    ConditionalBlock,
    Embedder,
    FeedForward,
    SIGReg,
    Transformer,
    modulate,
)

__all__ = [
    "ARPredictor",
    "Attention",
    "Block",
    "ConditionalBlock",
    "Embedder",
    "FeedForward",
    "JEPA",
    "MLP",
    "SIGReg",
    "Transformer",
    "detach_clone",
    "modulate",
]
