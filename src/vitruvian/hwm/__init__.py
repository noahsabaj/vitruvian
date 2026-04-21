# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Hierarchical World Models (HWM) package for Vitruvian.

Adapts HWM (arXiv:2604.03208) hierarchical CEM planning onto a
LeWM (JEPA) low-level world model on Unitree G1. See
docs/decisions/008-hwm-planning-layer.md and
docs/decisions/007-lewm-world-model.md for rationale.
"""

from .action_codec import MacroActionEncoder
from .backbone_adapter import LeWMBackboneAdapter, load_lewm_jepa_from_checkpoint
from .backbone_dinov3 import DINOv3Backbone
from .backbone_dinov3_patches import DINOv3PatchBackbone
from .cache import CacheKey, EmbeddingCache
from .compile_utils import bf16_autocast, compile_model, compiled_no_grad_forward
from .data import G1WaypointDataset
from .encoder_history import EncoderHistory
from .goal_builder import MacroNNRetriever
from .high_level import HighLevelModel
from .jepa_v4 import JEPAv4, load_jepa_v4_from_checkpoint
from .jepa_v5 import JEPAv5, JEPAv5PlannerBackbone, load_jepa_v5_from_checkpoint
from .mlp_predictor import MLPPredictor
from .patch_predictor import PatchARPredictor
from .mppi import MPPI
from .objectives import LossInfo, PredictionLoss, VICRegLoss
from .planners import (
    GoalL2Cost,
    HierarchicalPlanner,
    HighLevelPlanner,
    LowLevelPlanner,
    encode_goal,
)

__all__ = [
    "CacheKey",
    "DINOv3Backbone",
    "DINOv3PatchBackbone",
    "EmbeddingCache",
    "EncoderHistory",
    "G1WaypointDataset",
    "GoalL2Cost",
    "HierarchicalPlanner",
    "HighLevelModel",
    "HighLevelPlanner",
    "JEPAv4",
    "JEPAv5",
    "JEPAv5PlannerBackbone",
    "LeWMBackboneAdapter",
    "LossInfo",
    "LowLevelPlanner",
    "MLPPredictor",
    "MPPI",
    "MacroActionEncoder",
    "MacroNNRetriever",
    "PatchARPredictor",
    "PredictionLoss",
    "VICRegLoss",
    "bf16_autocast",
    "compile_model",
    "compiled_no_grad_forward",
    "encode_goal",
    "load_jepa_v4_from_checkpoint",
    "load_jepa_v5_from_checkpoint",
    "load_lewm_jepa_from_checkpoint",
]
