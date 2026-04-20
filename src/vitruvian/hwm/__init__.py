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
from .data import G1WaypointDataset
from .goal_builder import MacroNNRetriever
from .high_level import HighLevelModel
from .jepa_v4 import JEPAv4, load_jepa_v4_from_checkpoint
from .mlp_predictor import MLPPredictor
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
    "DINOv3Backbone",
    "G1WaypointDataset",
    "GoalL2Cost",
    "HierarchicalPlanner",
    "HighLevelModel",
    "HighLevelPlanner",
    "JEPAv4",
    "LeWMBackboneAdapter",
    "LossInfo",
    "LowLevelPlanner",
    "MLPPredictor",
    "MPPI",
    "MacroActionEncoder",
    "MacroNNRetriever",
    "PredictionLoss",
    "VICRegLoss",
    "encode_goal",
    "load_jepa_v4_from_checkpoint",
    "load_lewm_jepa_from_checkpoint",
]
