# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Models layer — backbones, predictors, unified JEPA, registry."""

from vitruvian.models.backbones import (
    DEFAULT_DINOV3_ID,
    Backbone,
    DINOv3ClsBackbone,
    DINOv3PatchBackbone,
    LeWMBackbone,
    load_lewm_jepa_from_checkpoint,
)
from vitruvian.models.jepa import JEPA, PlannerBackbone
from vitruvian.models.predictors import (
    ARPredictor,
    PatchARPredictor,
    PrefixPatchPredictor,
)
from vitruvian.models.registry import BACKBONES, PREDICTORS, build_jepa, load_jepa

__all__ = [
    "ARPredictor",
    "BACKBONES",
    "Backbone",
    "DEFAULT_DINOV3_ID",
    "DINOv3ClsBackbone",
    "DINOv3PatchBackbone",
    "JEPA",
    "LeWMBackbone",
    "PREDICTORS",
    "PatchARPredictor",
    "PlannerBackbone",
    "PrefixPatchPredictor",
    "build_jepa",
    "load_jepa",
    "load_lewm_jepa_from_checkpoint",
]
