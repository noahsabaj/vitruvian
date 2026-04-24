# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Training layer — JEPATrainer, VFHERTrainer, loss functions."""

from vitruvian.training.iql import (
    VFHERConfig,
    VFHERTrainer,
    ValueHead,
    ema_update,
    expectile_loss,
)
from vitruvian.training.losses import prediction_loss, vicreg_std_loss
from vitruvian.training.trainer import JEPATrainer, TrainerConfig, cosine_lr_factor

__all__ = [
    "JEPATrainer",
    "TrainerConfig",
    "VFHERConfig",
    "VFHERTrainer",
    "ValueHead",
    "cosine_lr_factor",
    "ema_update",
    "expectile_loss",
    "prediction_loss",
    "vicreg_std_loss",
]
