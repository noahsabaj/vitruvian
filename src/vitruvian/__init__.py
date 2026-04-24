# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Vitruvian — humanoid machine-intelligence library.

Top-level subpackages:

* ``vitruvian.env`` — G1 sim env builders and camera constants.
* ``vitruvian.data`` — datasets, the precomputed-embedding cache,
  and the rollout collector.
* ``vitruvian.models`` — backbones, predictors, the unified ``JEPA``
  composer, the value head, and the ``build_jepa``/``load_jepa``
  registry entrypoints.
* ``vitruvian.planning`` — ``MPPIPlanner``, the cost-strategy
  protocols, and the ``EncoderHistory`` ring buffer.
* ``vitruvian.training`` — ``JEPATrainer``, ``VFHERTrainer``, and
  the loss functions.
* ``vitruvian.lewm_compat`` — vendored subset of external LeWM
  (``Embedder``, ``ARPredictor``, ``MLP``, ``Transformer`` blocks).
* ``vitruvian.cli`` — ``vit-train`` / ``vit-plan`` / ``vit-eval`` /
  ``vit-collect`` console-script entrypoints.
* ``vitruvian.utils`` — config loader, compile/autocast helpers,
  structured run logging.
"""

__version__ = "0.4.8"


def main() -> None:
    print("Hello from vitruvian!")
