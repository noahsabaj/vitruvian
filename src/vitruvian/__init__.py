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
  ``vit-collect`` / ``vit-rollout`` console-script entrypoints.
* ``vitruvian.utils`` — the YAML config loader and compile/autocast
  helpers.
"""

__version__ = "0.4.8"


def main() -> None:
    """Entry point for the bare ``vitruvian`` console script."""
    print(
        "Vitruvian — a minimum viable humanoid substrate for machine "
        "intelligence.\n"
        "CLI tools: vit-train, vit-plan, vit-eval, vit-collect, vit-rollout.\n"
        "See the README and docs/roadmap.md to get started."
    )
