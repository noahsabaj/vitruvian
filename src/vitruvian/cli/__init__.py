# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""CLI layer — ``vit-train`` / ``vit-plan`` / ``vit-eval`` / ``vit-collect``
/ ``vit-rollout``.

Entry points are wired via ``pyproject.toml:[project.scripts]``. Each
module is a thin argparse + run-loop wrapper; the real work happens
in :mod:`vitruvian.training`, :mod:`vitruvian.planning`, etc.
"""
