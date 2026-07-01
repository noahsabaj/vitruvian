# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""``vit-collect`` — G1 rollout data collector.

Usage::

    uv run vit-collect configs/collect/diverse.yaml
    uv run vit-collect configs/collect/narrow.yaml --override seed=1
    uv run vit-collect configs/collect/diverse.yaml --single-process

Builds a :class:`CollectionConfig` from the YAML, then dispatches to
:func:`run_collection`. The orchestrator spawns subprocess workers by
default; pass ``--single-process`` for CI-sized configs that don't need
Warp VRAM isolation between chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vitruvian.data import (
    CollectionConfig,
    CommandSpec,
    diverse_config,
    narrow_config,
)
from vitruvian.data.collection import run_collection
from vitruvian.utils import load_config


def _cfg_from_dict(cfg: dict) -> CollectionConfig:
    mode = cfg.get("mode", "diverse")
    out_h5 = Path(cfg["out_h5"]).expanduser()
    kwargs = {
        k: v
        for k, v in cfg.items()
        if k
        in (
            "episodes_per_command",
            "episode_steps",
            "chunk_size",
            "seed",
            "img_size",
            "keep_chunks",
        )
    }
    if "policy_ckpt" in cfg and cfg["policy_ckpt"] is not None:
        kwargs["policy_ckpt"] = Path(cfg["policy_ckpt"]).expanduser()

    if mode == "narrow":
        return narrow_config(out_h5, **kwargs)
    if mode == "diverse":
        return diverse_config(out_h5, **kwargs)
    commands = tuple(CommandSpec(**c) for c in cfg["commands"])
    return CollectionConfig(out_h5=out_h5, commands=commands, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect G1 rollouts.")
    ap.add_argument("config", type=Path, help="YAML config path")
    ap.add_argument("--override", "-o", action="append", default=[])
    ap.add_argument(
        "--single-process",
        action="store_true",
        help="Skip subprocess isolation between chunks. Only for small "
        "configs (< 20 eps total); larger runs will hit Warp VRAM creep.",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Accept a partial dataset when some chunks fail. Without "
        "this flag, any chunk subprocess returning a non-zero rc causes "
        "the run to raise after the merge step (the surviving chunks "
        "are still written to the output HDF5).",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of chunk subprocesses to run CONCURRENTLY (default 1 "
        "= serial). Each chunk is GPU-isolated, so this is bounded by GPU "
        "memory: ~16-24 on a 96 GB card, ~2-3 on 8 GB. The main collection "
        "speed lever. Ignored with --single-process.",
    )
    args = ap.parse_args()

    cfg_dict = load_config(args.config, overrides=args.override)
    cfg = _cfg_from_dict(cfg_dict)
    run_collection(
        cfg,
        single_process=args.single_process,
        allow_partial=args.allow_partial,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
