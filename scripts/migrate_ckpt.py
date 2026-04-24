#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""One-shot checkpoint-schema migrator.

M4.8's ``load_jepa`` reads legacy v4/v5 checkpoint configs and migrates
them to the unified schema in memory. This script writes that migrated
form back to disk so the on-disk artifacts stop depending on the
legacy-dispatch logic.

Usage::

    # Migrate a single file (writes .bak next to it, then overwrites).
    uv run python scripts/migrate_ckpt.py ~/.vitruvian/m4f_v5/best.pt

    # Write to a new path instead of overwriting (no .bak needed).
    uv run python scripts/migrate_ckpt.py ~/.vitruvian/m4f_v5/best.pt --copy

    # Migrate every *.pt in a directory.
    uv run python scripts/migrate_ckpt.py ~/.vitruvian/m4f_v5/ --dir

The script is **idempotent**: checkpoints already in the unified schema
(where ``config["backbone"]`` is a dict with a ``name`` field) are
skipped with a log line and no file changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vitruvian.models.registry import (  # noqa: E402
    _detect_legacy_schema,
    _migrate_v4_config,
    _migrate_v5_config,
    _normalize_state_dict,
)


def is_unified_schema(cfg: dict) -> bool:
    """A checkpoint is already in the unified schema iff ``config`` has
    a dict at ``config["backbone"]`` with a ``"name"`` field."""
    bb = cfg.get("backbone")
    return isinstance(bb, dict) and "name" in bb


def migrate_ckpt(
    src: Path,
    dst: Path | None = None,
    *,
    make_backup: bool = True,
    dry_run: bool = False,
) -> bool:
    """Load ``src``, migrate its config + state_dict, save to ``dst``.

    Args:
        src: Source checkpoint path.
        dst: Destination path. If None, overwrites ``src`` (after
            writing a ``.bak`` copy when ``make_backup``).
        make_backup: If True and dst is None, write ``src.bak``
            before overwriting.
        dry_run: Print what would happen without writing.

    Returns:
        True if a migration was performed, False if the ckpt was
        already in the unified schema (no-op).
    """
    print(f"[migrate] reading {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "config" not in ckpt or "state_dict" not in ckpt:
        raise RuntimeError(
            f"{src} is missing 'config' or 'state_dict' — not a Vitruvian "
            f"JEPA checkpoint?"
        )

    cfg = dict(ckpt["config"])
    if is_unified_schema(cfg):
        print(f"  already unified schema — skipping")
        return False

    legacy = _detect_legacy_schema(cfg)
    if legacy == "v4":
        new_cfg = _migrate_v4_config(cfg)
    elif legacy == "v5":
        new_cfg = _migrate_v5_config(cfg)
    else:
        raise RuntimeError(
            f"{src}: legacy schema not recognized (keys: {sorted(cfg)[:10]})"
        )
    new_state = _normalize_state_dict(ckpt["state_dict"])

    new_ckpt = dict(ckpt)
    new_ckpt["config"] = new_cfg
    new_ckpt["state_dict"] = new_state
    new_ckpt["migrated_from"] = legacy

    out = dst if dst is not None else src
    if dry_run:
        print(f"  [dry-run] would write migrated v{legacy} → {out}")
        return True

    if dst is None and make_backup:
        bak = src.with_suffix(src.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"  wrote backup: {bak}")
        else:
            print(f"  backup already exists at {bak}; not overwriting")

    torch.save(new_ckpt, out)
    print(f"  wrote migrated v{legacy} → {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate legacy v4/v5 JEPA checkpoints to the unified schema."
    )
    ap.add_argument("path", type=Path, help="Checkpoint file (or directory with --dir)")
    ap.add_argument(
        "--dir",
        action="store_true",
        help="Treat ``path`` as a directory and migrate every *.pt file inside "
        "(non-recursive).",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Write result to ``<path>.migrated.pt`` next to the source instead "
        "of overwriting the source file. No backup written.",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="With in-place migration, skip the .bak file. Use with caution.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing.",
    )
    args = ap.parse_args()

    path = args.path.expanduser().resolve()
    if args.dir:
        if not path.is_dir():
            raise SystemExit(f"{path} is not a directory")
        ckpts = sorted(path.glob("*.pt"))
        if not ckpts:
            raise SystemExit(f"No *.pt files in {path}")
    else:
        if not path.is_file():
            raise SystemExit(f"{path} is not a file (pass --dir for a directory)")
        ckpts = [path]

    for ckpt_path in ckpts:
        dst = (
            ckpt_path.with_suffix(ckpt_path.suffix + ".migrated.pt")
            if args.copy
            else None
        )
        try:
            migrate_ckpt(
                ckpt_path,
                dst=dst,
                make_backup=not args.no_backup,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[error] {ckpt_path}: {e}")


if __name__ == "__main__":
    main()
