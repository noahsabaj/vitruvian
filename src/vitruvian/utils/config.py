# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""YAML config loader with dotted-path overrides.

Deliberately minimal — no Hydra, no OmegaConf. A config is just a
``dict``, overrides are dotted strings like ``trainer.lr=3e-5``, and
type coercion is best-effort (ints, floats, bools, ``null``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def _coerce_scalar(s: str) -> Any:
    """Best-effort parse of an override value."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none"):
        return None
    for caster in (int, float):
        try:
            return caster(s)
        except ValueError:
            pass
    return s


def _apply_override(cfg: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(
            f"override must be 'key.path=value'; got {dotted!r}"
        )
    key, _, value = dotted.partition("=")
    parts = key.split(".")
    node: Any = cfg
    for p in parts[:-1]:
        existing = node.get(p) if isinstance(node, dict) else None
        if existing is None:
            new_node: dict[str, Any] = {}
            node[p] = new_node
            node = new_node
        elif isinstance(existing, dict):
            node = existing
        else:
            # Refuse to silently overwrite a scalar/list with a mapping — that
            # almost always means the override path has a typo.
            raise ValueError(
                f"override {dotted!r}: cannot descend into {'.'.join(parts[:parts.index(p)+1])!r} "
                f"because it is a {type(existing).__name__}, not a mapping"
            )
    node[parts[-1]] = _coerce_scalar(value)


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load a YAML config and apply dotted overrides.

    Args:
        path: Path to a YAML file.
        overrides: List of ``key.path=value`` strings. Later overrides
            win. Types are coerced (``3e-5`` → float; ``true`` → bool;
            ``null`` → None).
    """
    import yaml

    path = Path(path).expanduser()
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(cfg)
    for override in overrides or []:
        _apply_override(cfg, override)
    return cfg


__all__ = ["load_config"]
