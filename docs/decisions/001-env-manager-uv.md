# ADR 001 — Environment Manager: `uv`

**Status:** Accepted
**Date:** 2026-04-16

## Context

The project needs a Python environment and dependency manager. The local
machine already has Anaconda (Python 3.12.7), `uv`, and system `pip`
available. The primary dependencies (MuJoCo, MJX, JAX, Brax, wandb) are
all on PyPI; none require conda.

## Decision

Use `uv` as the sole environment and dependency manager for this
project. Anaconda remains on the system for unrelated work but is not
used here.

## Consequences

- Dependency resolution 10-100× faster than conda.
- Lockfile-based reproducibility via `uv.lock` (tracked in git).
- JAX CUDA 12 wheels install cleanly from PyPI against the system CUDA
  12.8 toolkit.
- Community answers for JAX/MuJoCo questions are often conda-flavored;
  occasional translation required, but the ecosystem has shifted.

## Alternatives considered

- **conda / mamba** — older community default. Slow solver, ships its
  own CUDA libraries that can collide with system CUDA, heavier shell
  footprint. Rejected.
- **pip + venv** — works, but no lockfile semantics. Rejected in favor
  of `uv`, which is a strict superset.
