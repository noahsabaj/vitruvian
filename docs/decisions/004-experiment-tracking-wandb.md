# ADR 004 — Experiment Tracking: Weights & Biases

**Status:** Accepted
**Date:** 2026-04-16

## Context

From the first training run, every experiment must be reproducible,
diffable, and browsable. Three options were considered: wandb,
TensorBoard, and HuggingFace Trackio.

## Decision

Use **Weights & Biases (wandb)** as the primary experiment tracking
platform. All training runs log to wandb from day 1.

## Consequences

- Every reference implementation we will touch (`mujoco_playground`,
  `dreamerv3`, `tdmpc2`) already logs to wandb, so wiring is minimal.
- Cloud-hosted dashboards, shareable links, team-ready if the project
  opens up.
- One-line integration; offline mode available when needed.
- Free-tier dependency on an external service accepted. If that becomes
  a problem we can fall back to TensorBoard without code changes beyond
  the logger config.

## Alternatives considered

- **TensorBoard** — local, no account, simpler. Rejected: diff and
  comparison UX is weaker and wandb is the community default.
- **HuggingFace Trackio** — promising, HF-integrated, newer. Rejected
  for now; revisit if/when the project publishes to the HF Hub.
