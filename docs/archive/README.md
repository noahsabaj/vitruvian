# docs/archive

Historical artifacts preserved for reference. These are not consumed by any
build or test.

## `lewm_local_edits.patch`

The uncommitted local edits Vitruvian carried against `external/le-wm/` before
M4.9 retired the submodule. The patch covers:

* `jepa.py` — adds the optional `proprio_encoder` kwarg + the rollout-signature
  changes Vitruvian depended on (Terver et al. 2512.24497 feature-conditioning
  recipe). **This patch is already applied in `src/vitruvian/lewm_compat/jepa.py`.**
* `train.py` — LeWM's own Lightning training driver with G1-specific tweaks.
  **Not applied anywhere in Vitruvian** — we never ran LeWM's trainer; we built
  our own on top of the vendored classes (`src/vitruvian/training/trainer.py`).

Apply via `git apply docs/archive/lewm_local_edits.patch` inside a fresh LeWM
checkout if you ever need to revive the submodule workflow. The canonical
vendored copy is `src/vitruvian/lewm_compat/` and should be the source of truth
going forward.
