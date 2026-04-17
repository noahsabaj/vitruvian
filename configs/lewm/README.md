# `configs/lewm/`

Canonical copies of Hydra config fragments for LeWM that we maintain
outside the `external/le-wm` submodule. The submodule itself is
pinned to upstream main; we don't push into it, so config edits need
to live here and be synced in.

**Sync them in before running LeWM commands that reference them:**

```bash
# from repo root
cp configs/lewm/g1.yaml external/le-wm/config/train/data/g1.yaml
```

(or add a symlink once, since `g1.yaml` is gitignored inside the
submodule's git boundary anyway.)

Currently tracked:

- `g1.yaml` — data config for `python train.py data=g1`. References
  `~/.stable_worldmodel/g1_joystick_expert.h5`, produced by
  `scripts/m4_collect_g1_rollouts.py`.
