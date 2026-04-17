# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""G1 waypoint dataset for HWM high-level training.

Reads the M4.2 HDF5 expert rollouts and emits macro-segmented
samples: ``n_macros`` consecutive macros (each of ``step_skip``
primitive actions) plus a future goal frame.

See docs/decisions/008-hwm-planning-layer.md (§ Data pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class G1WaypointDataset(Dataset):
    """Macro-chunked expert-rollout dataset.

    HDF5 input schema (produced by scripts/m4_collect_g1_rollouts.py):
        pixels     (N_total, 224, 224, 3) uint8
        action     (N_total, 29)          float32
        ep_offset  (E,)                   int64   # episode starts
        ep_len     (E,)                   int32   # episode lengths
        (proprio / state also present; ignored here since LeWM's
        encoder consumes pixels only.)

    Output per __getitem__:
        pixels        (N+1, 3, 224, 224) float32 in [0, 1]
            — frames at macro boundaries (start, +1 macro, …, +N macros)
        macro_actions (N,   step_skip, 29) float32
            — primitive actions grouped into the N macros
        goal_pixel    (3, 224, 224)       float32 in [0, 1]
            — frame at (N + goal_offset) macros from the sample start
    """

    def __init__(
        self,
        h5_path: str | Path,
        step_skip: int = 50,
        n_macros: int = 4,
        goal_offset_range: tuple[int, int] = (2, 8),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.step_skip = int(step_skip)
        self.n_macros = int(n_macros)
        self.goal_offset_lo, self.goal_offset_hi = goal_offset_range
        self._rng = np.random.default_rng(seed)

        # Open once to read index arrays; close. Per-worker handles
        # are reopened lazily in __getitem__ so this dataset is
        # DataLoader-fork-safe.
        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:]
            ep_len = f["ep_len"][:]

        self.ep_offset = ep_offset.astype(np.int64)
        self.ep_len = ep_len.astype(np.int64)

        # Enumerate valid (episode_idx, macro_start_idx) samples.
        # A sample needs:
        #   - n_macros primitives worth of actions starting at
        #     (ep_offset[e] + mstart*step_skip)
        #   - a goal frame at +(n_macros + goal_offset_hi)*step_skip
        # so the episode must contain at least
        # (n_macros + goal_offset_hi) macros + 1 frame.
        self.samples: list[tuple[int, int]] = []
        required = (self.n_macros + self.goal_offset_hi) * self.step_skip + 1
        for e, L in enumerate(self.ep_len):
            if L < required:
                continue
            # last valid macro start so that we still have goal room
            last = (L - required) // self.step_skip
            for mstart in range(last + 1):
                self.samples.append((int(e), int(mstart)))

        if not self.samples:
            raise RuntimeError(
                f"No valid samples. Check step_skip={step_skip}, "
                f"n_macros={n_macros}, goal_offset_range="
                f"({self.goal_offset_lo}, {self.goal_offset_hi}), "
                f"ep_len stats: min={int(self.ep_len.min())}, "
                f"mean={float(self.ep_len.mean()):.1f}, "
                f"max={int(self.ep_len.max())}"
            )

        self._h5: h5py.File | None = None  # lazy per-worker handle

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, mstart = self.samples[i]
        off = int(self.ep_offset[e])
        S = self.step_skip
        N = self.n_macros
        start = off + mstart * S

        # Macro-boundary frames: (N+1, H, W, 3)
        macro_ts = [start + k * S for k in range(N + 1)]
        h5 = self._open()
        pixels = h5["pixels"][macro_ts]  # (N+1, H, W, 3) uint8

        # Primitive actions in the N macros: (N, S, 29)
        actions_raw = h5["action"][start : start + N * S]  # (N*S, 29)
        actions_chunked = actions_raw.reshape(N, S, 29)

        # Goal frame: sample goal_offset ∈ [lo, hi); frame at
        # +(N + goal_offset) macros from sample start.
        goal_offset = int(
            self._rng.integers(self.goal_offset_lo, self.goal_offset_hi + 1)
        )
        goal_idx = start + (N + goal_offset) * S
        goal_pixel = h5["pixels"][goal_idx]  # (H, W, 3) uint8

        # Convert to float-in-[0,1] + (C, H, W).
        pixels_t = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255.0
        macro_actions_t = torch.from_numpy(actions_chunked).float()
        goal_pixel_t = torch.from_numpy(goal_pixel).permute(2, 0, 1).float() / 255.0

        return {
            "pixels": pixels_t,
            "macro_actions": macro_actions_t,
            "goal_pixel": goal_pixel_t,
        }

    def __getstate__(self) -> dict[str, Any]:
        # h5py.File is not picklable; drop it for DataLoader worker fork.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state
