# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.7 — ``EncoderHistory``: ring buffer of encoded-frame latents for
plan-time MPPI history.

**Why this module exists.** Before M4.7, ``m4c_hierarchical_plan.py``
maintained a plain ``pixel_hist: list[torch.Tensor]`` and a matching
``action_hist: list[np.ndarray]``, and at each macro boundary it did:

    pix = render_head_cam(env_ctx)
    pix_t = torch.from_numpy(pix)...
    pixel_hist.append(pix_t[0, 0])
    if len(pixel_hist) > HIST_SIZE: pixel_hist = pixel_hist[-HIST_SIZE:]
    ...
    ph = torch.stack(pixel_hist, dim=0)     # → planner input
    curr_emb = backbone.encode(pix_t).squeeze(0).squeeze(0)

and separately, inside ``LowLevelPlanner.plan``, the entire
``pixel_history`` tensor was re-encoded via ``self.jepa.rollout`` each
MPPI iteration — which internally re-runs ``self.encode`` on every
history frame even though only the *newest* frame is new.

With 10 macros + 3-frame history, that's 30 encoder forwards per
walking run — 20 of them redundant. Under v5 DINOv3 ViT-B at ~10 ms
per forward in inference, that's ~200 ms of pure waste per run, saved
across the whole eval matrix. More importantly the ad-hoc list
management is error-prone: M4.5 had a bug where the action history was
offset by one wrt. the pixel history (fixed during M4.6 debug).

This module factors the pattern into a small ring-buffer class with
an explicit encoder-per-push invariant.

**Invariants:**

  * Each ``push(pixels)`` triggers *exactly one* encoder forward
    on the newly supplied frame — never on already-encoded ones.
  * ``latest_window()`` returns the last ``size`` frames as a single
    tensor of shape ``(size, *emb_shape)``; if fewer than ``size``
    frames have been pushed, the oldest is repeated (same padding
    semantics as the pre-refactor ad-hoc code used).
  * ``reset()`` drops all history without touching the encoder.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch


class EncoderHistory:
    """Append-only ring buffer of encoded frames.

    Parameters
    ----------
    size
        Max number of frames retained. Older frames are evicted FIFO.
    encoder
        Callable that maps one pixel tensor ``(3, H, W)`` float in
        ``[0, 1]`` to one encoded tensor (shape up to the encoder —
        ``(D,)`` for CLS, ``(N, D)`` for patches). Must accept either a
        single-frame tensor or a leading batched shape; we always pass
        ``(1, 1, 3, H, W)`` and squeeze the batch/time dims out.

    Notes
    -----
    The encoder is expected to be wrapped appropriately (frozen, in
    eval mode, possibly compiled). This class does not take a ``device``
    — the encoder decides placement.
    """

    def __init__(
        self,
        size: int,
        encoder: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.size = int(size)
        self._encoder = encoder
        self._buf: deque[torch.Tensor] = deque(maxlen=self.size)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def push(self, pixels_chw: torch.Tensor) -> torch.Tensor:
        """Encode and retain a single new frame.

        Args:
            pixels_chw: ``(3, H, W)`` float in [0, 1] (or uint8 which
                the encoder is responsible for normalizing).

        Returns:
            The encoded frame (also stored in the buffer).
        """
        if pixels_chw.dim() != 3:
            raise ValueError(
                f"expected (3, H, W), got {tuple(pixels_chw.shape)}"
            )
        # Encoder expects (B, T, 3, H, W); we pass (1, 1, ...) and
        # squeeze twice. This matches the DINOv3 backbones' / PlannerBackbone
        # encode() signature.
        batched = pixels_chw.unsqueeze(0).unsqueeze(0)
        emb = self._encoder(batched).squeeze(0).squeeze(0)
        self._buf.append(emb)
        return emb

    def reset(self) -> None:
        """Drop all history."""
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    # ------------------------------------------------------------------
    # Read views
    # ------------------------------------------------------------------

    def latest(self) -> torch.Tensor:
        """Most recent encoded frame. Raises if nothing pushed yet."""
        if not self._buf:
            raise RuntimeError("EncoderHistory is empty — push() before latest()")
        return self._buf[-1]

    def latest_window(self) -> torch.Tensor:
        """Last ``size`` encoded frames stacked into a single tensor.

        Left-pads by repeating the earliest frame so the returned
        tensor is always ``(self.size, *emb.shape)`` — matches the
        pre-refactor padding behavior in ``LowLevelPlanner.plan``.
        """
        if not self._buf:
            raise RuntimeError("EncoderHistory is empty — push() before latest_window()")
        frames = list(self._buf)
        if len(frames) < self.size:
            pad_n = self.size - len(frames)
            frames = [frames[0]] * pad_n + frames
        return torch.stack(frames, dim=0)


__all__ = ["EncoderHistory"]
