# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""EncoderHistory unit tests."""

from __future__ import annotations

import torch

from vitruvian.planning import EncoderHistory


class CountingEncoder:
    """Tracks call count so we can assert encode-once semantics."""

    output_dim = 16

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        # pixels: (B, T, 3, H, W) — reduce to (B, T, D)
        B, T = pixels.size(0), pixels.size(1)
        return torch.full(
            (B, T, self.output_dim),
            fill_value=float(self.calls),
        )


def test_encode_once_per_push() -> None:
    enc = CountingEncoder()
    hist = EncoderHistory(size=3, encoder=enc)

    for i in range(5):
        frame = torch.zeros(3, 224, 224)
        hist.push(frame)
    assert enc.calls == 5, enc.calls


def test_latest_window_pads_when_short() -> None:
    enc = CountingEncoder()
    hist = EncoderHistory(size=3, encoder=enc)

    hist.push(torch.zeros(3, 224, 224))
    window = hist.latest_window()
    # (size, D). First two entries are padded copies of the only real frame.
    assert window.shape == (3, 16), window.shape
    assert torch.equal(window[0], window[1])
    assert torch.equal(window[1], window[2])


def test_latest_window_ring() -> None:
    enc = CountingEncoder()
    hist = EncoderHistory(size=3, encoder=enc)
    for _ in range(5):
        hist.push(torch.zeros(3, 224, 224))
    win = hist.latest_window()
    # Latest three encodings had call counts 3, 4, 5.
    assert win[0, 0].item() == 3.0
    assert win[1, 0].item() == 4.0
    assert win[2, 0].item() == 5.0


def test_reset_clears_history() -> None:
    enc = CountingEncoder()
    hist = EncoderHistory(size=3, encoder=enc)
    hist.push(torch.zeros(3, 224, 224))
    hist.reset()
    # After reset, the first post-reset push should be the only one in the buffer.
    hist.push(torch.zeros(3, 224, 224))
    win = hist.latest_window()
    # All three window entries are padded copies of the single push after reset.
    assert torch.equal(win[0], win[1])
    assert torch.equal(win[1], win[2])
