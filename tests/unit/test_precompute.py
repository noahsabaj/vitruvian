# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""precompute._stream_encode — streaming + ragged-final-batch + dtype cast."""

from __future__ import annotations

from pathlib import Path

import torch

from vitruvian.data.precompute import _stream_encode


def test_stream_encode_ragged_and_dtype(synthetic_h5: Path) -> None:
    """Streams the HDF5 pixel array in ragged batches (50 frames, batch 7 → a
    final short batch of 1), writes each per-frame embedding into the right row,
    and down-casts to the requested out_dtype."""
    seen_rows = {"n": 0}

    def fake_encode(batch: torch.Tensor) -> torch.Tensor:
        # (B, 1, 3, H, W) -> (B, 1, D); tag each row with its batch size so we
        # can confirm every frame was visited exactly once.
        b = batch.shape[0]
        seen_rows["n"] += b
        return torch.ones(b, 1, 8)

    out = _stream_encode(
        synthetic_h5, fake_encode,
        out_shape=(50, 8), out_dtype=torch.float16, batch_size=7, device="cpu",
    )
    assert out.shape == (50, 8)
    assert out.dtype == torch.float16
    assert (out == 1).all()
    assert seen_rows["n"] == 50  # every frame encoded once, ragged tail included


def test_stream_encode_rejects_row_mismatch(synthetic_h5: Path) -> None:
    """A wrong out_shape row count is caught up front, not silently truncated."""
    import pytest

    with pytest.raises(RuntimeError, match="rows"):
        _stream_encode(
            synthetic_h5, lambda b: torch.ones(b.shape[0], 1, 8),
            out_shape=(49, 8), out_dtype=torch.float32, batch_size=8, device="cpu",
        )
