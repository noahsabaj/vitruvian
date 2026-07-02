# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Tests for ``vitruvian.data.collection`` orchestrator failure policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from vitruvian.data import CollectionConfig, CommandSpec, run_collection
from vitruvian.data.collection import merge_hdf5_chunks_streaming


def _fake_chunk_writer(out: Path) -> None:
    """Write a tiny valid HDF5 chunk matching the collector's schema."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 5
    with h5py.File(out, "w") as f:
        f.create_dataset(
            "pixels", data=np.zeros((n, 16, 16, 3), dtype=np.uint8)
        )
        f.create_dataset("action", data=np.zeros((n, 29), dtype=np.float32))
        f.create_dataset("proprio", data=np.zeros((n, 103), dtype=np.float32))
        f.create_dataset("state", data=np.zeros((n, 40), dtype=np.float32))
        f.create_dataset("ep_len", data=np.array([n], dtype=np.int32))
        f.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        f.create_dataset(
            "commands", data=np.zeros((1, 3), dtype=np.float32)
        )


def _config_with_3_chunks(tmp_path: Path) -> CollectionConfig:
    """Build a config that produces exactly 3 chunks (3 commands, 1 ep each)."""
    return CollectionConfig(
        out_h5=tmp_path / "out.h5",
        episodes_per_command=1,
        episode_steps=5,
        commands=(
            CommandSpec(0.0, 0.0, 0.0),
            CommandSpec(0.5, 0.0, 0.0),
            CommandSpec(-0.5, 0.0, 0.0),
        ),
        chunk_size=1,
        seed=0,
        img_size=16,
        keep_chunks=False,
    )


def test_run_collection_raises_on_partial_failure(tmp_path: Path) -> None:
    """With the default ``allow_partial=False``, a single failing chunk
    causes the whole run to raise after the surviving chunks merge."""
    cfg = _config_with_3_chunks(tmp_path)
    call_count = {"n": 0}

    def fake_spawn(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:  # second chunk fails
            return 1
        _fake_chunk_writer(kwargs["out"])
        return 0

    with patch("vitruvian.data.collection._spawn_worker", side_effect=fake_spawn):
        with pytest.raises(RuntimeError, match="chunks failed"):
            run_collection(cfg, single_process=False)

    # Merged output exists — the surviving chunks were merged.
    assert cfg.out_h5.exists()


def test_run_collection_accepts_partial_with_flag(tmp_path: Path) -> None:
    """With ``allow_partial=True``, the same partial-failure scenario
    completes without raising."""
    cfg = _config_with_3_chunks(tmp_path)
    call_count = {"n": 0}

    def fake_spawn(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return 1
        _fake_chunk_writer(kwargs["out"])
        return 0

    with patch("vitruvian.data.collection._spawn_worker", side_effect=fake_spawn):
        run_collection(cfg, single_process=False, allow_partial=True)

    assert cfg.out_h5.exists()


def test_run_collection_all_fail_raises_regardless(tmp_path: Path) -> None:
    """If every chunk fails, there's nothing to merge — raise even with
    ``allow_partial=True`` (message is different but the outcome is the
    same: no dataset, error)."""
    cfg = _config_with_3_chunks(tmp_path)

    def fake_spawn(**kwargs):
        return 1

    with patch("vitruvian.data.collection._spawn_worker", side_effect=fake_spawn):
        with pytest.raises(RuntimeError, match="All chunks failed"):
            run_collection(cfg, single_process=False, allow_partial=True)


def _identifiable_chunk(path: Path, base: int, cmd: list[float], n: int = 3) -> None:
    """A chunk whose every row equals ``base`` — so a scrambled merge is visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=np.full((n, 4, 4, 3), base % 256, dtype=np.uint8))
        f.create_dataset("action", data=np.full((n, 29), base, dtype=np.float32))
        f.create_dataset("proprio", data=np.full((n, 103), base, dtype=np.float32))
        f.create_dataset("state", data=np.full((n, 40), base, dtype=np.float32))
        f.create_dataset("ep_len", data=np.array([n], dtype=np.int32))
        f.create_dataset("ep_offset", data=np.array([0], dtype=np.int64))
        f.create_dataset("commands", data=np.asarray([cmd], dtype=np.float32))


def test_merge_preserves_row_alignment(tmp_path: Path) -> None:
    """The streaming merge must keep pixels/action/proprio/state rows aligned
    with ep_offset/ep_len/commands across chunks — a scrambled merge silently
    corrupts the dataset. Also checks the distinct-command mode labelling."""
    c0 = tmp_path / "chunk_0000.h5"
    c1 = tmp_path / "chunk_0001.h5"
    _identifiable_chunk(c0, 10, [0.5, 0.0, 0.0])
    _identifiable_chunk(c1, 20, [-0.5, 0.0, 0.0])
    out = tmp_path / "merged.h5"
    merge_hdf5_chunks_streaming([c0, c1], out)

    with h5py.File(out, "r") as f:
        assert f["action"].shape[0] == 6
        assert (f["action"][:3] == 10).all() and (f["action"][3:] == 20).all()
        assert (f["proprio"][:3] == 10).all() and (f["proprio"][3:] == 20).all()
        assert (f["state"][:3] == 10).all() and (f["state"][3:] == 20).all()
        assert list(f["ep_offset"][:]) == [0, 3]
        assert list(f["ep_len"][:]) == [3, 3]
        # commands stay aligned to their episodes
        assert f["commands"][0, 0] == 0.5 and f["commands"][1, 0] == -0.5
        assert int(f.attrs["n_distinct_commands"]) == 2
        assert f.attrs["collection_mode"] == "diverse-command-grid"


def test_merge_single_command_labelled_narrow(tmp_path: Path) -> None:
    """Many chunks/episodes of ONE command is 'narrow', not 'diverse' — the
    label must key on distinct commands, not chunk count."""
    files = []
    for i in range(3):
        p = tmp_path / f"chunk_{i:04d}.h5"
        _identifiable_chunk(p, i + 1, [0.5, 0.0, 0.0])  # same command each chunk
        files.append(p)
    out = tmp_path / "merged.h5"
    merge_hdf5_chunks_streaming(files, out)
    with h5py.File(out, "r") as f:
        assert int(f.attrs["n_distinct_commands"]) == 1
        assert f.attrs["collection_mode"] == "narrow"
