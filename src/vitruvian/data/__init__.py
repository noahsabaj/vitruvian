# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Data layer — datasets, precomputed-embedding cache, collectors."""

from vitruvian.data.cache import CacheKey, EmbeddingCache
from vitruvian.data.collection import (
    CollectionConfig,
    CommandSpec,
    DEFAULT_DIVERSE_GRID,
    NARROW_COMMAND,
    collect_chunk,
    diverse_config,
    merge_hdf5_chunks_streaming,
    narrow_config,
    run_collection,
)
from vitruvian.data.datasets import (
    G1EmbSeqDataset,
    G1HERTransitionDataset,
    G1PatchSeqDataset,
)
from vitruvian.data.precompute import build_cls_cache, build_patch_cache

__all__ = [
    "CacheKey",
    "CollectionConfig",
    "CommandSpec",
    "DEFAULT_DIVERSE_GRID",
    "EmbeddingCache",
    "G1EmbSeqDataset",
    "G1HERTransitionDataset",
    "G1PatchSeqDataset",
    "NARROW_COMMAND",
    "build_cls_cache",
    "build_patch_cache",
    "collect_chunk",
    "diverse_config",
    "merge_hdf5_chunks_streaming",
    "narrow_config",
    "run_collection",
]
