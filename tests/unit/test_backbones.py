# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Backbone Protocol conformance + fake-backbone output shapes."""

from __future__ import annotations

import torch

from vitruvian.models import Backbone


def test_fake_cls_conforms_to_protocol(fake_cls_backbone) -> None:
    assert isinstance(fake_cls_backbone, Backbone)
    pixels = torch.zeros(2, 3, 3, 224, 224)
    emb = fake_cls_backbone.encode(pixels)
    assert emb.shape == (2, 3, fake_cls_backbone.output_dim)


def test_fake_patch_conforms_to_protocol(fake_patch_backbone) -> None:
    assert isinstance(fake_patch_backbone, Backbone)
    pixels = torch.zeros(2, 3, 3, 224, 224)
    emb = fake_patch_backbone.encode(pixels)
    assert emb.shape == (
        2, 3, fake_patch_backbone.n_patches, fake_patch_backbone.output_dim
    )


def test_backbones_are_deterministic(fake_patch_backbone) -> None:
    px = torch.randn(1, 1, 3, 224, 224)
    a = fake_patch_backbone.encode(px)
    b = fake_patch_backbone.encode(px)
    assert torch.allclose(a, b)
