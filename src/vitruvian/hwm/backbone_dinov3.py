# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""Frozen DINOv3 ViT-B/16 backbone for M4.5 world-model rebuild.

Replaces LeWM's trained-from-scratch ViT-tiny with Meta's pretrained
``facebook/dinov3-vitb16-pretrain-lvd1689m``. All 86M encoder parameters
are frozen at construction; only downstream modules (proprio MLP,
predictor, action encoder) train.

Preserves the same minimal flat-CLS API as
``backbone_adapter.LeWMBackboneAdapter`` so ``LowLevelPlanner`` and
``m4c_hierarchical_plan.py`` work unchanged:

    encoder = DINOv3Backbone(...)
    emb = encoder.encode(pixels)   # (B, T, 768)
    encoder.output_dim             # 768

Rationale for the swap is in ``docs/decisions/008-hwm-planning-layer.md``
(M4.5 amendment) and in the plan at
``~/.claude/plans/yes-we-are-in-delegated-russell.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

# Per HF model card + DINOv2/v3 convention.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DINOv3Backbone(nn.Module):
    """Thin wrapper around HuggingFace DINOv3 ViT-B/16.

    Input:  pixels (B, T, 3, H, W) uint8 or float32 in [0, 1].
    Output: emb    (B, T, D_out)   float32 (always cast back from FP16).

    The internal DINOv3 runs in FP16 for VRAM headroom (~500 MB saved vs
    FP32); the CLS output is cast to float32 before returning so it can
    participate in autograd for the downstream trainable modules.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_ID,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel  # lazy import

        self.model_id = model_id
        self.dtype = dtype
        self.device_str = device
        self.frozen = freeze

        self.dinov3 = AutoModel.from_pretrained(model_id, dtype=dtype).to(device)
        if freeze:
            for p in self.dinov3.parameters():
                p.requires_grad_(False)
            self.dinov3.eval()

        hidden = int(self.dinov3.config.hidden_size)
        self.output_dim = hidden

        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Quick sanity check the output has the expected shape.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
            out = self.dinov3(pixel_values=dummy)
            assert out.last_hidden_state.shape[-1] == hidden, (
                f"DINOv3 last_hidden_state last dim {out.last_hidden_state.shape[-1]} "
                f"!= config.hidden_size {hidden}"
            )

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, 3, H, W) pixels to ImageNet-normalized floats in
        the backbone's dtype."""
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.dtype != torch.float32 and pixels.dtype != torch.float16:
            pixels = pixels.float()
        pixels = self._normalize(pixels)
        return pixels.to(self.dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames → flat CLS embeddings.

        Args:
            pixels: (B, T, 3, H, W) — T may be 1.
        Returns:
            emb: (B, T, D_out) float32.
        """
        assert pixels.dim() == 5, (
            f"expected (B, T, 3, H, W), got {tuple(pixels.shape)}"
        )
        B, T = pixels.shape[0], pixels.shape[1]
        pixels_flat = pixels.reshape(B * T, *pixels.shape[2:])
        pixels_flat = self._prep(pixels_flat)
        if self.frozen:
            with torch.no_grad():
                out = self.dinov3(pixel_values=pixels_flat)
        else:
            out = self.dinov3(pixel_values=pixels_flat)
        # CLS token is index 0; register tokens 1..4 and patches 5..N are
        # discarded for the flat-CLS-only pipeline (per Terver et al.).
        cls = out.last_hidden_state[:, 0, :]  # (B*T, D)
        cls = cls.float().reshape(B, T, -1)
        return cls

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)
