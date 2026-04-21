# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
"""M4.6 — Frozen DINOv3 ViT-B/16 backbone that returns 7×7 subsampled
PATCH tokens (not the CLS token).

The v4 CLS-only pipeline produced a pose-invariant embedding: any
mid-walk G1 frame landed in the same cos region, so MPPI had no
gradient to steer. DINO-WM, V-JEPA 2-AC, and Terver et al. all use
patch features — patches preserve the spatial layout of the robot in
the frame, which is the discriminator we need.

This module extracts the 196 patch tokens (14×14 grid) from DINOv3
and subsamples to 7×7 = 49 patches to fit in our 8 GB VRAM / 20 GB
cache budget. Same frozen DINOv3 weights as v4 — we just keep more of
the output.

API mirrors ``backbone_dinov3.DINOv3Backbone`` so downstream code (the
JEPAv5 module, the planner's history plumbing) can swap between CLS
and patch backbones with minimal surgery.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

# ImageNet normalization — HF docs; same as used for DINOv2/v3 downstream.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DINOv3PatchBackbone(nn.Module):
    """Wraps a frozen DINOv3 ViT-B/16 and returns the 7×7 subsampled
    patch-token grid per frame.

    Input:  pixels (B, T, 3, H, W) uint8 or float (any dtype).
    Output: emb    (B, T, N_patches, D_out)   float32
            where N_patches = ((H/16)//2)**2  (49 for H=224)
            and D_out = encoder hidden_size   (768 for ViT-B/16).

    The encoder runs in FP16 for VRAM. The returned tensor is cast to
    float32 so downstream modules (trainable projection, predictor) can
    hold autograd state without FP16 precision loss.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_ID,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        freeze: bool = True,
        spatial_stride: int = 2,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            lazy: If True, skip loading the DINOv3 weights into VRAM at
                construction. Use when training reads a precomputed
                patch cache — the 344 MB of weights would otherwise
                waste VRAM that the trainable predictor needs. The
                geometry fields (output_dim, grid_side, n_patches) are
                set from known ViT-B/16 defaults so downstream shape
                checks still work. Call ``load_eagerly()`` to hydrate
                the weights later (needed for plan-time encoding).
        """
        super().__init__()

        self.model_id = model_id
        self.dtype = dtype
        self.device_str = device
        self.frozen = freeze
        self.spatial_stride = int(spatial_stride)
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.dinov3: nn.Module | None = None

        if lazy:
            # Assume ViT-B/16 defaults at 224×224: 14×14 patch grid,
            # 768-D hidden, 4 register tokens. Override with
            # ``load_eagerly()`` or inspection at plan time.
            self.output_dim = 768
            self._n_skip = 5  # CLS + 4 registers
            self.grid_side_full = 14
            self.grid_side = 14 // self.spatial_stride
            self.n_patches = self.grid_side * self.grid_side
            return

        self.load_eagerly()

    def load_eagerly(self) -> None:
        """Load the DINOv3 weights into VRAM. Idempotent."""
        if self.dinov3 is not None:
            return
        from transformers import AutoModel

        self.dinov3 = AutoModel.from_pretrained(
            self.model_id, dtype=self.dtype
        ).to(self.device_str)
        if self.frozen:
            for p in self.dinov3.parameters():
                p.requires_grad_(False)
            self.dinov3.eval()

        hidden = int(self.dinov3.config.hidden_size)
        self.output_dim = hidden
        n_reg = int(getattr(self.dinov3.config, "num_register_tokens", 4))
        self._n_skip = 1 + n_reg

        with torch.no_grad():
            dummy = torch.zeros(
                1, 3, 224, 224, device=self.device_str, dtype=self.dtype
            )
            out = self.dinov3(pixel_values=dummy)
            seq_len = int(out.last_hidden_state.shape[1])
            n_patches_full = seq_len - self._n_skip
            side = int(n_patches_full**0.5)
            assert side * side == n_patches_full, (
                f"DINOv3 output {n_patches_full} patches is not a square grid"
            )
        self.grid_side_full = side
        self.grid_side = side // self.spatial_stride
        self.n_patches = self.grid_side * self.grid_side

    def _prep(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 255.0
        elif pixels.dtype not in (torch.float32, torch.float16):
            pixels = pixels.float()
        pixels = self._normalize(pixels)
        return pixels.to(self.dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames → 7×7 patch tokens (flattened to 49).

        Args:
            pixels: (B, T, 3, H, W) where H=W=224 for standard DINOv3.
        Returns:
            emb: (B, T, N_patches, D_out) float32.
        """
        if self.dinov3 is None:
            self.load_eagerly()
        assert pixels.dim() == 5, f"expected (B, T, 3, H, W), got {tuple(pixels.shape)}"
        B, T = pixels.shape[0], pixels.shape[1]
        pixels_flat = pixels.reshape(B * T, *pixels.shape[2:])
        pixels_flat = self._prep(pixels_flat)
        if self.frozen:
            with torch.no_grad():
                out = self.dinov3(pixel_values=pixels_flat)
        else:
            out = self.dinov3(pixel_values=pixels_flat)
        # last_hidden_state: (B*T, 1 + n_reg + G*G, D_out)
        # Drop CLS + register tokens.
        patches = out.last_hidden_state[:, self._n_skip :, :]  # (B*T, G*G, D)
        # Reshape to (B*T, G, G, D) and subsample spatially.
        G = self.grid_side_full
        patches = patches.reshape(B * T, G, G, -1)
        if self.spatial_stride > 1:
            patches = patches[:, :: self.spatial_stride, :: self.spatial_stride, :]
        patches = patches.reshape(B * T, self.n_patches, -1)  # (B*T, N, D)
        patches = patches.float().reshape(B, T, self.n_patches, -1)
        return patches

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)
