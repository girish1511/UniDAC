# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

class LatFreqWeight(nn.Module):
    """
    Maps latitude phi (radians) at patch resolution [H', W'] to per-frequency
    weights in [w_min, 1], shape [H'*W', F], where F = D_head//4.
    """
    def __init__(self, D: int, hidden: int = 64, w_min: float = 0.5):
        super().__init__()
        self.D = D
        self.w_min = w_min

        in_dim = 3  # [phi, sin(phi), cos(phi)]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, D),
        )
        # init last layer bias to push weights near 1.0
        nn.init.constant_(self.mlp[-1].bias, 4.0)  # sigmoid(4) ~ 0.982 -> ~ near 1
        # small weights init
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)

    def forward(self, phi_patch: torch.Tensor) -> torch.Tensor:
        """
        phi_patch: [B, H'W', 1] in radians (e.g., [-pi/2, +pi/2])
        returns: wx [H'*W', F] in [w_min, 1]
        """
        assert phi_patch.ndims == 3
        assert phi_patch.shape[-1] == 1
        # phi = phi_patch.reshape(B, -1, 1)  # [B, HW, 1]
        feats = torch.cat([phi_patch, torch.sin(phi_patch), torch.cos(phi_patch)], dim=-1)  # [B, HW, 3]
        logits = self.mlp(feats)  # [B, HW, D]
        s = torch.sigmoid(logits)  # [0,1]
        w = self.w_min + (1.0 - self.w_min) * s  # [w_min, 1]
        return w  # [B, HW, D]


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )

        # self.learn_lat_freq_w = LatFreqWeight(w_min=0.5)

        self._init_weights()

    def forward(self, *, H: int, W: int, lat_patch: Tensor=None, long_patch: Tensor=None) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # # Shift coords by adding a uniform value in [-shift, shift]
        # if self.training and self.shift_coords is not None:
        #     shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
        #     coords += shift_hw[None, :]

        # # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        # if self.training and self.jitter_coords is not None:
        #     jitter_max = np.log(self.jitter_coords)
        #     jitter_min = -jitter_max
        #     jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
        #     coords *= jitter_hw[None, :]

        # # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        # if self.training and self.rescale_coords is not None:
        #     rescale_max = np.log(self.rescale_coords)
        #     rescale_min = -rescale_max
        #     rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
        #     coords *= rescale_hw

        # Prepare angles and sin/cos

        # if long_patch is not None:
        #     lat_coords = lat_patch.squeeze(1).flatten(-2,-1) / (np.pi/2)
        #     long_coords = long_patch.squeeze(1).flatten(-2,-1) / (np.pi)
        #     sph_coords = torch.stack([lat_coords, long_coords], dim=-1)
        #     angles = 2 * math.pi * sph_coords[..., None] / self.periods[None, None, None, :] # [B, HW, 2, D//4]
        #     angles = angles.flatten(-2, -1)
        #     angles = angles.tile(2) # [B, HW, D]

        #     shift = 0.8
        #     lat_patch_flat = lat_patch.squeeze(1).flatten(-2,-1)
        #     cos_lat = torch.cos(lat_patch_flat)
        #     w_lat = shift + (1-shift)*cos_lat # [B, HW]

        #     angles = (angles * w_lat[...,None]).unsqueeze(1) # [B, 1, HW, D]

        # else:
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]

        if lat_patch is not None:
            shift = 0.5
            lat_patch_flat = lat_patch.squeeze(1).flatten(-2,-1)
            # w_lat = self.learn_lat_freq_w(lat_patch_flat)
            # angles = (angles[None,...] * w_lat).unsqueeze(1)

            cos_lat = torch.abs(torch.cos(lat_patch_flat))
            w_lat = shift + (1-shift)*cos_lat
            angles = (angles[None,...] * w_lat[..., None]).unsqueeze(1)

        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods