"""Synthetic fields shared by the tests."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from pyniftireg.filters import gaussian_smooth

DT = torch.float64


def shape_for(ndim: int) -> tuple[int, ...]:
    return (48, 48) if ndim == 2 else (32, 32, 32)


def smooth_field(
    ndim: int, *, max_norm: float = 3.0, sigma: float = 4.0, seed: int = 0, batch: int = 1
) -> Tensor:
    """Random smooth SVF ``(batch, D, *shape)`` with ``max‖v‖ == max_norm`` voxels."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(batch, ndim, *shape_for(ndim), generator=g, dtype=DT)
    v = gaussian_smooth(v, sigma)
    return v * (max_norm / v.norm(dim=1).max())


def interior(x: Tensor, margin: int) -> Tensor:
    """Crop ``margin`` voxels from every spatial edge of ``(B, C, *spatial)``."""
    idx = (slice(None), slice(None)) + (slice(margin, -margin),) * (x.ndim - 2)
    return x[idx]


def rotation_svf(shape: tuple[int, ...], angle: float) -> Tensor:
    """SVF whose exponential rotates axes (0, 1) by ``angle`` radians about the grid centre."""
    from pyniftireg.grid import identity

    x = identity(shape, dtype=DT)
    c = torch.tensor([(n - 1) / 2 for n in shape], dtype=DT).view(1, -1, *[1] * len(shape))
    r = x - c
    v = torch.zeros_like(x)
    v[:, 0] = -angle * r[:, 1]
    v[:, 1] = angle * r[:, 0]
    return v


def shapes_pair(ndim: int, n: int, *, square: float, circle: float) -> tuple[Tensor, Tensor]:
    """Centred square/cube (half-width ``square``) and circle/sphere (radius ``circle``) masks."""
    from pyniftireg.grid import identity

    x = identity((n,) * ndim, dtype=DT) - (n - 1) / 2
    sq = (x.abs().amax(1, keepdim=True) <= square).to(DT)
    ci = (x.norm(dim=1, keepdim=True) <= circle).to(DT)
    return sq, ci


def oblique_affine(spacing: tuple[float, ...], origin: tuple[float, ...]) -> np.ndarray:
    """3D affine with a ~20° rotation, anisotropic spacing and a flipped axis."""
    a, b = np.deg2rad(20.0), np.deg2rad(-10.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    m = np.eye(4)
    m[:3, :3] = rz @ rx @ np.diag(spacing)
    m[:3, 3] = origin
    return m
