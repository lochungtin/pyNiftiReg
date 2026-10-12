"""N-D separable filtering helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_kernel1d(
    sigma: float,
    *,
    truncate: float = 3.0,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    radius = max(math.ceil(truncate * sigma), 1)
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_smooth(x: Tensor, sigma: float | Sequence[float], *, truncate: float = 3.0) -> Tensor:
    """Smooth ``x`` of shape ``(B, C, *spatial)`` with a Gaussian of ``sigma`` voxels per axis.

    Edges use replicate padding, so constant fields are preserved exactly.
    """
    d = x.ndim - 2
    sigmas = [float(s) for s in (sigma if isinstance(sigma, Sequence) else [sigma] * d)]
    if len(sigmas) != d:
        raise ValueError(f"expected {d} sigmas, got {len(sigmas)}")
    for axis, s in enumerate(sigmas):
        if s <= 0:
            continue
        k = gaussian_kernel1d(s, truncate=truncate, dtype=x.dtype, device=x.device)
        r = (k.numel() - 1) // 2
        moved = x.movedim(2 + axis, -1)
        shape = moved.shape
        flat = moved.reshape(-1, 1, shape[-1])
        flat = F.conv1d(F.pad(flat, (r, r), mode="replicate"), k.view(1, 1, -1))
        x = flat.reshape(shape).movedim(-1, 2 + axis)
    return x
