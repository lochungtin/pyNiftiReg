"""Jacobians of displacement fields and the QC statistics built on them.

Because fields are in voxel units, ``det J`` is the same in voxel and world coordinates (the
spacing matrix cancels: ``det(S J S⁻¹) = det J``), so no spacing argument is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pyniftireg.grid import _check_field


def jacobian_matrix(u: Tensor) -> Tensor:
    """``J = I + ∇u`` by central differences (one-sided at edges), shape ``(B, *spatial, D, D)``.

    ``J[..., i, j] = δᵢⱼ + ∂uᵢ/∂xⱼ``.
    """
    d = _check_field(u)
    grads = torch.gradient(u, dim=tuple(range(2, 2 + d)))
    jac = torch.stack(grads, dim=-1).movedim(1, -2)
    return jac + torch.eye(d, dtype=u.dtype, device=u.device)


def det(u: Tensor) -> Tensor:
    """``det J`` of the map ``id + u``, shape ``(B, *spatial)``.

    Computed from the gradient components directly (no stacked ``(..., D, D)`` matrix), by
    explicit cofactor expansion: this keeps peak memory low on large volumes, is faster than
    ``torch.linalg.det`` on many small matrices, and avoids linalg's lazy initialisation, which
    fails when first hit from several threads.
    """
    d = _check_field(u)
    g = [torch.gradient(u[:, i], dim=tuple(range(1, 1 + d))) for i in range(d)]
    # J[i][j] = δij + ∂u_i/∂x_j
    j = [[g[i][k] + 1.0 if i == k else g[i][k] for k in range(d)] for i in range(d)]
    if d == 2:
        return j[0][0] * j[1][1] - j[0][1] * j[1][0]
    return (
        j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
    )


@dataclass(frozen=True)
class JacobianStats:
    """Per-sample ``det J`` statistics, each a ``(B,)`` tensor."""

    min: Tensor
    max: Tensor
    mean: Tensor
    p01: Tensor
    p99: Tensor
    fold_fraction: Tensor


def _quantile(x: Tensor, q: float) -> Tensor:
    # kthvalue has no size limit, unlike torch.quantile.
    k = min(max(round(q * (x.numel() - 1)) + 1, 1), x.numel())
    return x.kthvalue(k).values


def jacobian_stats(u: Tensor, mask: Tensor | None = None) -> JacobianStats:
    """Per-sample ``det J`` statistics, optionally within ``mask (B or 1, 1, *spatial)``."""
    dj = det(u).flatten(1)
    m = (
        None
        if mask is None
        else mask.to(torch.bool).expand(dj.shape[0], 1, *u.shape[2:]).flatten(1)
    )
    rows = [dj[b] if m is None else dj[b][m[b]] for b in range(dj.shape[0])]
    if any(r.numel() == 0 for r in rows):
        raise ValueError("mask selects no voxels for at least one sample")

    def stack(fn: object) -> Tensor:
        assert callable(fn)
        return torch.stack([fn(r) for r in rows])

    return JacobianStats(
        min=stack(torch.min),
        max=stack(torch.max),
        mean=stack(torch.mean),
        p01=stack(lambda r: _quantile(r, 0.01)),
        p99=stack(lambda r: _quantile(r, 0.99)),
        fold_fraction=stack(lambda r: (r <= 0).to(r.dtype).mean()),
    )
