"""N-D uniform cubic B-spline control-point grids <-> dense fields.

Control-point layout (NiftyReg-style): with spacing ``s`` voxels along an axis, control point ``k``
sits at voxel position ``(k - 1)·s``, so a grid covering ``n`` voxels needs ``floor((n-1)/s) + 4``
points (one before the image, two after). The match to NiftyReg's actual files is pinned by the
M2 round-trip tests.

Both directions are separable tensor-product operations built from per-axis weight matrices, so
they are differentiable and work for any number of dimensions.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
from torch import Tensor


def num_control_points(n: int, spacing: float) -> int:
    """Control points needed along an axis of ``n`` voxels with ``spacing`` voxels between them."""
    return math.floor((n - 1) / spacing) + 4


def basis(u: Tensor) -> Tensor:
    """Cubic B-spline basis weights for offsets ``u ∈ [0, 1)``, shape ``(*u.shape, 4)``."""
    u2, u3 = u * u, u * u * u
    return torch.stack(
        [
            (1 - u) ** 3 / 6,
            (3 * u3 - 6 * u2 + 4) / 6,
            (-3 * u3 + 3 * u2 + 3 * u + 1) / 6,
            u3 / 6,
        ],
        dim=-1,
    )


def weight_matrix(
    n: int,
    spacing: float,
    n_cp: int | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """``(n, n_cp)`` matrix mapping control-point values to voxel values along one axis."""
    n_cp = num_control_points(n, spacing) if n_cp is None else n_cp
    x = torch.arange(n, dtype=torch.float64) / spacing
    i = torch.floor(x).long()
    b = basis(x - i)
    w = torch.zeros(n, n_cp, dtype=torch.float64)
    cols = i[:, None] + torch.arange(4)
    if int(cols.max()) >= n_cp:
        raise ValueError(f"{n_cp} control points cannot cover {n} voxels at spacing {spacing}")
    w.scatter_(1, cols, b)
    return w.to(dtype=dtype or torch.get_default_dtype(), device=device)


def _apply_per_axis(x: Tensor, mats: Sequence[Tensor]) -> Tensor:
    for axis, m in enumerate(mats):
        x = torch.tensordot(x, m, dims=([2 + axis], [1])).movedim(-1, 2 + axis)
    return x


def _spacings(spacing: float | Sequence[float], d: int) -> list[float]:
    s = [float(v) for v in (spacing if isinstance(spacing, Sequence) else [spacing] * d)]
    if len(s) != d:
        raise ValueError(f"expected {d} spacings, got {len(s)}")
    return s


def to_dense(cp: Tensor, shape: Sequence[int], spacing: float | Sequence[float]) -> Tensor:
    """Evaluate control points ``(B, C, *cp_shape)`` on a dense grid ``(B, C, *shape)``."""
    d = len(shape)
    sp = _spacings(spacing, d)
    mats = [
        weight_matrix(n, s, cp.shape[2 + a], dtype=cp.dtype, device=cp.device)
        for a, (n, s) in enumerate(zip(shape, sp, strict=True))
    ]
    return _apply_per_axis(cp, mats)


def from_dense(field: Tensor, spacing: float | Sequence[float]) -> Tensor:
    """Least-squares fit of control points ``(B, C, *cp_shape)`` to a dense ``field``.

    Exact for anything in the spline space (e.g. polynomials up to cubic). Uses the separable
    pseudo-inverse, which equals the pseudo-inverse of the full tensor-product system.
    """
    shape = field.shape[2:]
    sp = _spacings(spacing, len(shape))
    mats = [
        torch.linalg.pinv(weight_matrix(n, s, dtype=torch.float64)).to(field)
        for n, s in zip(shape, sp, strict=True)
    ]
    return _apply_per_axis(field, mats)


def evaluate(cp: Tensor, coords: Tensor) -> Tensor:
    """Evaluate control points ``(B, C, *cp_shape)`` at continuous control-grid indices.

    ``coords (B, D, *out)`` are in control-point index units (control point ``k`` at ``k``), so the
    four points supporting ``c`` along an axis are ``floor(c) - 1 .. floor(c) + 2``. Use this when
    the evaluation points don't lie on an axis-aligned grid (e.g. an oblique image sampled from a
    world-aligned NiftyReg control grid); otherwise :func:`to_dense` is faster. Indices outside
    the grid are clamped. Returns ``(B, C, *out)``.
    """
    d = coords.shape[1]
    cp_shape = cp.shape[2:]
    if len(cp_shape) != d:
        raise ValueError(f"{d}-D coords for a {len(cp_shape)}-D control grid")
    bsz, c = cp.shape[:2]
    out_shape = coords.shape[2:]
    flat_coords = coords.reshape(coords.shape[0], d, -1).to(cp.dtype)
    if flat_coords.shape[0] != bsz:
        flat_coords = flat_coords.expand(bsz, -1, -1)
    fl = torch.floor(flat_coords)
    base = fl.long() - 1
    w = basis(flat_coords - fl)  # (B, D, N, 4)
    flat_cp = cp.reshape(bsz, c, -1)
    strides = [math.prod(cp_shape[a + 1 :]) for a in range(d)]
    out = torch.zeros(bsz, c, flat_coords.shape[2], dtype=cp.dtype, device=cp.device)
    for offsets in itertools.product(range(4), repeat=d):
        lin = torch.zeros_like(base[:, 0])
        weight = torch.ones_like(flat_coords[:, 0])
        for a, o in enumerate(offsets):
            idx = (base[:, a] + o).clamp(0, cp_shape[a] - 1)
            lin = lin + idx * strides[a]
            weight = weight * w[:, a, :, o]
        vals = flat_cp.gather(2, lin[:, None, :].expand(bsz, c, -1))
        out = out + weight[:, None, :] * vals
    return out.reshape(bsz, c, *out_shape)
