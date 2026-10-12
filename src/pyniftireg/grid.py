"""Identity grids, ``grid_sample`` coordinate conversion and pull-back warping.

Conventions:

- Fields are ``(B, D, *spatial)`` in voxel units. Channel ``i`` is the component along array axis
  ``i`` (not ``grid_sample``'s reversed x/y/z order).
- A displacement ``u`` represents the map ``φ(x) = x + u(x)``. Warping is pull-back:
  ``warp(I, u)(x) = I(φ(x))``.
- This module is the only place that converts to ``grid_sample``'s normalised ``[-1, 1]``
  coordinates, and the only place ``align_corners`` is set.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

ALIGN_CORNERS = True

PaddingMode = Literal["zeros", "border", "reflection"]


def identity(
    shape: Sequence[int], *, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> Tensor:
    """Identity map in voxel coordinates, shape ``(1, D, *shape)``."""
    axes = [torch.arange(n, device=device, dtype=dtype or torch.get_default_dtype()) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]


def to_grid_sample(coords: Tensor, shape: Sequence[int]) -> Tensor:
    """Convert voxel coords ``(B, D, *out)`` into an array of ``shape`` to a ``grid_sample`` grid.

    Returns ``(B, *out, D)`` normalised to ``[-1, 1]`` with the last axis in x/y/z order.
    """
    d = coords.shape[1]
    if d != len(shape):
        raise ValueError(f"coords have {d} components but target shape is {len(shape)}-D")
    scale = torch.tensor(
        [2.0 / max(n - 1, 1) for n in shape], dtype=coords.dtype, device=coords.device
    ).view(1, d, *([1] * (coords.ndim - 2)))
    return (coords * scale - 1.0).flip(1).movedim(1, -1)


def _check_field(field: Tensor) -> int:
    d = field.shape[1] if field.ndim >= 2 else -1
    if field.ndim != d + 2 or d not in (2, 3):
        raise ValueError(
            f"expected a (B, D, *spatial) field with D in (2, 3), got {tuple(field.shape)}"
        )
    return d


def _match_batch(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    if a.shape[0] == b.shape[0]:
        return a, b
    if a.shape[0] == 1:
        return a.expand(b.shape[0], *a.shape[1:]), b
    if b.shape[0] == 1:
        return a, b.expand(a.shape[0], *b.shape[1:])
    raise ValueError(f"batch sizes {a.shape[0]} and {b.shape[0]} are not broadcastable")


def sample_at(
    image: Tensor,
    coords: Tensor,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    padding_mode: PaddingMode = "zeros",
) -> Tensor:
    """Sample ``image (B, C, *in)`` at voxel ``coords (B, D, *out)``, giving ``(B, C, *out)``.

    ``bilinear`` is trilinear in 3D.
    """
    _check_field(coords)
    image, coords = _match_batch(image, coords)
    grid = to_grid_sample(coords.to(image.dtype), image.shape[2:])
    return F.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=ALIGN_CORNERS
    )


def warp(
    image: Tensor,
    disp: Tensor,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    padding_mode: PaddingMode = "zeros",
) -> Tensor:
    """Pull-back warp: ``out(x) = image(x + disp(x))`` on ``disp``'s grid."""
    coords = identity(disp.shape[2:], device=disp.device, dtype=disp.dtype) + disp
    return sample_at(image, coords, mode=mode, padding_mode=padding_mode)


def warp_labels(
    labels: Tensor,
    disp: Tensor,
    *,
    mode: Literal["soft", "nearest"] = "soft",
    num_classes: int | None = None,
) -> Tensor:
    """Warp an integer label map ``(B, 1, *spatial)``.

    ``soft`` warps a one-hot encoding linearly and takes the argmax, which keeps thin structures
    smoother than ``nearest``. Voxels mapped outside the field of view become label 0.
    """
    if labels.shape[1] != 1:
        raise ValueError("labels must have a single channel")
    if mode == "nearest":
        out = warp(labels.to(disp.dtype), disp, mode="nearest")
        return out.round().to(labels.dtype)
    n = int(labels.max()) + 1 if num_classes is None else num_classes
    onehot = F.one_hot(labels[:, 0].long(), n).movedim(-1, 1).to(disp.dtype)
    soft = warp(onehot, disp)
    # Mass pulled from outside the field of view goes to the background class.
    soft[:, 0] += 1.0 - soft.sum(1)
    return soft.argmax(1, keepdim=True).to(labels.dtype)


def sample_points(field: Tensor, points: Tensor, *, padding_mode: PaddingMode = "border") -> Tensor:
    """Linearly sample ``field (B, C, *spatial)`` at voxel ``points (B, N, D)`` -> ``(B, N, C)``."""
    d = points.shape[-1]
    coords = points.movedim(-1, 1).reshape(points.shape[0], d, points.shape[1], *([1] * (d - 1)))
    out = sample_at(field, coords, padding_mode=padding_mode)
    return out.reshape(out.shape[0], out.shape[1], -1).movedim(1, -1)


def warp_points(points: Tensor, disp: Tensor) -> Tensor:
    """Apply the map ``φ = id + disp`` to voxel ``points (B, N, D)``.

    Under a pull-back image warp by ``φ``, points move with the image under ``φ⁻¹``, so pass the
    inverse displacement (``exp(-v)`` for an SVF) to carry landmarks along with a warped image.
    """
    return points + sample_points(disp, points)
