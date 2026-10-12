"""Image preprocessing and resampling between grids.

Images are ``(B, C, *spatial)`` tensors with an :class:`ImageSpace`. Label maps use the same
functions with ``mode="nearest"``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pyniftireg.filters import gaussian_smooth
from pyniftireg.grid import identity, sample_at
from pyniftireg.space import ImageSpace

#: World axis and sign for each orientation letter (NIfTI world is RAS+).
_AXIS_CODES = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}


def orientation_codes(space: ImageSpace) -> str:
    """Axis codes of ``space``, e.g. ``"RAS"``: the world direction each voxel axis points to."""
    codes = {v: k for k, v in _AXIS_CODES.items()}
    return "".join(codes[ws] for ws in _axis_map(space))


def _axis_map(space: ImageSpace) -> list[tuple[int, int]]:
    """``(world axis, sign)`` for each voxel axis, assigning the strongest cosines first."""
    d = space.ndim
    cos = np.abs(space.direction)  # (world, voxel)
    out: list[tuple[int, int] | None] = [None] * d
    free_w, free_v = set(range(d)), set(range(d))
    for _ in range(d):
        w, v = max(((w, v) for w in free_w for v in free_v), key=lambda p: cos[p])
        out[v] = (w, 1 if space.direction[w, v] > 0 else -1)
        free_w.discard(w)
        free_v.discard(v)
    return [o for o in out if o is not None]


def reorient(x: Tensor, space: ImageSpace, target: str = "RAS") -> tuple[Tensor, ImageSpace]:
    """Permute and flip the voxel axes of ``x (B, C, *spatial)`` so they point along ``target``.

    No interpolation: the world position of every voxel is unchanged. 2D spaces use the first two
    letters of ``target``.
    """
    d = space.ndim
    target = target.upper()[:d]
    want = [_AXIS_CODES[c] for c in target]
    if sorted(w for w, _ in want) != list(range(d)):
        raise ValueError(f"orientation {target!r} must name each of the {d} world axes once")
    have = _axis_map(space)
    perm, flips = [], []
    for w, sign in want:
        v = next(i for i, (hw, _) in enumerate(have) if hw == w)
        perm.append(v)
        flips.append(have[v][1] != sign)
    # new index i <- old index perm[i] (flipped: n - 1 - i)
    m = np.zeros((d + 1, d + 1))
    m[d, d] = 1
    new_shape = tuple(space.shape[p] for p in perm)
    for i, (p, f) in enumerate(zip(perm, flips, strict=True)):
        m[p, i] = -1.0 if f else 1.0
        m[p, d] = new_shape[i] - 1 if f else 0.0
    out = x.permute(0, 1, *[2 + p for p in perm])
    dims = [2 + i for i, f in enumerate(flips) if f]
    if dims:
        out = out.flip(dims)
    return out.contiguous(), ImageSpace(new_shape, space.affine @ m)


def resample_to(
    x: Tensor,
    src: ImageSpace,
    dst: ImageSpace,
    *,
    affine: np.ndarray | None = None,
    mode: Literal["linear", "nearest"] = "linear",
    padding_mode: Literal["zeros", "border"] = "zeros",
) -> Tensor:
    """Resample ``x`` on ``src`` onto the grid of ``dst`` through world space.

    ``affine``: optional world-mm matrix with ``affine @ dst_world = src_world`` (NiftyReg's
    ``Affine * Reference = Floating`` with ``dst`` the reference), e.g. to put B onto A's grid.
    """
    coords = _src_coords(src, dst, affine, x.device)
    out = sample_at(
        x.to(coords.dtype) if mode == "nearest" else x,
        coords,
        mode="nearest" if mode == "nearest" else "bilinear",
        padding_mode=padding_mode,
    )
    return out.to(x.dtype) if mode == "nearest" else out


def _src_coords(
    src: ImageSpace, dst: ImageSpace, affine: np.ndarray | None, device: torch.device
) -> Tensor:
    if src.ndim != dst.ndim:
        raise ValueError("src and dst must have the same dimensionality")
    m = np.linalg.inv(src.affine) @ (np.eye(src.ndim + 1) if affine is None else affine)
    m = m @ dst.affine  # dst voxel -> src voxel
    d = dst.ndim
    lin = torch.as_tensor(m[:d, :d], dtype=torch.float32, device=device)
    off = torch.as_tensor(m[:d, d], dtype=torch.float32, device=device)
    x = identity(dst.shape, device=device, dtype=torch.float32)
    return torch.einsum("ij,bj...->bi...", lin, x) + off.view(1, d, *([1] * d))


def resample_field(v: Tensor, src: ImageSpace, dst: ImageSpace) -> Tensor:
    """Re-express a voxel-unit field ``(B, D, *src.shape)`` on the ``dst`` grid.

    Both spaces must describe the same world frame (same subject, different sampling). Vectors are
    converted through world mm, so the map ``x + v(x)`` is the same physical map on both grids.
    """
    coords = _src_coords(src, dst, None, v.device)
    on_dst = sample_at(v, coords.to(v.dtype), padding_mode="border")
    return dst.displacement_to_voxel(src.displacement_to_world(on_dst))


def with_spacing(space: ImageSpace, spacing: float | Sequence[float]) -> ImageSpace:
    """Space covering the same extent as ``space`` (same origin and directions) at ``spacing``."""
    d = space.ndim
    sp = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (d,))
    extent = (np.array(space.shape) - 1) * space.spacing
    shape = tuple(math.floor(e / s + 1e-6) + 1 for e, s in zip(extent, sp, strict=True))
    affine = np.array(space.affine)
    affine[:d, :d] = space.direction * sp
    return ImageSpace(shape, affine)


def resample(
    x: Tensor,
    space: ImageSpace,
    spacing: float | Sequence[float],
    *,
    mode: Literal["linear", "nearest"] = "linear",
) -> tuple[Tensor, ImageSpace]:
    """Resample to a new voxel ``spacing`` in mm (see :func:`with_spacing`)."""
    dst = with_spacing(space, spacing)
    return resample_to(x, space, dst, mode=mode, padding_mode="border"), dst


def _quantile(x: Tensor, q: float) -> Tensor:
    k = min(max(round(q * (x.numel() - 1)) + 1, 1), x.numel())
    return x.kthvalue(k).values


def normalize_percentile(
    x: Tensor,
    lo: float = 1.0,
    hi: float = 99.0,
    *,
    mask: Tensor | None = None,
    clip: bool = True,
) -> Tensor:
    """Map the ``lo``/``hi`` percentiles of each sample and channel (within ``mask``) to 0/1."""
    out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, torch.float32))
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            vals = x[b, c] if mask is None else x[b, c][mask[min(b, mask.shape[0] - 1), 0] != 0]
            vals = vals.flatten().to(out.dtype)
            if vals.numel() == 0:
                raise ValueError("normalisation mask is empty")
            a, z = _quantile(vals, lo / 100), _quantile(vals, hi / 100)
            out[b, c] = (x[b, c] - a) / (z - a).clamp_min(1e-8)
    return out.clamp(0.0, 1.0) if clip else out


def n4_bias_correction(
    x: Tensor, space: ImageSpace, *, mask: Tensor | None = None, shrink: int = 4, iters: int = 50
) -> Tensor:
    """N4 bias-field correction of each sample/channel (needs SimpleITK, the ``n4`` extra)."""
    try:
        import SimpleITK
    except ImportError as e:  # pragma: no cover - depends on installed extras
        raise ImportError("N4 needs SimpleITK: pip install 'pyniftireg[n4]'") from e
    sitk: Any = SimpleITK  # untyped
    d = space.ndim
    rev = tuple(range(d))[::-1]  # SimpleITK arrays are (z, y, x)
    out = torch.empty_like(x)
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            arr = x[b, c].detach().cpu().double().numpy().transpose(rev)
            img = sitk.GetImageFromArray(np.ascontiguousarray(arr).astype(np.float32))
            img.SetSpacing([float(s) for s in space.spacing])
            small = sitk.Shrink(img, [shrink] * d)
            f = sitk.N4BiasFieldCorrectionImageFilter()
            f.SetMaximumNumberOfIterations([iters] * 4)
            if mask is None:
                small_mask = sitk.OtsuThreshold(small, 0, 1, 200)
            else:
                m = mask[min(b, mask.shape[0] - 1), 0].detach().cpu().numpy().transpose(rev)
                mimg = sitk.GetImageFromArray(np.ascontiguousarray(m != 0).astype(np.uint8))
                mimg.CopyInformation(img)
                small_mask = sitk.Shrink(mimg, [shrink] * d)
            f.Execute(small, small_mask)
            log_bias = f.GetLogBiasFieldAsImage(img)
            corrected = sitk.GetArrayFromImage(img / sitk.Exp(log_bias)).transpose(rev)
            out[b, c] = torch.from_numpy(np.ascontiguousarray(corrected)).to(x)
    return out


def dilate(mask: Tensor, space: ImageSpace, radius_mm: float) -> Tensor:
    """Binary dilation of ``mask (B, 1, *spatial)`` with a box of ``radius_mm`` per axis."""
    radii = [round(radius_mm / s) for s in space.spacing]
    out = (mask != 0).float()
    if radius_mm <= 0 or not any(radii):
        return out > 0
    # a box maximum is separable: one 1-D max-pool per axis
    for axis, r in enumerate(radii):
        if r == 0:
            continue
        moved = out.movedim(2 + axis, -1)
        shape = moved.shape
        flat = F.max_pool1d(moved.reshape(-1, 1, shape[-1]), 2 * r + 1, stride=1, padding=r)
        out = flat.reshape(shape).movedim(-1, 2 + axis)
    return out > 0


def erode(mask: Tensor, space: ImageSpace, radius_mm: float) -> Tensor:
    """Binary erosion (the dual of :func:`dilate`); voxels outside the image count as inside."""
    return ~dilate(mask == 0, space, radius_mm)


def otsu_threshold(x: Tensor, *, bins: int = 256, max_samples: int = 2_000_000) -> float:
    """Otsu's threshold of the values in ``x`` (maximises the between-class variance)."""
    vals = x.detach().flatten().float()
    if vals.numel() > max_samples:
        vals = vals[:: math.ceil(vals.numel() / max_samples)]
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return lo
    hist = torch.histc(vals, bins, lo, hi).double()
    centres = torch.linspace(lo, hi, bins + 1, dtype=torch.float64, device=hist.device)
    centres = 0.5 * (centres[1:] + centres[:-1])
    w0 = hist.cumsum(0)
    w1 = hist.sum() - w0
    m0 = (hist * centres).cumsum(0) / w0.clamp_min(1e-12)
    m1 = ((hist * centres).sum() - (hist * centres).cumsum(0)) / w1.clamp_min(1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    # Well-separated classes give a plateau (empty bins between them): take its middle, not the
    # first bin, which would sit right at the top of the lower class.
    top = (between >= between.max() * (1 - 1e-9)).nonzero()[:, 0]
    return float(0.5 * (centres[int(top.min())] + centres[int(top.max())]))


def foreground_mask(
    image: Tensor,
    space: ImageSpace,
    threshold: float | Literal["otsu"] = "otsu",
    *,
    smooth_mm: float = 2.0,
    open_mm: float = 3.0,
) -> Tensor:
    """Body/foreground mask ``(B, 1, *spatial)`` of an image.

    The image is smoothed (``smooth_mm``) so background noise doesn't cross the threshold
    (``"otsu"`` or a value), then opened (eroded and dilated by ``open_mm``) to remove the
    remaining speckle, which a later dilation would otherwise grow into large false regions.
    """
    x = image[:, :1].float()
    if smooth_mm > 0:
        x = gaussian_smooth(x, [smooth_mm / s for s in space.spacing])
    th = otsu_threshold(x) if threshold == "otsu" else float(threshold)
    m = x > th
    if open_mm > 0:
        m = dilate(erode(m, space, open_mm), space, open_mm)
    return m


def label_mask(labels: Tensor, values: Sequence[int] = ()) -> Tensor:
    """Union of ``values`` in ``labels`` (all non-zero labels if empty)."""
    if not values:
        return labels != 0
    return torch.isin(labels, torch.as_tensor(list(values), device=labels.device))


def preprocess(
    image: Tensor,
    space: ImageSpace,
    *,
    labels: Tensor | None = None,
    orientation: str | None = "RAS",
    spacing: float | Sequence[float] | None = None,
    n4: bool = False,
    normalize: tuple[float, float] | None = (1.0, 99.0),
    clip: bool = True,
) -> tuple[Tensor, Tensor | None, ImageSpace]:
    """Reorient, N4-correct, resample and normalise an image (and its labels, consistently)."""
    if orientation is not None:
        image, new_space = reorient(image, space, orientation)
        if labels is not None:
            labels, _ = reorient(labels, space, orientation)
        space = new_space
    if n4:
        image = n4_bias_correction(image.float(), space)
    if spacing is not None:
        image, new_space = resample(image.float(), space, spacing)
        if labels is not None:
            labels, _ = resample(labels, space, spacing, mode="nearest")
        space = new_space
    if normalize is not None:
        image = normalize_percentile(image, *normalize, clip=clip)
    return image, labels, space
