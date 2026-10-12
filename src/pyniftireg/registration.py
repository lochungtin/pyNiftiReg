"""PyTorch registration backend: affine, then a cubic B-spline stationary velocity field.

A NiftyReg-*style* reimplementation (``reg_aladin`` + ``reg_f3d -vel``), not a line-by-line port.
For a reference A and a floating B:

1. **Affine** (or rigid) in world mm, ``Affine * A_world = B_world`` (NiftyReg's convention), found
   by maximising the similarity over an image pyramid, started from aligning the centres of mass.
2. B is resampled onto A's grid with that affine (``B_aligned``), as in the NiftyReg pipeline.
3. **Non-rigid**: a cubic B-spline control-point grid on A's voxel grid holds the SVF ``v`` (voxel
   units). The loss is the (symmetric) dissimilarity between ``B_aligned ∘ exp(v)`` and A (and
   ``A ∘ exp(-v)`` and ``B_aligned``), plus bending energy and linear elasticity computed at the
   control points as NiftyReg does, plus an optional log-Jacobian penalty. The control-point
   spacing doubles at each coarser pyramid level and is refined exactly between levels. Gradients
   come from autograd; the optimiser is Adam.

The result follows :class:`~pyniftireg.result.RegistrationResult`: ``exp(v)`` pulls
``B_aligned`` back onto A. Similarity values in ``qc`` are this module's Parzen NMI, which is close
to but not identical with NiftyReg's ``reg_measure``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor

from pyniftireg import bspline
from pyniftireg.filters import gaussian_smooth
from pyniftireg.flow import auto_steps, exp
from pyniftireg.grid import identity, sample_at, warp
from pyniftireg.jacobian import det, jacobian_stats
from pyniftireg.preprocess import resample_to, with_spacing
from pyniftireg.result import ImageLike, RegistrationResult
from pyniftireg.space import ImageSpace

log = logging.getLogger(__name__)

PathLike = str | Path
Similarity = Literal["nmi", "lncc", "ssd"]


# --- similarity measures -----------------------------------------------------------------------


def _normalise(x: Tensor, mask: Tensor | None) -> Tensor:
    """Map the 0.5/99.5 percentiles of ``x`` (within ``mask``) to [0, 1], clipped."""
    vals = x.detach().flatten() if mask is None else x.detach()[mask.expand_as(x) > 0]
    if vals.numel() > 1_000_000:
        idx = torch.linspace(0, vals.numel() - 1, 1_000_000, device=vals.device).long()
        vals = vals[idx]
    lo, hi = torch.quantile(vals.float(), torch.tensor([0.005, 0.995], device=vals.device))
    out: Tensor = ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)
    return out


def _parzen(x: Tensor, bins: int) -> tuple[Tensor, Tensor]:
    """Cubic B-spline Parzen window: 4 bin indices ``(N, 4)`` and weights ``(N, 4)``."""
    c = x * (bins - 3) + 1.0  # keep the 4-tap support inside [0, bins - 1]
    fl = torch.floor(c).clamp(1, bins - 3)
    idx = fl.long()[:, None] - 1 + torch.arange(4, device=x.device)
    return idx, bspline.basis(c - fl)


def nmi(a: Tensor, b: Tensor, mask: Tensor | None = None, *, bins: int = 64) -> Tensor:
    """Normalised mutual information ``(H(a) + H(b)) / H(a, b)`` in [1, 2] (Parzen joint histogram).

    ``a`` and ``b`` are intensities in [0, 1] of the same shape; differentiable in both.
    """
    av, bv = a.flatten(), b.flatten()
    if mask is not None:
        m = mask.expand_as(a).flatten() > 0
        av, bv = av[m], bv[m]
    ia, wa = _parzen(av, bins)
    ib, wb = _parzen(bv, bins)
    joint_idx = (ia[:, :, None] * bins + ib[:, None, :]).flatten(1)
    joint_w = (wa[:, :, None] * wb[:, None, :]).flatten(1)
    hist = torch.zeros(bins * bins, dtype=a.dtype, device=a.device)
    hist = hist.index_add(0, joint_idx.flatten(), joint_w.flatten())
    p = hist / hist.sum().clamp_min(1e-12)
    pj = p.view(bins, bins)
    eps = 1e-12
    h_ab = -(p * torch.log(p + eps)).sum()
    pa, pb = pj.sum(1), pj.sum(0)
    h_a = -(pa * torch.log(pa + eps)).sum()
    h_b = -(pb * torch.log(pb + eps)).sum()
    return (h_a + h_b) / h_ab.clamp_min(eps)


def lncc(a: Tensor, b: Tensor, mask: Tensor | None, sigma: float | Sequence[float]) -> Tensor:
    """Mean local normalised cross-correlation with a Gaussian window of ``sigma`` voxels."""

    def smooth(x: Tensor) -> Tensor:
        return gaussian_smooth(x, sigma)

    ma, mb = smooth(a), smooth(b)
    va = (smooth(a * a) - ma * ma).clamp_min(1e-6)
    vb = (smooth(b * b) - mb * mb).clamp_min(1e-6)
    cov = smooth(a * b) - ma * mb
    cc = cov / torch.sqrt(va * vb)
    if mask is None:
        return cc.mean()
    m = mask.expand_as(cc).to(cc.dtype)
    return (cc * m).sum() / m.sum().clamp_min(1.0)


def ssd(a: Tensor, b: Tensor, mask: Tensor | None) -> Tensor:
    d2 = (a - b) ** 2
    if mask is None:
        return d2.mean()
    m = mask.expand_as(d2).to(d2.dtype)
    return (d2 * m).sum() / m.sum().clamp_min(1.0)


# --- regularisation on the control-point grid --------------------------------------------------

# Cubic B-spline value / first / second derivative weights at a control point (unit spacing).
_TAPS = {0: (1 / 6, 2 / 3, 1 / 6), 1: (-0.5, 0.0, 0.5), 2: (1.0, -2.0, 1.0)}


def _cp_derivative(cp: Tensor, orders: Sequence[int]) -> Tensor:
    """Derivative of the spline at the inner control points, per-axis ``orders`` (0, 1 or 2)."""
    out = cp
    for axis, o in enumerate(orders):
        k = torch.tensor(_TAPS[o], dtype=cp.dtype, device=cp.device)
        moved = out.movedim(2 + axis, -1)
        shape = moved.shape
        flat = F.conv1d(moved.reshape(-1, 1, shape[-1]), k.view(1, 1, 3))
        out = flat.reshape(*shape[:-1], shape[-1] - 2).movedim(-1, 2 + axis)
    return out


def bending_energy(cp: Tensor, spacing: Sequence[float]) -> Tensor:
    """NiftyReg-style bending energy, evaluated at the control points.

    ``cp`` is in voxels, ``spacing`` the control-point spacing in voxels; derivatives are taken in
    units of the spacing, so the value doesn't depend on the grid resolution.
    """
    d = cp.shape[1]
    scale = torch.tensor(spacing, dtype=cp.dtype, device=cp.device).view(1, d, *[1] * d)
    u = cp / scale  # displacement in units of the spacing (approximately; anisotropy ignored)
    total = cp.new_zeros(())
    for j in range(d):
        for k in range(j, d):
            orders = [0] * d
            orders[j] += 1
            orders[k] += 1
            term = (_cp_derivative(u, orders) ** 2).sum(1).mean()
            total = total + (term if j == k else 2 * term)
    return total


def linear_elasticity(cp: Tensor, spacing: Sequence[float]) -> Tensor:
    """Squared symmetric part of the velocity gradient at the control points."""
    d = cp.shape[1]
    scale = torch.tensor(spacing, dtype=cp.dtype, device=cp.device).view(1, d, *[1] * d)
    u = cp / scale
    grads = []
    for j in range(d):
        orders = [0] * d
        orders[j] = 1
        grads.append(_cp_derivative(u, orders))  # ∂_j u_i for all i, shape (B, D, ...)
    total = cp.new_zeros(())
    for i in range(d):
        for j in range(d):
            sym = 0.5 * (grads[j][:, i] + grads[i][:, j])
            total = total + (sym**2).mean()
    return total


# --- parameters --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TorchRegParams:
    """Registration parameters. Spacings follow NiftyReg: mm if positive, voxels if negative.

    The regularisation terms mirror ``reg_f3d``'s ``-be``, ``-le`` and ``-jl`` but are normalised
    differently: equal weights regularise less here. To reuse ``reg_f3d`` weights, convert them with
    :func:`from_niftyreg_weights`.
    """

    # affine
    rigid_only: bool = False
    affine_levels: int = 3
    affine_iters: int = 150
    affine_lr: float = 0.01
    #: Start the affine from the per-axis scale ratio of the two objects (moment matching).
    init_scale: bool = True
    # non-rigid
    cp_spacing: float | tuple[float, ...] = -5.0
    bending_energy: float = 0.001
    linear_elasticity: float = 0.01
    jacobian_log: float = 0.0
    similarity: Similarity = "nmi"
    lncc_sigma: float = -5.0
    bins: int = 64
    levels: int = 3
    #: Levels actually run, from the coarsest (NiftyReg's ``-lp``); None runs all of them.
    levels_to_perform: int | None = None
    max_iter: int = 150
    lr: float = 0.05
    symmetric: bool = True
    #: Voxels used per similarity evaluation (random subset when the mask is larger).
    max_samples: int = 2**20
    seed: int = 0
    device: str | None = None


#: Multipliers that turn ``reg_f3d`` weights (``-be``, ``-le``, ``-jl``) into this backend's, so
#: that both give comparable det J statistics. Calibrated on real data (2026-10-12): 24 pairs of
#: labelled Duke breast MRI (2 mm, 10 mm control points) and 20 pairs of TotalSegmentator thorax
#: MRI (3 mm, 15 mm), comparing x3/x30, x5/x50 and x10/x100. x5/x50 gave the best Dice on both
#: (final: breast 0.924 vs NiftyReg 0.905; thorax 0.802) with det J tails close to NiftyReg's;
#: x10/x100 (the earlier phantom calibration) was too stiff on the thorax and x3/x30 produced
#: extreme local expansions (det J up to ~20) on hard breast pairs.
NIFTYREG_WEIGHT_SCALE = {"bending_energy": 5.0, "linear_elasticity": 5.0, "jacobian_log": 50.0}


def from_niftyreg_weights(
    bending_energy: float, linear_elasticity: float, jacobian_log: float
) -> dict[str, float]:
    """This backend's regularisation weights equivalent to ``reg_f3d``'s ``-be``/``-le``/``-jl``."""
    s = NIFTYREG_WEIGHT_SCALE
    return {
        "bending_energy": bending_energy * s["bending_energy"],
        "linear_elasticity": linear_elasticity * s["linear_elasticity"],
        "jacobian_log": jacobian_log * s["jacobian_log"],
    }


def _per_axis(value: float | tuple[float, ...], space: ImageSpace) -> list[float]:
    """NiftyReg spacing convention -> voxels per axis."""
    vals = value if isinstance(value, tuple) else (value,) * space.ndim
    if len(vals) != space.ndim:
        raise ValueError(f"expected {space.ndim} values, got {len(vals)}")
    return [(-v if v < 0 else v / s) for v, s in zip(vals, space.spacing, strict=True)]


# --- the backend -------------------------------------------------------------------------------


@dataclass
class TorchReg:
    """PyTorch registration backend (implements :class:`~pyniftireg.result.Registration`).

    Runs on the GPU when available (``params.device``). Besides the file-based :meth:`register`,
    :meth:`register_arrays` works on tensors directly.
    """

    params: TorchRegParams = field(default_factory=TorchRegParams)
    verbose: bool = False
    #: Which affine start won in the last registration (0: identity scale, 1: moment matching).
    last_affine_start: int = 0
    #: One GPU, and torch.linalg's lazy initialisation isn't thread-safe: don't run it in threads.
    thread_safe: ClassVar[bool] = False

    @property
    def device(self) -> torch.device:
        if self.params.device is not None:
            return torch.device(self.params.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def register(
        self,
        reference: ImageLike,
        floating: ImageLike,
        *,
        reference_mask: ImageLike | None = None,
        floating_mask: ImageLike | None = None,
        reference_space: ImageSpace | None = None,
        floating_space: ImageSpace | None = None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        """Register ``floating`` (B) to ``reference`` (A): affine, then a non-rigid SVF.

        Images and masks are NIfTI paths, or numpy arrays / torch tensors of shape ``(*spatial)``
        or ``(1, 1, *spatial)``. An array's grid is ``reference_space`` / ``floating_space``
        (default: 1 mm voxels, identity orientation); a file carries its own, so passing a space
        with a path is an error. A mask must have its image's shape.

        ``affine``: skip the affine step and use this one (``affine @ A_world = B_world``).
        """
        a, sa = _as_image(reference, reference_space, "reference")
        b, sb = _as_image(floating, floating_space, "floating")
        ma = None if reference_mask is None else _as_mask(reference_mask, sa, "reference_mask")
        mb = None if floating_mask is None else _as_mask(floating_mask, sb, "floating_mask")
        return self.register_arrays(
            a, sa, b, sb, reference_mask=ma, floating_mask=mb, affine=affine
        )

    def register_arrays(
        self,
        reference: Tensor,
        reference_space: ImageSpace,
        floating: Tensor,
        floating_space: ImageSpace,
        *,
        reference_mask: Tensor | None = None,
        floating_mask: Tensor | None = None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        """Register ``floating`` (B) to ``reference`` (A), both ``(1, 1, *spatial)``.

        ``affine``: skip the affine step and use this one (``affine @ A_world = B_world``).
        """
        t0 = time.perf_counter()
        dev = self.device
        a = reference.to(dev, torch.float32)[:, :1]
        b = floating.to(dev, torch.float32)[:, :1]
        ma = None if reference_mask is None else (reference_mask.to(dev)[:, :1] > 0)
        mb = None if floating_mask is None else (floating_mask.to(dev)[:, :1] > 0)
        a = _normalise(a, ma)
        b = _normalise(b, mb)
        if affine is None:
            affine = self._affine(a, reference_space, b, floating_space, ma, mb)
        t1 = time.perf_counter()
        b_al = resample_to(b, floating_space, reference_space, affine=affine)
        mb_al = None
        if mb is not None:
            mb_al = resample_to(mb.float(), floating_space, reference_space, affine=affine) > 0.5
        v = self._velocity(a, b_al, reference_space, ma, mb_al)
        t2 = time.perf_counter()
        with torch.no_grad():
            u = exp(v)
            qc = {
                "nmi_affine": float(nmi(a, b_al, ma, bins=self.params.bins)),
                "nmi": float(nmi(a, warp(b_al, u), ma, bins=self.params.bins)),
                "time_affine_s": t1 - t0,
                "time_nonrigid_s": t2 - t1,
            }
            stats = jacobian_stats(u.cpu())
        return RegistrationResult(
            space=reference_space,
            affine=affine,
            velocity=v.detach().cpu(),
            jacobian=stats,
            qc=qc,
        )

    # --- affine --------------------------------------------------------------------------------

    def _similarity(
        self, a: Tensor, b: Tensor, mask: Tensor | None, sigma_vox: list[float]
    ) -> Tensor:
        """Similarity to maximise."""
        p = self.params
        if p.similarity == "nmi":
            return nmi(a, b, mask, bins=p.bins)
        if p.similarity == "lncc":
            return lncc(a, b, mask, sigma_vox)
        return -ssd(a, b, mask)

    def _affine(
        self,
        a: Tensor,
        sa: ImageSpace,
        b: Tensor,
        sb: ImageSpace,
        ma: Tensor | None,
        mb: Tensor | None,
    ) -> npt.NDArray[np.float64]:
        p = self.params
        d = sa.ndim
        dev = a.device
        com_a, sd_a = _moments(a if ma is None else ma.float(), sa)
        com_b, sd_b = _moments(b if mb is None else mb.float(), sb)
        # Affine(x) = L (x - c) + c + t, with c = A's centre of mass.
        c = torch.tensor(com_a, dtype=torch.float32, device=dev)
        # Translation in mm moves much faster than the dimensionless linear part.
        extent = float(np.max((np.array(sa.shape) - 1) * sa.spacing))
        b_inv = torch.as_tensor(np.linalg.inv(sb.affine), dtype=torch.float32, device=dev)
        levels = list(reversed(range(p.affine_levels)))
        similarity = self._similarity
        cache: dict[int, tuple[Tensor, Tensor | None, Tensor, Tensor, list[float]]] = {}

        def level_data(level: int) -> tuple[Tensor, Tensor | None, Tensor, Tensor, list[float]]:
            if level not in cache:
                f = 2**level
                al, ml, space_l = _pyramid(a, ma, sa, f)
                bl = (
                    gaussian_smooth(b, [0.5 * f * sa.spacing.mean() / s for s in sb.spacing])
                    if f > 1
                    else b
                )
                world = space_l.voxel_to_world(identity(space_l.shape, device=dev).movedim(1, -1))
                cache[level] = (al, ml, bl, world, _per_axis(p.lncc_sigma, space_l))
            return cache[level]

        class Start:
            """One optimisation run: translation plus a linear part (or rotation angles)."""

            def __init__(self, scale: npt.NDArray[np.float64]) -> None:
                self.t = torch.tensor(com_b - com_a, dtype=torch.float32, device=dev)
                self.t.requires_grad_(True)
                if p.rigid_only:
                    self.lin = torch.zeros(d * (d - 1) // 2, device=dev, requires_grad=True)
                else:
                    diag = torch.tensor(scale, dtype=torch.float32, device=dev)
                    self.lin = torch.diag(diag).requires_grad_(True)
                self.opt = torch.optim.Adam(
                    [
                        {"params": [self.t], "lr": p.affine_lr * extent},
                        {"params": [self.lin], "lr": p.affine_lr},
                    ]
                )
                self.base_lrs = [g["lr"] for g in self.opt.param_groups]

            def matrix(self) -> Tensor:
                if not p.rigid_only:
                    return self.lin
                skew = torch.zeros(d, d, device=dev)
                iu = torch.triu_indices(d, d, 1, device=dev)
                skew = skew.index_put((iu[0], iu[1]), self.lin)
                rotation: Tensor = torch.linalg.matrix_exp(skew - skew.T)
                return rotation

            def loss(self, level: int) -> Tensor:
                al, ml, bl, world, sigma = level_data(level)
                mapped = (world - c) @ self.matrix().T + c + self.t
                vox = mapped @ b_inv[:d, :d].T + b_inv[:d, d]
                return -similarity(al, sample_at(bl, vox.movedim(-1, 1)), ml, sigma)

            def run(self, level: int) -> float:
                for group, lr in zip(self.opt.param_groups, self.base_lrs, strict=True):
                    group["lr"] = lr
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, p.affine_iters)
                for _ in range(p.affine_iters):
                    self.opt.zero_grad()
                    self.loss(level).backward()  # type: ignore[no-untyped-call]
                    self.opt.step()
                    sched.step()
                with torch.no_grad():
                    return float(self.loss(level))

        # Multi-start at the coarsest level: identity scale, and the per-axis scale between the
        # two objects' spatial spreads (moment matching). From identity, NMI descent can settle
        # on the wrong scale when subjects differ a lot in size (a Duke pair ended at 0.8
        # instead of ~1.2); the moment estimate in turn is misled when the masks include parts
        # whose position varies (arms in thorax scans). The better similarity decides.
        scales = [np.ones(d)]
        if p.init_scale and not p.rigid_only:
            moment = np.clip(sd_b / np.maximum(sd_a, 1e-6), 0.5, 2.0)
            if not np.allclose(moment, 1.0, atol=0.02):
                scales.append(moment)
        starts = [Start(sc) for sc in scales]
        losses = [st.run(levels[0]) for st in starts]
        best = int(np.argmin(losses))
        start = starts[best]
        for level in levels[1:]:
            start.run(level)
        self.last_affine_start = best  # 0: identity scale, 1: moment matching
        if self.verbose:
            log.info("affine start %d chosen (coarse losses %s)", best, losses)
        with torch.no_grad():
            mat = start.matrix().double().cpu().numpy()
            out = np.eye(d + 1)
            cc, tt = c.double().cpu().numpy(), start.t.detach().double().cpu().numpy()
            out[:d, :d] = mat
            out[:d, d] = cc + tt - mat @ cc
        return out

    # --- non-rigid -----------------------------------------------------------------------------

    def _velocity(
        self,
        a: Tensor,
        b: Tensor,
        space: ImageSpace,
        ma: Tensor | None,
        mb: Tensor | None,
    ) -> Tensor:
        """Dense SVF ``(1, D, *space.shape)`` (voxels) with ``b ∘ exp(v) ≈ a``."""
        p = self.params
        d = space.ndim
        s_final = _per_axis(p.cp_spacing, space)
        gen = torch.Generator(device=a.device).manual_seed(p.seed)
        cp: Tensor | None = None
        cp_spacing: list[float] = []
        n_levels = p.levels if p.levels_to_perform is None else p.levels_to_perform
        for level in reversed(range(p.levels)):
            if level < p.levels - n_levels:
                break
            f = 2**level
            al, mal, space_l = _pyramid(a, ma, space, f)
            bl, mbl, _ = _pyramid(b, mb, space, f)
            # control points: spacing (in full-resolution voxels) doubles at each coarser level
            new_spacing = [s * f for s in s_final]
            n_cp = [
                bspline.num_control_points(n, s)
                for n, s in zip(space.shape, new_spacing, strict=True)
            ]
            if cp is None:
                cp = torch.zeros(1, d, *n_cp, device=a.device)
            else:
                cp = _refine(cp, cp_spacing, new_spacing, space.shape)
            cp_spacing = new_spacing
            cp = cp.detach().requires_grad_(True)
            cp = self._optimise_level(cp, cp_spacing, f, al, bl, mal, mbl, space_l, gen)
        assert cp is not None
        with torch.no_grad():
            return bspline.to_dense(cp, space.shape, cp_spacing)

    def _optimise_level(
        self,
        cp: Tensor,
        cp_spacing: list[float],
        f: int,
        a: Tensor,
        b: Tensor,
        ma: Tensor | None,
        mb: Tensor | None,
        space_l: ImageSpace,
        gen: torch.Generator,
    ) -> Tensor:
        p = self.params
        level_spacing = [s / f for s in cp_spacing]  # control-point spacing in level voxels
        mats = [
            bspline.weight_matrix(n, s, cp.shape[2 + i], device=cp.device)
            for i, (n, s) in enumerate(zip(space_l.shape, level_spacing, strict=True))
        ]
        sigma = _per_axis(p.lncc_sigma, space_l)
        opt = torch.optim.Adam([cp], lr=p.lr * f)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p.max_iter, 0.05 * p.lr * f)
        n_vox = math.prod(space_l.shape)
        best = math.inf
        stall = 0
        iters = 0
        for _ in range(p.max_iter):
            iters += 1
            opt.zero_grad()
            v = bspline._apply_per_axis(cp, mats) / f  # level voxel units
            n_sq = auto_steps(v)
            u = exp(v, steps=n_sq)
            warped = warp(b, u, padding_mode="border")
            sim = self._sample_similarity(a, warped, ma, sigma, n_vox, gen)
            if p.symmetric:
                u_inv = exp(v, -1.0, steps=n_sq)
                sim = 0.5 * (
                    sim
                    + self._sample_similarity(
                        b, warp(a, u_inv, padding_mode="border"), mb, sigma, n_vox, gen
                    )
                )
            loss = -sim
            if p.bending_energy > 0:
                loss = loss + p.bending_energy * bending_energy(cp, cp_spacing)
            if p.linear_elasticity > 0:
                loss = loss + p.linear_elasticity * linear_elasticity(cp, cp_spacing)
            if p.jacobian_log > 0:
                dj = det(u).clamp_min(1e-4)
                loss = loss + p.jacobian_log * (torch.log(dj) ** 2).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            sched.step()
            value = float(loss.detach())
            if value < best - 1e-5:
                best, stall = value, 0
            else:
                stall += 1
                if stall >= 20:
                    break
        if self.verbose:
            log.info("level /%d: %d iterations, loss %.4f", f, iters, best)
        return cp.detach()

    def _sample_similarity(
        self,
        a: Tensor,
        b: Tensor,
        mask: Tensor | None,
        sigma: list[float],
        n_vox: int,
        gen: torch.Generator,
    ) -> Tensor:
        p = self.params
        if p.similarity == "nmi" and n_vox > p.max_samples:
            keep = torch.rand(a.shape, generator=gen, device=a.device) < p.max_samples / n_vox
            mask = keep if mask is None else (mask & keep)
        return self._similarity(a, b, mask, sigma)


def _as_tensor(x: npt.NDArray[Any] | Tensor, ndim: int | None, name: str) -> Tensor:
    """``(1, 1, *spatial)`` view of an array; ``ndim`` is the spatial dimension, if known."""
    t = x if isinstance(x, Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    d = ndim if ndim is not None else (t.ndim if t.ndim in (2, 3) else t.ndim - 2)
    if t.ndim == d + 2 and t.shape[:2] == (1, 1):
        return t
    if t.ndim == d and d in (2, 3):
        return t[None, None]
    raise ValueError(
        f"{name}: expected shape (*spatial) or (1, 1, *spatial) in 2D/3D, got {tuple(t.shape)}"
    )


def _as_image(x: ImageLike, space: ImageSpace | None, name: str) -> tuple[Tensor, ImageSpace]:
    """Load a file or wrap an array, with its space."""
    if isinstance(x, str | Path):
        if space is not None:
            raise ValueError(
                f"{name}_space is for arrays only; {name} is a file with its own space"
            )
        from pyniftireg.io import load_nifti

        return load_nifti(x)
    t = _as_tensor(x, None if space is None else space.ndim, name)
    if space is None:
        return t, ImageSpace.from_spacing(t.shape[2:])
    if tuple(t.shape[2:]) != space.shape:
        raise ValueError(f"{name} has shape {tuple(t.shape[2:])}, its space {space.shape}")
    return t, space


def _as_mask(x: ImageLike, space: ImageSpace, name: str) -> Tensor:
    """A mask on the grid ``space`` of its image."""
    if isinstance(x, str | Path):
        from pyniftireg.io import load_nifti

        t = load_nifti(x)[0]
    else:
        t = _as_tensor(x, space.ndim, name)
    if tuple(t.shape[2:]) != space.shape:
        raise ValueError(f"{name} has shape {tuple(t.shape[2:])}, its image {space.shape}")
    return t


def _moments(
    x: Tensor, space: ImageSpace
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Weighted centre of mass (world mm) and standard deviation along each world axis."""
    w = x.detach()[0, 0].double().clamp_min(0).flatten()
    idx = identity(space.shape, device=w.device, dtype=torch.float64)[0].flatten(1)
    total = w.sum().clamp_min(1e-12)
    mean = (idx * w).sum(1) / total
    centred = idx - mean[:, None]
    cov_vox = (centred * w) @ centred.T / total
    lin = torch.tensor(space.affine[: space.ndim, : space.ndim], dtype=torch.float64)
    cov_world = lin @ cov_vox.cpu() @ lin.T
    com = space.voxel_to_world(mean[None].cpu())[0]
    sd = torch.sqrt(torch.diagonal(cov_world).clamp_min(0))
    return np.asarray(com.numpy(), dtype=np.float64), np.asarray(sd.numpy(), dtype=np.float64)


def _pyramid(
    x: Tensor, mask: Tensor | None, space: ImageSpace, f: int
) -> tuple[Tensor, Tensor | None, ImageSpace]:
    """Pyramid level ``f`` (downsampling factor): level voxel j sits at full voxel j·f."""
    if f == 1:
        return x, mask, space
    sl = with_spacing(space, [float(s) * f for s in space.spacing])
    smooth = gaussian_smooth(x, 0.5 * f)
    xl = resample_to(smooth, space, sl, padding_mode="border")
    ml = None
    if mask is not None:
        ml = resample_to(mask.float(), space, sl, mode="nearest") > 0.5
    return xl, ml, sl


def _refine(cp: Tensor, old: list[float], new: list[float], shape: Sequence[int]) -> Tensor:
    """Re-express a control grid at a finer spacing (exact: nested knots)."""
    with torch.no_grad():
        dense = bspline.to_dense(cp, shape, old)
        return bspline.from_dense(dense, new)
