"""Diffeomorphic flows of stationary velocity fields (SVFs).

All functions take and return displacement fields ``(B, D, *spatial)`` in voxel units, where a
displacement ``u`` stands for the map ``φ = id + u`` (see :mod:`pyniftireg.grid`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from pyniftireg.grid import PaddingMode, _check_field, identity, sample_at, warp

#: Scaling and squaring starts from a step no larger than this (first-order accurate).
MAX_STEP_VOXELS = 0.5
#: RK4 path integration takes steps no larger than this.
MAX_RK_STEP_VOXELS = 2.0

Weight = float | Tensor


def compose(u1: Tensor, u2: Tensor, *, padding_mode: PaddingMode = "border") -> Tensor:
    """Displacement of ``φ1 ∘ φ2``, i.e. ``x ↦ x + u2(x) + u1(x + u2(x))``."""
    return u2 + warp(u1, u2, padding_mode=padding_mode)


def _as_weight(t: Weight, v: Tensor) -> Tensor | float:
    """Normalise a time ``t`` to something that broadcasts against ``v``.

    Accepts a float, a per-sample tensor ``(B,)``, or a weight map ``(B or 1, 1, *spatial)``.
    """
    if not isinstance(t, Tensor):
        return float(t)
    d = v.ndim - 2
    if t.ndim == 0:
        return t.to(v)
    if t.ndim == 1:
        return t.to(v).view(-1, 1, *([1] * d))
    if t.ndim == v.ndim and t.shape[1] == 1 and t.shape[2:] == v.shape[2:]:
        return t.to(v)
    raise ValueError(
        f"t must be a float, a (B,) tensor or a (B, 1, *spatial) map, got shape {tuple(t.shape)}"
    )


def auto_steps(v: Tensor, max_step: float = MAX_STEP_VOXELS) -> int:
    """Number of squarings so that ``max‖v‖ / 2**n <= max_step``."""
    m = float(v.detach().norm(dim=1).max())
    if m <= max_step:
        return 0
    return math.ceil(math.log2(m / max_step))


def exp(v: Tensor, t: Weight = 1.0, *, steps: int | Literal["auto"] = "auto") -> Tensor:
    """``exp(t·v)`` by scaling and squaring.

    ``t`` may be a float, per-sample ``(B,)`` or a weight map ``(B, 1, *spatial)``; the map case is
    the Eulerian regional flow ``exp(w(x)·v(x))``. Any real ``t`` is valid: ``t < 0`` and ``t > 1``
    extrapolate, and ``exp(-v)`` is the exact inverse of ``exp(v)``.
    """
    _check_field(v)
    w = v * _as_weight(t, v)
    n = auto_steps(w) if steps == "auto" else steps
    u: Tensor = w / 2**n
    for _ in range(n):
        u = compose(u, u)
    return u


def _rk4(y: Tensor, speed: Callable[[Tensor], Tensor], n: int) -> Tensor:
    """Advance positions ``y (B, D, *spatial)`` from s=0 to s=1 under ``dy/ds = speed(y)``."""
    h = 1.0 / n
    for _ in range(n):
        k1 = speed(y)
        k2 = speed(y + 0.5 * h * k1)
        k3 = speed(y + 0.5 * h * k2)
        k4 = speed(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _path(x: Tensor, tau: Tensor | float, vel: Callable[[Tensor], Tensor], n: int) -> Tensor:
    """``flow_v(x, τ)`` for a per-point (or global) time ``τ``."""
    return _rk4(x, lambda y: tau * vel(y), n)


def integrate(
    v: Tensor,
    w: Weight = 1.0,
    *,
    steps: int | Literal["auto"] = "auto",
    lagrangian: bool = True,
    min_steps: int = 4,
    newton_iters: int = 8,
    tol: float = 1e-4,
) -> Tensor:
    """Flow of ``v`` with a spatially varying time ``w``, integrated along each point's path.

    - Eulerian (``lagrangian=False``): ``w`` is fixed in space; ``ψ = exp(w·v)``, i.e. the flow of
      the velocity ``w(y)·v(y)``.
    - Lagrangian: ``w`` is attached to the tissue. The result ``ψ`` satisfies
      ``ψ(x) = flow_v(x, w(ψ(x)))``: each point is moved by the time assigned to its *source*
      location ``ψ(x)``. So when ``ψ`` is used as a pull-back warp, ``w`` lives in the input
      image's space, e.g. built from that image's labels. Each point's time ``τ = w(ψ(x))`` is
      found by a per-point Newton iteration.

    Paths are integrated with RK4, sampling only ``v`` and ``w``; the accumulated map is never
    re-interpolated, so there is no numerical diffusion. With a constant ``w`` both modes equal
    ``exp(w·v)`` up to discretisation error.

    The Lagrangian map is only invertible if tissue does not collide, which needs
    ``∇w(ψ(x)) · v < 1`` along every path (roughly ``|∇w · v| < 1``). A sharp mask edge facing the
    flow direction violates this, so ``w`` must be smoothed (``weights.py`` guards this).
    """
    _check_field(v)
    weight = _as_weight(w, v)
    if steps == "auto":
        m = float((v * weight).detach().norm(dim=1).max())
        n = max(min_steps, math.ceil(m / MAX_RK_STEP_VOXELS))
    else:
        n = steps
    x = identity(v.shape[2:], device=v.device, dtype=v.dtype)

    def vel(y: Tensor) -> Tensor:
        return sample_at(v, y, padding_mode="border")

    if not (isinstance(weight, Tensor) and weight.ndim == v.ndim):
        return _path(x.expand_as(v), weight, vel, n) - x
    wmap = weight

    def at(field: Tensor, y: Tensor) -> Tensor:
        return sample_at(field, y, padding_mode="border")

    b = max(v.shape[0], wmap.shape[0])
    wmap, v = wmap.expand(b, *wmap.shape[1:]), v.expand(b, *v.shape[1:])
    if not lagrangian:
        wv = torch.cat([wmap, v], dim=1)

        def speed(y: Tensor) -> Tensor:
            s = at(wv, y)
            return s[:, :1] * s[:, 1:]

        return _rk4(x, speed, n) - x
    # Newton on g(τ) = τ - w(flow_v(x, τ)) = 0, with d flow_v(x, τ)/dτ = v(flow_v(x, τ)).
    # A stationary flow composes, flow_v(x, τ + Δ) = flow_v(flow_v(x, τ), Δ), so each update only
    # integrates the change Δ onward from the current point instead of the whole path again.
    d = v.shape[1]
    grad_w = torch.cat(torch.gradient(wmap, dim=tuple(range(2, v.ndim))), dim=1)
    packed = torch.cat([wmap, grad_w, v], dim=1)  # one lookup gives w, ∇w and v
    tau = wmap
    y = _path(x, tau, vel, n)
    for _ in range(newton_iters):
        s = at(packed, y)
        resid = tau - s[:, :1]
        if float(resid.abs().max()) < tol:
            break
        v_y = s[:, 1 + d :]
        slope = (s[:, 1 : 1 + d] * v_y).sum(1, keepdim=True)
        delta = -resid / (1.0 - slope).clamp_min(0.1)
        m = max(1, math.ceil(float((delta * v_y).norm(dim=1).max()) / MAX_RK_STEP_VOXELS))
        y = _path(y, delta, vel, m)
        tau = tau + delta
    return y - x


def invert(u: Tensor, *, iters: int = 100, tol: float = 1e-4) -> Tensor:
    """Inverse of a general displacement field by fixed-point iteration ``u⁻¹ = -u ∘ (id + u⁻¹)``.

    Converges when the map is a diffeomorphism with ``‖∇u‖ < 1``. For an SVF use ``exp(-v)``
    instead, which is exact.
    """
    _check_field(u)
    inv = -u
    for _ in range(iters):
        new = -warp(u, inv, padding_mode="border")
        done = float((new - inv).abs().max()) < tol
        inv = new
        if done:
            break
    return inv
