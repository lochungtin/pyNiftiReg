import pytest
import torch

from helpers import DT, interior, rotation_svf, shape_for, smooth_field
from pyniftireg import flow
from pyniftireg.filters import gaussian_smooth
from pyniftireg.grid import identity, warp
from pyniftireg.jacobian import det

M = 8  # interior margin, keeps border padding effects out of comparisons


def test_exp_zero_is_identity(ndim: int) -> None:
    v = torch.zeros(1, ndim, *shape_for(ndim), dtype=DT)
    assert torch.equal(flow.exp(v), v)


def test_exp_translation_is_exact(ndim: int) -> None:
    v = torch.zeros(1, ndim, *shape_for(ndim), dtype=DT)
    v[:, 0], v[:, -1] = 3.7, -1.2
    assert torch.allclose(flow.exp(v, 2.0), 2 * v, atol=1e-10)


def test_exp_rotation(ndim: int) -> None:
    shape = shape_for(ndim)
    angle = 0.5
    u = flow.exp(rotation_svf(shape, angle))
    x = identity(shape, dtype=DT)
    c = torch.tensor([(n - 1) / 2 for n in shape], dtype=DT).view(1, -1, *[1] * ndim)
    r = x - c
    expected = torch.zeros_like(x)
    cos, sin = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    expected[:, 0] = cos * r[:, 0] - sin * r[:, 1] - r[:, 0]
    expected[:, 1] = sin * r[:, 0] + cos * r[:, 1] - r[:, 1]
    near = (r[:, :2].norm(dim=1, keepdim=True) < 10).expand_as(u)
    assert (u - expected)[near].abs().max() < 0.05


@pytest.mark.parametrize(("s", "t"), [(0.5, 0.7), (-0.4, 1.5), (2.0, 1.0)])
def test_group_law(ndim: int, s: float, t: float) -> None:
    v = smooth_field(ndim)
    lhs = flow.compose(flow.exp(v, s), flow.exp(v, t))
    rhs = flow.exp(v, s + t)
    assert (interior(lhs - rhs, M)).abs().max() < 0.05


def test_exp_inverse(ndim: int) -> None:
    v = smooth_field(ndim)
    assert interior(flow.compose(flow.exp(v), flow.exp(v, -1.0)), M).abs().max() < 0.05
    assert interior(flow.compose(flow.exp(v, -1.0), flow.exp(v)), M).abs().max() < 0.05


@pytest.mark.parametrize("t", [-3.0, 3.0, 5.0])
def test_extrapolation_stays_fold_free(ndim: int, t: float) -> None:
    v = smooth_field(ndim, max_norm=2.0)
    u = flow.exp(v, t)
    assert float(det(u).min()) > 0


def test_auto_steps_grow_with_t(ndim: int) -> None:
    v = smooth_field(ndim, max_norm=1.0)
    assert flow.auto_steps(0.25 * v) == 0
    assert flow.auto_steps(v) == 1
    assert flow.auto_steps(8 * v) == 4


def test_per_sample_t(ndim: int) -> None:
    v = smooth_field(ndim, batch=2)
    u = flow.exp(v, torch.tensor([0.5, -1.0]), steps=6)
    assert torch.allclose(u[0:1], flow.exp(v[0:1], 0.5, steps=6))
    assert torch.allclose(u[1:2], flow.exp(v[1:2], -1.0, steps=6))


def test_invert_general_displacement(ndim: int) -> None:
    u = flow.exp(smooth_field(ndim, max_norm=2.0))
    inv = flow.invert(u)
    assert interior(flow.compose(u, inv), M).abs().max() < 1e-3
    assert interior(inv - flow.exp(smooth_field(ndim, max_norm=2.0), -1.0), M).abs().max() < 0.05


def test_integrate_constant_weight_matches_exp(ndim: int) -> None:
    v = smooth_field(ndim)
    ref = flow.exp(v, 0.8)
    w = torch.full((1, 1, *shape_for(ndim)), 0.8, dtype=DT)
    for lag in (True, False):
        assert interior(flow.integrate(v, w, lagrangian=lag) - ref, M).abs().max() < 0.05


def test_eulerian_integrate_matches_exp_of_weighted_field(ndim: int) -> None:
    v = smooth_field(ndim)
    w = smooth_field(ndim, seed=1, max_norm=1.0)[:, :1] + 0.5
    euler = flow.integrate(v, w, lagrangian=False)
    assert interior(euler - flow.exp(v, w), M).abs().max() < 0.05


def test_lagrangian_vs_eulerian_analytic(ndim: int) -> None:
    """Translation v = a·e0 with weight w(p) = alpha·p0.

    Lagrangian: psi0 = x0 + a·w(psi(x))  =>  psi0 = x0 / (1 - a·alpha).
    Eulerian:   dy/ds = a·alpha·y        =>  y = x0·exp(a·alpha).
    """
    a, alpha = 4.0, 0.05
    shape = (40,) + (8,) * (ndim - 1)
    v = torch.zeros(1, ndim, *shape, dtype=DT)
    v[:, 0] = a
    x0 = identity(shape, dtype=DT)[:, :1]
    w = alpha * x0
    lag = flow.integrate(v, w)[:, 0]
    eul = flow.integrate(v, w, lagrangian=False)[:, 0]
    sel = (x0[:, 0] >= 2) & (x0[:, 0] <= 24)
    x = x0[:, 0][sel]
    assert (lag[sel] - (x / (1 - a * alpha) - x)).abs().max() < 1e-3
    assert (eul[sel] - (x * torch.exp(torch.tensor(a * alpha, dtype=DT)) - x)).abs().max() < 0.02


def _smooth_bump() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (64, 8)
    v = torch.zeros(1, 2, *shape, dtype=DT)
    v[:, 0] = -6.0  # pull-back x -> x - 6: content moves +6 voxels
    w = torch.zeros(1, 1, *shape, dtype=DT)
    w[:, :, 24:40] = 1.0
    return v, gaussian_smooth(w, (3.0, 0.0))  # max |6·∂w| ≈ 0.8 < 1: tissue cannot collide


def test_lagrangian_satisfies_defining_identity() -> None:
    """psi(x) = x - 6·w(psi(x)) for the translation v = -6·e0."""
    v, w = _smooth_bump()
    psi = flow.integrate(v, w)
    resid = psi[:, :1] + 6 * warp(w, psi, padding_mode="border")
    assert resid.abs().max() < 1e-3
    assert float(det(psi).min()) > 0
    # source centre 32 (w≈1) is fetched from output ~38; far background stays put
    assert abs(float(psi[0, 0, 38, 4]) + 6.0) < 0.1
    assert psi[0, 0, 2:8].abs().max() < 1e-3


def test_eulerian_does_not_follow_tissue() -> None:
    v, w = _smooth_bump()
    eul = flow.integrate(v, w, lagrangian=False)
    resid = eul[:, :1] + 6 * warp(w, eul, padding_mode="border")
    assert resid.abs().max() > 0.5  # w stayed in output space instead
    # scaling and squaring converges to the RK4 path integral with more squarings
    assert (eul - flow.exp(v, w, steps=9))[:, :, 4:-4].abs().max() < 0.05


def test_rejects_bad_weight_shape(ndim: int) -> None:
    v = smooth_field(ndim)
    with pytest.raises(ValueError):
        flow.exp(v, torch.ones(1, 2, *shape_for(ndim)))


@pytest.mark.parametrize("lagrangian", [True, False])
def test_integrate_broadcasts_weight_batch(ndim: int, lagrangian: bool) -> None:
    v = smooth_field(ndim)
    w = torch.stack([torch.full(shape_for(ndim), t, dtype=DT) for t in (0.5, -1.0)])[:, None]
    out = flow.integrate(v, w, lagrangian=lagrangian)
    assert out.shape == (2, ndim, *shape_for(ndim))
    for i, t in enumerate((0.5, -1.0)):
        assert interior(out[i : i + 1] - flow.exp(v, t), M).abs().max() < 0.05
