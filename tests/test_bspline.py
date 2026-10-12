import pytest
import torch

from helpers import DT
from pyniftireg import bspline
from pyniftireg.grid import identity


def test_partition_of_unity_and_linear_reproduction() -> None:
    for spacing in (1, 3, 5, 2.5):
        w = bspline.weight_matrix(37, spacing, dtype=DT)
        assert torch.allclose(w.sum(1), torch.ones(37, dtype=DT))
        knots = (torch.arange(w.shape[1], dtype=DT) - 1) * spacing
        assert torch.allclose(w @ knots, torch.arange(37, dtype=DT))


def test_quadratic_reproduction() -> None:
    """Uniform cubic B-splines reproduce x² with control values t_k² - s²/3."""
    s = 4.0
    w = bspline.weight_matrix(30, s, dtype=DT)
    t = (torch.arange(w.shape[1], dtype=DT) - 1) * s
    assert torch.allclose(w @ (t**2 - s**2 / 3), torch.arange(30, dtype=DT) ** 2)


@pytest.mark.parametrize("spacing", [4, 5.0])
def test_cubic_polynomial_round_trip(ndim: int, spacing: float) -> None:
    shape = (33, 29, 21)[:ndim]
    x = identity(shape, dtype=DT) / 10
    field = torch.cat([x[:, :1] ** 3 - x[:, -1:] * x[:, :1], 2 * x[:, -1:] ** 2 + 1], dim=1)
    cp = bspline.from_dense(field, spacing)
    assert cp.shape[2:] == tuple(bspline.num_control_points(n, spacing) for n in shape)
    dense = bspline.to_dense(cp, shape, spacing)
    assert torch.allclose(dense, field, atol=1e-8)


def test_anisotropic_spacing_and_gradients() -> None:
    n_cp = (bspline.num_control_points(20, 3), bspline.num_control_points(13, 2.5))
    assert n_cp == (10, 8)
    cp = torch.randn(1, 2, *n_cp, dtype=DT, requires_grad=True)
    dense = bspline.to_dense(cp, (20, 13), (3, 2.5))
    assert dense.shape == (1, 2, 20, 13)
    dense.sum().backward()
    assert cp.grad is not None and torch.isfinite(cp.grad).all()


def test_too_few_control_points() -> None:
    with pytest.raises(ValueError):
        bspline.to_dense(torch.zeros(1, 1, 3, 3), (20, 20), 5)


def test_evaluate_matches_to_dense(ndim: int) -> None:
    shape = (17, 13, 11)[:ndim]
    spacing = (4.0, 3.0, 2.5)[:ndim]
    n_cp = [bspline.num_control_points(n, s) for n, s in zip(shape, spacing, strict=True)]
    cp = torch.randn(2, 3, *n_cp, dtype=DT)
    dense = bspline.to_dense(cp, shape, spacing)
    s = torch.tensor(spacing, dtype=DT).view(1, -1, *[1] * ndim)
    coords = identity(shape, dtype=DT) / s + 1  # voxel x sits at control index x/s + 1
    assert torch.allclose(bspline.evaluate(cp, coords), dense)
