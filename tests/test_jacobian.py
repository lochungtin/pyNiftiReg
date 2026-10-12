import torch

from helpers import DT, rotation_svf, shape_for, smooth_field
from pyniftireg.flow import exp
from pyniftireg.jacobian import det, jacobian_matrix, jacobian_stats


def test_identity_and_affine(ndim: int) -> None:
    shape = shape_for(ndim)
    assert torch.allclose(
        det(torch.zeros(1, ndim, *shape, dtype=DT)), torch.ones(1, *shape, dtype=DT)
    )
    from pyniftireg.grid import identity

    x = identity(shape, dtype=DT)
    scale = torch.tensor([1.5, 0.5, 2.0][:ndim], dtype=DT).view(1, -1, *[1] * ndim)
    u = (scale - 1) * x  # φ(x) = diag(scale)·x
    assert torch.allclose(det(u), torch.full_like(u[:, 0], float(scale.prod())))
    j = jacobian_matrix(u)
    assert j.shape == (1, *shape, ndim, ndim)
    assert torch.allclose(j[(0,) + (3,) * ndim], torch.diag(scale.flatten()))


def test_rotation_preserves_volume(ndim: int) -> None:
    u = exp(rotation_svf(shape_for(ndim), 0.4))
    dj = det(u)[(slice(None),) + (slice(8, -8),) * ndim]
    assert (dj - 1).abs().max() < 0.01


def test_stats_and_folds() -> None:
    u = torch.zeros(2, 2, 10, 10, dtype=DT)
    u[1, 0] = -2 * torch.arange(10, dtype=DT).view(10, 1)  # φ0 = -x0 -> det = -1 everywhere
    s = jacobian_stats(u)
    assert torch.allclose(s.min, torch.tensor([1.0, -1.0], dtype=DT))
    assert torch.allclose(s.fold_fraction, torch.tensor([0.0, 1.0], dtype=DT))
    mask = torch.zeros(1, 1, 10, 10, dtype=torch.bool)
    mask[..., :5, :] = True
    s = jacobian_stats(u, mask)
    assert torch.allclose(s.mean, torch.tensor([1.0, -1.0], dtype=DT))


def test_det_matches_linalg(ndim: int) -> None:
    u = smooth_field(ndim, max_norm=4.0)
    j = jacobian_matrix(u)
    assert torch.allclose(det(u), torch.linalg.det(j), atol=1e-10)
