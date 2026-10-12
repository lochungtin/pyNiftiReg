import pytest
import torch

from helpers import DT, shape_for, smooth_field
from pyniftireg.grid import identity, sample_points, warp, warp_labels, warp_points


def test_identity_warp_is_exact(ndim: int) -> None:
    shape = shape_for(ndim)
    img = torch.rand(2, 3, *shape, dtype=DT)
    out = warp(img, torch.zeros(1, ndim, *shape, dtype=DT))
    assert torch.allclose(out, img, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_translation_moves_the_named_axis(ndim: int, axis: int) -> None:
    """Channel i of a field must act on array axis i (guards the grid_sample x/y/z flip)."""
    if axis >= ndim:
        pytest.skip("axis not present")
    shape = (6, 7, 8)[:ndim]
    img = torch.rand(1, 1, *shape, dtype=DT)
    u = torch.zeros(1, ndim, *shape, dtype=DT)
    u[:, axis] = 2.0
    out = warp(img, u)
    n = shape[axis]
    assert torch.allclose(out.narrow(2 + axis, 0, n - 2), img.narrow(2 + axis, 2, n - 2))
    assert out.narrow(2 + axis, n - 2, 2).abs().max() < 1e-9  # pulled from outside the FOV


def test_fractional_shift_is_linear(ndim: int) -> None:
    shape = shape_for(ndim)
    ramp = identity(shape, dtype=DT)[:, :1]  # value = index along axis 0
    u = torch.zeros(1, ndim, *shape, dtype=DT)
    u[:, 0] = 0.3
    out = warp(ramp, u)
    assert torch.allclose(out[:, :, :-1], ramp[:, :, :-1] + 0.3)


def test_labels_identity_and_shift(ndim: int) -> None:
    shape = shape_for(ndim)
    lab = torch.randint(0, 4, (1, 1, *shape))
    zero = torch.zeros(1, ndim, *shape, dtype=DT)
    for mode in ("soft", "nearest"):
        assert torch.equal(warp_labels(lab, zero, mode=mode), lab)
    u = zero.clone()
    u[:, 1] = -1.0
    out = warp_labels(lab, u)
    assert torch.equal(out[:, :, :, 1:], lab[:, :, :, :-1])
    assert torch.all(out[:, :, :, 0] == 0)
    assert out.dtype == lab.dtype


def test_soft_labels_keep_thin_structure_under_subvoxel_shift() -> None:
    lab = torch.zeros(1, 1, 16, 16, dtype=torch.long)
    lab[:, :, 8, :] = 1  # one-voxel-thick line
    u = torch.zeros(1, 2, 16, 16, dtype=DT)
    u[:, 0] = 0.4
    assert int((warp_labels(lab, u) == 1).sum()) == 16


def test_points_follow_field(ndim: int) -> None:
    v = smooth_field(ndim)
    pts = torch.rand(1, 20, ndim, dtype=DT) * 20 + 4
    vals = sample_points(v, pts)
    # sampling at integer points matches direct indexing
    ip = pts.round()
    idx = (0, slice(None), *ip[0].long().T)
    assert torch.allclose(sample_points(v, ip)[0], v[idx].T)
    assert torch.allclose(warp_points(pts, v), pts + vals)


def test_batch_broadcast(ndim: int) -> None:
    shape = shape_for(ndim)
    img = torch.rand(1, 1, *shape, dtype=DT)
    u = smooth_field(ndim, batch=3)
    out = warp(img, u)
    assert out.shape == (3, 1, *shape)
    assert torch.allclose(out[1], warp(img, u[1:2])[0])


def test_rejects_unsupported_dims() -> None:
    with pytest.raises(ValueError):
        warp(torch.rand(1, 1, 4), torch.zeros(1, 1, 4))
