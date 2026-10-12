import numpy as np
import pytest
import torch

from pyniftireg.space import ImageSpace

DT = torch.float64


def oblique(ndim: int) -> ImageSpace:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(ndim, ndim)))
    a = np.eye(ndim + 1)
    a[:ndim, :ndim] = q @ np.diag(np.linspace(0.7, 2.5, ndim))
    a[:ndim, ndim] = rng.normal(size=ndim) * 50
    return ImageSpace((10,) * ndim, a)


def test_from_spacing(ndim: int) -> None:
    sp = ImageSpace.from_spacing((4,) * ndim, spacing=[2.0, 3.0, 4.0][:ndim], origin=5.0)
    assert np.allclose(sp.spacing, [2.0, 3.0, 4.0][:ndim])
    assert np.allclose(sp.origin, 5.0)
    assert np.allclose(sp.direction, np.eye(ndim))
    p = torch.ones(1, ndim, dtype=DT)
    assert torch.allclose(sp.voxel_to_world(p), torch.tensor([[7.0, 8.0, 9.0][:ndim]], dtype=DT))


def test_roundtrip_oblique(ndim: int) -> None:
    sp = oblique(ndim)
    assert np.allclose(sp.spacing, np.linspace(0.7, 2.5, ndim))
    p = torch.rand(5, 7, ndim, dtype=DT) * 10
    assert torch.allclose(sp.world_to_voxel(sp.voxel_to_world(p)), p)
    u = torch.randn(2, ndim, *(4,) * ndim, dtype=DT)
    assert torch.allclose(sp.displacement_to_voxel(sp.displacement_to_world(u)), u)
    # vector conversion agrees with differencing point conversions
    pts = torch.rand(ndim, dtype=DT) * 5
    vec = torch.randn(ndim, dtype=DT)
    expected = sp.voxel_to_world(pts + vec) - sp.voxel_to_world(pts)
    u1 = vec.view(1, ndim, *[1] * ndim)
    assert torch.allclose(sp.displacement_to_world(u1).flatten(), expected)


def test_immutable_and_validated() -> None:
    sp = ImageSpace.from_spacing((4, 4))
    with pytest.raises(ValueError):
        sp.affine[0, 0] = 2.0
    with pytest.raises(ValueError):
        ImageSpace((4, 4), np.eye(4))
    with pytest.raises(ValueError):
        ImageSpace((4, 4), np.zeros((3, 3)))
