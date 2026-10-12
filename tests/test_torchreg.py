"""PyTorch registration backend: losses, regularisers, and recovery of known transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import interior, shapes_pair
from pyniftireg import bspline
from pyniftireg.filters import gaussian_smooth
from pyniftireg.flow import exp
from pyniftireg.grid import identity, warp
from pyniftireg.preprocess import resample_to
from pyniftireg.registration import (
    TorchReg,
    TorchRegParams,
    _refine,
    bending_energy,
    linear_elasticity,
    lncc,
    nmi,
)
from pyniftireg.space import ImageSpace


def dice(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a != 0, b != 0
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * int((a & b).sum()) / denom


FAST = TorchRegParams(levels=2, max_iter=80, affine_levels=2, affine_iters=60)


def _texture(shape: tuple[int, ...], seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return gaussian_smooth(torch.rand(1, 1, *shape, generator=g), 1.5)


def test_similarity_measures() -> None:
    a = _texture((32, 32))
    b = _texture((32, 32), seed=2)
    a01 = (a - a.min()) / (a.max() - a.min())
    b01 = (b - b.min()) / (b.max() - b.min())
    assert float(nmi(a01, a01)) > float(nmi(a01, b01)) + 0.2
    # NMI doesn't care about the contrast mapping
    assert float(nmi(a01, 1 - a01)) == pytest.approx(float(nmi(a01, a01)), abs=0.02)
    assert float(lncc(a01, a01, None, 3.0)) == pytest.approx(1.0, abs=1e-3)
    assert float(lncc(a01, 2 * a01 + 0.3, None, 3.0)) == pytest.approx(1.0, abs=1e-3)
    mask = torch.zeros_like(a01, dtype=torch.bool)
    mask[..., 8:24, 8:24] = True
    x = a01.clone().requires_grad_(True)
    nmi(x, b01, mask).backward()
    assert x.grad is not None and float(x.grad[~mask].abs().max()) == 0.0
    assert float(x.grad[mask].abs().max()) > 0


def test_regularisers(ndim: int) -> None:
    shape = (9,) * ndim
    sp = [2.0] * ndim
    x = identity(shape, dtype=torch.float64)
    linear = 0.3 * x.flip(1)  # affine velocity: no bending
    assert float(bending_energy(linear, sp)) == pytest.approx(0.0, abs=1e-12)
    rot = torch.zeros_like(x)
    rot[:, 0], rot[:, 1] = -0.2 * x[:, 1], 0.2 * x[:, 0]  # antisymmetric gradient
    assert float(linear_elasticity(rot, sp)) == pytest.approx(0.0, abs=1e-12)
    bent = 0.05 * x**2
    assert float(bending_energy(bent, sp)) > 0
    assert float(linear_elasticity(bent, sp)) > 0


def test_refine_is_exact(ndim: int) -> None:
    shape = (37,) * ndim if ndim == 2 else (21, 18, 25)
    g = torch.Generator().manual_seed(0)
    coarse = [8.0] * ndim
    fine = [4.0] * ndim
    n_cp = [bspline.num_control_points(n, s) for n, s in zip(shape, coarse, strict=True)]
    cp = torch.randn(1, ndim, *n_cp, generator=g, dtype=torch.float64)
    refined = _refine(cp, coarse, fine, shape)
    assert torch.allclose(
        bspline.to_dense(refined, shape, fine), bspline.to_dense(cp, shape, coarse), atol=1e-8
    )


def test_recovers_known_velocity(ndim: int) -> None:
    shape = (48, 48) if ndim == 2 else (32, 32, 32)
    tex = _texture(shape)
    g = torch.Generator().manual_seed(3)
    v_true = gaussian_smooth(torch.randn(1, ndim, *shape, generator=g), 5.0)
    v_true = v_true * (2.5 / v_true.norm(dim=1).max())
    moving = warp(tex, exp(-v_true), padding_mode="border")  # moving ∘ exp(v_true) ≈ tex
    space = ImageSpace.from_spacing(shape, 1.0)
    params = TorchRegParams(cp_spacing=-4.0, levels=2, max_iter=150)
    r = TorchReg(params).register_arrays(tex, space, moving, space, affine=np.eye(ndim + 1))
    err = interior(r.velocity - v_true, 6).norm(dim=1)
    assert float(err.mean()) < 0.15 and float(err.max()) < 0.8
    assert float(r.jacobian.min) > 0
    assert r.qc["nmi"] > r.qc["nmi_affine"]


def test_affine_sign_and_recovery(tmp_path: Path) -> None:
    """``Affine * A_world = B_world``: B is A's content shifted by +3 mm, rotated and scaled."""
    shape = (40, 40, 40)
    space = ImageSpace.from_spacing(shape, 1.5, origin=(-10.0, 5.0, 0.0))
    a = _texture(shape, seed=5) * gaussian_smooth(
        (identity(shape) - 19.5).norm(dim=1, keepdim=True).lt(15).float(), 2.0
    )
    true = np.eye(4)
    ang = np.deg2rad(6.0)
    true[:2, :2] = 1.05 * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    centre = space.voxel_to_world(torch.full((1, 3), 19.5, dtype=torch.float64))[0].numpy()
    true[:3, 3] = centre - true[:3, :3] @ centre + np.array([3.0, 0.0, 0.0])
    # B(world) = A(true^-1 world): sampling B at true @ x gives A(x)
    b = resample_to(a, space, space, affine=np.linalg.inv(true))
    r = TorchReg(TorchRegParams(levels=1, max_iter=1)).register_arrays(a, space, b, space)
    # mapping error over the textured ball (radius 15 voxels = 22.5 mm)
    pts = centre + 22.5 * np.random.default_rng(0).uniform(-0.6, 0.6, (500, 3))
    hom = np.c_[pts, np.ones(len(pts))]
    err = np.linalg.norm((hom @ (r.affine - true).T)[:, :3], axis=1)
    assert err.max() < 0.5, (err.max(), r.affine)
    rigid = TorchReg(TorchRegParams(rigid_only=True, levels=1, max_iter=1, affine_levels=2))
    shift = np.eye(4)
    shift[0, 3] = 3.0
    shifted = resample_to(a, space, space, affine=np.linalg.inv(shift))
    rr = rigid.register_arrays(a, space, shifted, space)
    assert np.allclose(rr.affine[:3, 3], [3.0, 0.0, 0.0], atol=0.2)
    assert np.allclose(rr.affine[:3, :3], np.eye(3), atol=0.02)


@pytest.mark.parametrize("similarity", ["ssd", "nmi"])
def test_square_to_circle(similarity: str) -> None:
    n = 64
    sq, ci = shapes_pair(2, n, square=13.5, circle=15.5)
    space = ImageSpace.from_spacing((n, n), 1.5)
    params = TorchRegParams(
        cp_spacing=-4.0, similarity=similarity, bending_energy=0.0005, linear_elasticity=0.0
    )
    r = TorchReg(params).register_arrays(
        gaussian_smooth(sq, 1.5), space, gaussian_smooth(ci, 1.5), space
    )
    b_aligned = resample_to(ci, space, space, affine=r.affine) > 0.5
    moved = warp(sq, exp(-r.velocity.double())) > 0.5
    assert dice(moved, b_aligned) > 0.95
    assert dice(moved, b_aligned) > dice(sq > 0.5, b_aligned) + 0.02
    assert float(r.jacobian.min) > 0


def test_register_from_files_with_masks(tmp_path: Path) -> None:
    pytest.importorskip("nibabel")
    from pyniftireg.io import save_nifti

    shape = (40, 40)
    space = ImageSpace.from_spacing(shape, 2.0)
    a = _texture(shape)
    b = torch.roll(a, 2, dims=2)
    mask = torch.zeros(1, 1, *shape)
    mask[..., 5:35, 5:35] = 1
    for name, img in (("a", a), ("b", b), ("m", mask)):
        save_nifti(tmp_path / f"{name}.nii.gz", img, space)
    r = TorchReg(FAST).register(
        tmp_path / "a.nii.gz",
        tmp_path / "b.nii.gz",
        reference_mask=tmp_path / "m.nii.gz",
        floating_mask=tmp_path / "m.nii.gz",
    )
    assert r.velocity.shape == (1, 2, *shape)
    assert r.affine[0, 2] == pytest.approx(4.0, abs=0.5)  # 2 voxels of 2 mm along axis 0


def test_register_accepts_arrays() -> None:
    """numpy and torch inputs, with and without spaces, match ``register_arrays``."""
    shape = (40, 40)
    space = ImageSpace.from_spacing(shape, 2.0)
    a = _texture(shape)
    b = torch.roll(a, 2, dims=2)
    mask = torch.zeros(1, 1, *shape)
    mask[..., 5:35, 5:35] = 1
    expected = TorchReg(FAST).register_arrays(
        a, space, b, space, reference_mask=mask, floating_mask=mask
    )
    r_np = TorchReg(FAST).register(
        a[0, 0].numpy(),
        b[0, 0].numpy().astype(np.float64),
        reference_mask=mask[0, 0].numpy() > 0,
        floating_mask=mask.numpy(),
        reference_space=space,
        floating_space=space,
    )
    r_t = TorchReg(FAST).register(a, b[0, 0], reference_mask=mask, floating_mask=mask)
    for r in (r_np, r_t):
        assert r.velocity.shape == (1, 2, *shape)
    assert np.allclose(r_np.affine, expected.affine, atol=1e-4)
    assert torch.allclose(r_np.velocity, expected.velocity, atol=1e-4)
    # default space: 1 mm voxels, so the 2-voxel shift is 2 mm
    assert r_t.space.shape == shape and np.allclose(r_t.space.spacing, 1.0)
    assert r_t.affine[0, 2] == pytest.approx(2.0, abs=0.3)


def test_register_rejects_bad_inputs(tmp_path: Path) -> None:
    space = ImageSpace.from_spacing((16, 16), 1.0)
    img = torch.zeros(16, 16)
    reg = TorchReg(FAST)
    with pytest.raises(ValueError, match="shape"):
        reg.register(img, torch.zeros(2, 16, 16, 16))
    with pytest.raises(ValueError, match="space"):
        reg.register(img, img, reference_space=ImageSpace.from_spacing((8, 8), 1.0))
    with pytest.raises(ValueError, match="reference_mask"):
        reg.register(img, img, reference_mask=torch.zeros(8, 8))
    with pytest.raises(ValueError, match="arrays only"):
        reg.register(tmp_path / "a.nii.gz", img, reference_space=space)
