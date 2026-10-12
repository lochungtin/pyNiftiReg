import numpy as np
import pytest
import torch

from helpers import DT, interior, oblique_affine, shape_for, smooth_field
from pyniftireg import preprocess as pp
from pyniftireg.grid import identity
from pyniftireg.space import ImageSpace


def _messy_space(ndim: int) -> ImageSpace:
    """Permuted, flipped, anisotropic, slightly oblique axes."""
    if ndim == 3:
        a = oblique_affine((1.2, 0.8, 2.0), (10.0, -5.0, 3.0))
        a[:, :3] = a[:, [2, 0, 1]]  # permute voxel axes
        return ImageSpace((12, 16, 20), a)
    a = np.array([[0.0, -1.5, 20.0], [0.9, 0.1, -4.0], [0.0, 0.0, 1.0]])
    return ImageSpace((14, 18), a)


def _world_image(space: ImageSpace) -> torch.Tensor:
    """Image whose channels are the world coordinates of each voxel."""
    x = identity(space.shape, dtype=DT).movedim(1, -1)
    return space.voxel_to_world(x).movedim(-1, 1)


def test_reorient_preserves_world_positions(ndim: int) -> None:
    space = _messy_space(ndim)
    img = _world_image(space)
    out, new = pp.reorient(img, space, "RAS")
    assert pp.orientation_codes(new) == "RAS"[:ndim]
    assert torch.allclose(out, _world_image(new), atol=1e-9)
    back, again = pp.reorient(out, new, pp.orientation_codes(space))
    assert torch.equal(back, img)
    assert np.allclose(again.affine, space.affine)


def test_reorient_rejects_bad_code() -> None:
    space = ImageSpace.from_spacing((4, 4, 4))
    with pytest.raises(ValueError):
        pp.reorient(torch.zeros(1, 1, 4, 4, 4), space, "RRS")


def test_resample_linear_function_is_exact(ndim: int) -> None:
    space = _messy_space(ndim)
    img = _world_image(space)
    out, new = pp.resample(img, space, 0.7)
    assert np.allclose(new.spacing, 0.7)
    assert np.allclose(new.origin, space.origin) and np.allclose(new.direction, space.direction)
    assert torch.allclose(out, _world_image(new), atol=1e-4)


def test_resample_to_with_affine(ndim: int) -> None:
    """``affine @ dst_world = src_world``: a world shift moves the content the other way."""
    space = ImageSpace.from_spacing(shape_for(ndim), 2.0)
    img = _world_image(space)
    shift = np.eye(ndim + 1)
    shift[0, ndim] = 4.0
    out = pp.resample_to(img, space, space, affine=shift)
    assert torch.allclose(interior(out - img, 4)[:, 0], torch.tensor(4.0, dtype=DT), atol=1e-4)


def test_resample_labels_nearest(ndim: int) -> None:
    space = ImageSpace.from_spacing(shape_for(ndim), 1.0)
    labels = torch.randint(0, 4, (1, 1, *shape_for(ndim)))
    out, _ = pp.resample(labels, space, 0.5, mode="nearest")
    assert out.dtype == labels.dtype
    assert set(out.unique().tolist()) <= set(labels.unique().tolist())
    idx = (slice(None), slice(None)) + (slice(None, None, 2),) * ndim
    assert torch.equal(out[idx], labels)


def test_resample_field_round_trip(ndim: int) -> None:
    affine = oblique_affine((1.0,) * 3, (0.0,) * 3) if ndim == 3 else np.eye(3)
    fine = ImageSpace(shape_for(ndim), affine)
    v = smooth_field(ndim, max_norm=3.0, sigma=5.0)
    coarse = pp.with_spacing(fine, 2.0)
    vc = pp.resample_field(v, fine, coarse)
    # the same physical vector: coarse voxel units are twice as large
    assert float(vc.norm(dim=1).max()) == pytest.approx(1.5, rel=0.1)
    back = pp.resample_field(vc, coarse, fine)
    assert interior(back - v, 4).abs().max() < 0.05


def test_normalize_percentile() -> None:
    x = torch.arange(101, dtype=torch.float32).view(1, 1, 101, 1)
    out = pp.normalize_percentile(x, 10, 90)
    assert float(out[0, 0, 10, 0]) == 0.0 and float(out[0, 0, 90, 0]) == 1.0
    assert float(out[0, 0, 50, 0]) == pytest.approx(0.5)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :, 50:] = True
    out = pp.normalize_percentile(x, 0, 100, mask=mask, clip=False)
    assert float(out[0, 0, 50, 0]) == 0.0 and float(out[0, 0, 0, 0]) == pytest.approx(-1.0)


def test_dilate_and_label_mask(ndim: int) -> None:
    space = ImageSpace.from_spacing((9,) * ndim, 2.0)
    m = torch.zeros(1, 1, *space.shape, dtype=torch.long)
    m[(0, 0) + (4,) * ndim] = 3
    assert int(pp.label_mask(m, (3,)).sum()) == 1 and int(pp.label_mask(m, (1,)).sum()) == 0
    assert int(pp.dilate(pp.label_mask(m), space, 4.0).sum()) == 5**ndim


def test_n4_flattens_a_bias_field() -> None:
    space = ImageSpace.from_spacing((40, 40, 40), 2.0)
    x = identity(space.shape, dtype=torch.float32) - 19.5
    ball = (x.norm(dim=1, keepdim=True) < 16).float()
    bias = torch.exp(0.4 * x[:, :1] / 20)
    img = (ball * 100 + 10) * bias
    out = pp.n4_bias_correction(img, space, mask=ball)
    inside = ball[0, 0] > 0

    def cv(t: torch.Tensor) -> float:
        v = t[0, 0][inside]
        return float(v.std() / v.mean())

    assert cv(out) < 0.5 * cv(img)


def test_preprocess_keeps_labels_aligned(ndim: int) -> None:
    space = _messy_space(ndim)
    labels = torch.zeros(1, 1, *space.shape, dtype=torch.long)
    labels[(slice(None), slice(None)) + (slice(3, 9),) * ndim] = 2
    image = labels.float() * 50 + 7
    img, lab, new = pp.preprocess(image, space, labels=labels, spacing=0.6)
    assert lab is not None and pp.orientation_codes(new) == "RAS"[:ndim]
    assert float(img.min()) == 0.0 and float(img.max()) == 1.0
    agree = ((img > 0.5) == (lab == 2)).float().mean()
    assert float(agree) > 0.97


def test_foreground_mask_ignores_noise_speckle(ndim: int) -> None:
    """A noisy background must not survive: dilation would grow every speckle into a blob."""
    shape = (64,) * ndim if ndim == 2 else (40, 40, 40)
    space = ImageSpace.from_spacing(shape, 1.0)
    x = identity(shape) - torch.tensor([(n - 1) / 2 for n in shape]).view(1, -1, *[1] * ndim)
    body = (x.norm(dim=1, keepdim=True) < shape[0] / 4).float()
    g = torch.Generator().manual_seed(0)
    noisy = 0.6 * body + 0.08 * torch.rand(1, 1, *shape, generator=g)
    noisy[(slice(None), slice(None)) + (slice(2, 3),) * ndim] = 0.9  # one bright speckle
    assert pp.otsu_threshold(noisy) == pytest.approx(0.35, abs=0.15)
    m = pp.foreground_mask(noisy, space)
    assert float((m == (body > 0)).float().mean()) > 0.98
    assert not bool(m[(0, 0) + (2,) * ndim])
    assert torch.equal(
        pp.erode(pp.dilate(body > 0, space, 2.0), space, 2.0) | (body > 0),
        pp.erode(pp.dilate(body > 0, space, 2.0), space, 2.0),
    )
