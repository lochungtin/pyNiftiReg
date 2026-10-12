from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import oblique_affine
from pyniftireg.space import ImageSpace

pytest.importorskip("nibabel")
from pyniftireg.io import load_nifti, save_nifti


def test_roundtrip_2d(tmp_path: Path) -> None:
    sp = ImageSpace.from_spacing((7, 5), spacing=(1.5, 2.0), origin=(10.0, -3.0))
    img = torch.rand(1, 1, 7, 5)
    save_nifti(tmp_path / "a.nii.gz", img, sp)
    out, sp2 = load_nifti(tmp_path / "a.nii.gz")
    assert torch.equal(out, img)
    assert sp2.ndim == 2 and np.allclose(sp2.affine, sp.affine)


def test_roundtrip_3d_oblique_multichannel(tmp_path: Path) -> None:
    sp = ImageSpace((6, 5, 4), oblique_affine((1.2, -1.0, 2.0), (5.0, 6.0, 7.0)))
    img = torch.rand(1, 3, 6, 5, 4, dtype=torch.float64)
    save_nifti(tmp_path / "a.nii", img, sp)
    out, sp2 = load_nifti(tmp_path / "a.nii", dtype=torch.float64)
    assert torch.allclose(out, img)
    assert np.allclose(sp2.affine, sp.affine, atol=1e-6)


def test_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_nifti(tmp_path / "a.nii", torch.zeros(1, 1, 3, 3), ImageSpace.from_spacing((4, 4)))
