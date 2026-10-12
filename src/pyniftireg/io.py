"""NIfTI I/O (requires the ``io`` extra: nibabel).

Images load as ``(1, C, *spatial)`` tensors plus an :class:`ImageSpace`. 2D images are stored as
NIfTI with a singleton third axis; they load back as 2D when that axis has size 1 and is not mixed
with the in-plane axes. The world affine is nibabel's best affine (sform if set, else qform),
which is the same priority NiftyReg uses.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from pyniftireg.space import ImageSpace

PathLike = str | Path


def require_nibabel() -> ModuleType:
    try:
        import nibabel
    except ImportError as e:  # pragma: no cover - depends on installed extras
        raise ImportError("NIfTI I/O needs nibabel: pip install 'pyniftireg[io]'") from e
    return nibabel


def space_from_nifti_affine(affine: npt.ArrayLike, shape: tuple[int, ...]) -> ImageSpace:
    """Build an ImageSpace from a 4x4 NIfTI affine, reducing to 2D when ``shape`` is 2D."""
    a = np.asarray(affine, dtype=np.float64)
    if len(shape) == 3:
        return ImageSpace(shape, a)
    if len(shape) != 2:
        raise ValueError(f"only 2D and 3D images are supported, got shape {shape}")
    if not (np.allclose(a[:2, 2], 0) and np.allclose(a[2, :2], 0)):
        raise ValueError("2D image whose affine mixes the third axis into the plane")
    a2 = np.eye(3)
    a2[:2, :2] = a[:2, :2]
    a2[:2, 2] = a[:2, 3]
    return ImageSpace(shape, a2)


def nifti_affine(space: ImageSpace) -> npt.NDArray[np.float64]:
    """4x4 NIfTI affine for ``space`` (2D spaces get a unit third axis)."""
    if space.ndim == 3:
        return np.array(space.affine)
    a = np.eye(4)
    a[:2, :2] = space.affine[:2, :2]
    a[:2, 3] = space.affine[:2, 2]
    return a


def load_nifti(path: PathLike, *, dtype: torch.dtype = torch.float32) -> tuple[Tensor, ImageSpace]:
    """Load a scalar or multi-channel image as ``(1, C, *spatial)`` and its space.

    Channels come from NIfTI dims 4 and 5 (time / vector components).
    """
    img = require_nibabel().load(str(path))
    data = np.asarray(img.dataobj)
    spatial = tuple(int(n) for n in img.shape[:3]) if data.ndim >= 3 else tuple(img.shape)
    if len(spatial) == 3 and spatial[2] == 1:
        spatial = spatial[:2]
    data = data.reshape(*img.shape[:3], -1) if data.ndim >= 3 else data[..., None]
    data = data.reshape(*spatial, -1)
    t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(data, -1, 0))).to(dtype)[None]
    return t, space_from_nifti_affine(img.affine, spatial)


def save_nifti(path: PathLike, image: Tensor, space: ImageSpace) -> None:
    """Save ``(1, C, *spatial)`` or ``(*spatial)``; C > 1 goes to NIfTI dim 4."""
    nib = require_nibabel()
    x = image.detach().cpu()
    if x.ndim == space.ndim + 2:
        if x.shape[0] != 1:
            raise ValueError("can only save a single sample")
        x = x[0]
    elif x.ndim == space.ndim:
        x = x[None]
    if tuple(x.shape[1:]) != space.shape:
        raise ValueError(f"image shape {tuple(x.shape[1:])} does not match space {space.shape}")
    arr = np.moveaxis(x.numpy(), 0, -1)
    if space.ndim == 2:
        arr = arr[:, :, None, :]
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    img = nib.Nifti1Image(arr, nifti_affine(space))
    img.set_sform(nifti_affine(space), code=1)
    img.set_qform(nifti_affine(space), code=1)
    nib.save(img, str(path))
