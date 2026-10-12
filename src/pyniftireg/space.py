"""Image geometry: grid shape plus a voxel-index -> world (mm) affine.

Everything in the core math works in voxel units; ``ImageSpace`` is the metadata needed to convert
to millimetres at NIfTI I/O and NiftyReg boundaries, or to express QC limits in mm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class ImageSpace:
    """Grid ``shape`` and a ``(D+1, D+1)`` homogeneous ``affine`` mapping voxel indices to world mm.

    Voxel axis ``i`` is array axis ``i`` (the same order as field channels), as in a NIfTI sform.
    """

    shape: tuple[int, ...]
    affine: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        shape = tuple(int(n) for n in self.shape)
        affine = np.array(self.affine, dtype=np.float64)
        d = len(shape)
        if affine.shape != (d + 1, d + 1):
            raise ValueError(
                f"affine must be {(d + 1, d + 1)} for a {d}-D shape, got {affine.shape}"
            )
        if not np.allclose(affine[-1], np.eye(d + 1)[-1]):
            raise ValueError("affine must be homogeneous (last row [0, ..., 0, 1])")
        if abs(np.linalg.det(affine[:d, :d])) < 1e-12:
            raise ValueError("affine linear part is singular")
        affine.setflags(write=False)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "affine", affine)

    @classmethod
    def from_spacing(
        cls,
        shape: Sequence[int],
        spacing: float | Sequence[float] = 1.0,
        origin: float | Sequence[float] = 0.0,
    ) -> ImageSpace:
        """Axis-aligned space with the given voxel ``spacing`` and world ``origin`` of voxel 0."""
        d = len(shape)
        affine = np.eye(d + 1)
        affine[:d, :d] = np.diag(np.broadcast_to(np.asarray(spacing, dtype=np.float64), (d,)))
        affine[:d, d] = np.broadcast_to(np.asarray(origin, dtype=np.float64), (d,))
        return cls(tuple(shape), affine)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> npt.NDArray[np.float64]:
        """Voxel size in mm along each voxel axis (column norms of the linear part)."""
        return np.asarray(np.linalg.norm(self.affine[: self.ndim, : self.ndim], axis=0))

    @property
    def direction(self) -> npt.NDArray[np.float64]:
        """Unit direction cosines of each voxel axis, as columns."""
        d: npt.NDArray[np.float64] = self.affine[: self.ndim, : self.ndim] / self.spacing
        return d

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        o: npt.NDArray[np.float64] = self.affine[: self.ndim, self.ndim].copy()
        return o

    def _mats(self, like: Tensor, inverse: bool) -> tuple[Tensor, Tensor]:
        a = np.linalg.inv(self.affine) if inverse else self.affine.copy()
        d = self.ndim
        lin = torch.as_tensor(a[:d, :d], dtype=like.dtype, device=like.device)
        off = torch.as_tensor(a[:d, d], dtype=like.dtype, device=like.device)
        return lin, off

    def voxel_to_world(self, points: Tensor) -> Tensor:
        """Map voxel coordinates ``(..., D)`` to world mm ``(..., D)``."""
        lin, off = self._mats(points, inverse=False)
        return points @ lin.T + off

    def world_to_voxel(self, points: Tensor) -> Tensor:
        """Map world mm ``(..., D)`` to voxel coordinates ``(..., D)``."""
        lin, off = self._mats(points, inverse=True)
        return points @ lin.T + off

    def displacement_to_world(self, disp: Tensor) -> Tensor:
        """Convert a voxel-unit field ``(B, D, *spatial)`` to world-mm vectors (linear part)."""
        lin, _ = self._mats(disp, inverse=False)
        return torch.einsum("ij,bj...->bi...", lin, disp)

    def displacement_to_voxel(self, disp: Tensor) -> Tensor:
        """Convert a world-mm field ``(B, D, *spatial)`` to voxel-unit vectors."""
        lin, _ = self._mats(disp, inverse=True)
        return torch.einsum("ij,bj...->bi...", lin, disp)
