"""Registration result and the protocol a registration backend implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from torch import Tensor

from pyniftireg.jacobian import JacobianStats
from pyniftireg.space import ImageSpace

#: An image or mask: a NIfTI file, or an array of shape ``(*spatial)`` or ``(1, 1, *spatial)``.
ImageLike = str | Path | npt.NDArray[Any] | Tensor


@dataclass(frozen=True)
class RegistrationResult:
    """Affine pre-alignment plus a non-rigid SVF, both relative to the reference image A.

    - ``affine``: ``(D+1, D+1)`` world-mm matrix mapping reference world points to floating world
      points (NiftyReg's ``Affine * Reference = Floating``).
    - ``velocity``: ``(1, D, *space.shape)`` SVF in voxel units on the reference grid.
      ``exp(velocity)`` is the pull-back map from the reference grid onto the floating image
      after affine alignment: ``B_aligned ∘ exp(v) ≈ A``, and so ``A ∘ exp(-v) ≈ B_aligned``.
    - ``qc``: Jacobian statistics of ``exp(velocity)`` plus backend-specific scalars.
    """

    space: ImageSpace
    affine: npt.NDArray[np.float64]
    velocity: Tensor
    jacobian: JacobianStats
    qc: dict[str, float] = field(default_factory=dict)


class Registration(Protocol):
    def register(
        self,
        reference: ImageLike,
        floating: ImageLike,
        *,
        reference_mask: ImageLike | None = None,
        floating_mask: ImageLike | None = None,
        reference_space: ImageSpace | None = None,
        floating_space: ImageSpace | None = None,
        affine: npt.NDArray[np.float64] | None = None,
    ) -> RegistrationResult:
        """Register ``floating`` (B) to ``reference`` (A): affine, then a non-rigid SVF.

        Images and masks are NIfTI paths or numpy/torch arrays. An array's grid is
        ``reference_space`` / ``floating_space`` (default: 1 mm voxels, identity orientation); a
        file carries its own.

        ``affine``: skip the affine step and use this one (``affine @ A_world = B_world``), e.g. to
        register several compartments relative to one shared affine.
        """
        ...
