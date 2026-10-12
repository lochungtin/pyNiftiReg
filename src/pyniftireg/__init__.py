"""pyNiftiReg: a PyTorch reimplementation of NiftyReg-style registration.

Affine (``reg_aladin``-like) pre-alignment followed by a cubic B-spline stationary velocity field
(``reg_f3d -vel``-like), on CPU or GPU.
"""

from importlib.metadata import PackageNotFoundError, version

from pyniftireg.registration import (
    NIFTYREG_WEIGHT_SCALE,
    TorchReg,
    TorchRegParams,
    from_niftyreg_weights,
)
from pyniftireg.result import ImageLike, Registration, RegistrationResult
from pyniftireg.space import ImageSpace

try:
    __version__ = version("pyniftireg")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "NIFTYREG_WEIGHT_SCALE",
    "ImageLike",
    "ImageSpace",
    "Registration",
    "RegistrationResult",
    "TorchReg",
    "TorchRegParams",
    "__version__",
    "from_niftyreg_weights",
]
