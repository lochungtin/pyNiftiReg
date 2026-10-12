# pyNiftiReg

A PyTorch reimplementation of [NiftyReg](https://github.com/KCL-BMEIS/niftyreg)-style image
registration, extracted from the `drift-aug` project. It runs on CPU or GPU and needs no NiftyReg
binaries.

For a reference image A and a floating image B it computes:

1. An **affine** (or rigid) transform in world mm, using NiftyReg's convention
   `Affine * A_world = B_world`, like `reg_aladin`. It maximises the similarity with Adam over an
   image pyramid, starting from aligned centres of mass. At the coarsest level it tries two starts:
   identity scale and moment-matched scale.
2. A **stationary velocity field** (SVF) on A's voxel grid, like `reg_f3d -vel`. The field is
   parametrised by a cubic B-spline control-point grid. The loss is the (symmetric) NMI, LNCC or SSD
   between `B_aligned ∘ exp(v)` and A, plus NiftyReg-style bending energy, linear elasticity and an
   optional log-Jacobian penalty. The control-point spacing doubles at each coarser pyramid level and
   is refined exactly between levels.

It is NiftyReg-*style*, not a line-by-line port. The regularisation weights are normalised
differently from `reg_f3d`'s. To reuse `-be`/`-le`/`-jl` values, convert them with
`from_niftyreg_weights` (multipliers ×5/×5/×50, calibrated on real breast and thorax MRI).
NMI values in `qc` come from this package's Parzen estimator and cannot be compared with
`reg_measure`.

## Install

```bash
pip install -e .            # numpy + torch
pip install -e '.[io]'      # + nibabel, for NIfTI files
pip install -e '.[n4]'      # + SimpleITK, for N4 bias correction in pyniftireg.preprocess
uv sync --extra dev         # development: pytest, ruff, mypy, all extras
```

## Usage

```python
from pyniftireg import TorchReg, TorchRegParams, from_niftyreg_weights
from pyniftireg.flow import exp
from pyniftireg.grid import warp
from pyniftireg.io import load_nifti, save_nifti
from pyniftireg.preprocess import resample_to

params = TorchRegParams(
    cp_spacing=-5.0,  # NiftyReg convention: mm if positive, voxels if negative
    similarity="nmi",
    **from_niftyreg_weights(bending_energy=0.001, linear_elasticity=0.01, jacobian_log=0.0),
)
reg = TorchReg(params)
result = reg.register("A.nii.gz", "B.nii.gz", reference_mask="A_mask.nii.gz")

result.affine      # (D+1, D+1) world-mm matrix, Affine @ A_world = B_world
result.velocity    # (1, D, *A.shape) SVF in A's voxel units
result.jacobian    # det J statistics of exp(velocity)
result.qc          # NMI before/after the non-rigid step, timings

# Warp B onto A
a, sa = load_nifti("A.nii.gz")
b, sb = load_nifti("B.nii.gz")
b_aligned = resample_to(b, sb, sa, affine=result.affine)
b_on_a = warp(b_aligned, exp(result.velocity))
save_nifti("B_on_A.nii.gz", b_on_a, sa)
```

`register` also takes numpy arrays or torch tensors of shape `(*spatial)` or `(1, 1, *spatial)`,
for the images and the masks, and paths and arrays can be mixed. Give an array's grid with
`reference_space=` / `floating_space=` (an `ImageSpace`). Without one, the array is taken to have
1 mm voxels and identity orientation:

```python
import numpy as np
from pyniftireg import ImageSpace

a = np.load("A.npy")  # (H, W, D)
b = np.load("B.npy")
result = reg.register(a, b, reference_space=ImageSpace.from_spacing(a.shape, (0.8, 0.8, 2.0)))
```

Pass `affine=` to skip the affine step.

## Modules

| module | contents |
| --- | --- |
| `registration` | `TorchReg`, `TorchRegParams`, NMI / LNCC / SSD, bending energy, linear elasticity |
| `result` | `RegistrationResult` and the `Registration` protocol |
| `space` | `ImageSpace`: grid shape plus voxel → world affine |
| `bspline` | cubic B-spline control grids: `to_dense`, `from_dense`, `evaluate` |
| `flow` | SVF exponential (scaling and squaring), `compose`, `integrate`, `invert` |
| `grid` | identity grids, `warp`, `warp_labels`, `sample_at`, point warping |
| `jacobian` | Jacobian matrices, determinants, `jacobian_stats` |
| `preprocess` | reorientation, resampling between grids, percentile normalisation, N4, masks |
| `filters` | separable Gaussian smoothing |
| `io` | NIfTI load/save (`[io]` extra) |

Conventions: fields are `(B, D, *spatial)` in voxel units. Voxel axis `i` is array axis `i`, as in
a NIfTI sform. A displacement `u` stands for the map `id + u`.

## Tests

```bash
uv run pytest
```
