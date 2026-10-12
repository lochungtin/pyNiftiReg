# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

pyNiftiReg (`pyniftireg`) is a PyTorch reimplementation of NiftyReg-style registration (`reg_aladin` affine, then `reg_f3d -vel` B-spline stationary velocity field), extracted from the sibling project `../drift_aug` (`src/drift/registration/torchreg.py` and the modules it depends on). It is NiftyReg-*style*, not a line-by-line port, and needs no NiftyReg binaries.

## Commands

- `uv sync --extra dev`: install (pytest, ruff, mypy, plus the `[io]` nibabel and `[n4]` SimpleITK extras)
- `uv run pytest`: all tests; `uv run pytest tests/test_torchreg.py::test_square_to_circle`: a single test
- `uv run ruff check src tests`, `uv run ruff format src tests`, `uv run mypy` (strict, `src` only)
- `uv build`: wheel + sdist into `dist/`
- **Don't run tests (or anything else that uses the GPU) without the user's go-ahead**: the GPU is often busy with other jobs. `CUDA_VISIBLE_DEVICES=` forces CPU. Don't run several GPU processes at once: PyTorch's caching allocator keeps memory after a process's GPU work ends.

## Architecture

`registration.py` holds the whole algorithm in `TorchReg`; the other modules are general field and image utilities it is built on.

`TorchReg.register` accepts NIfTI paths or numpy/torch arrays (`result.ImageLike`) for images and masks. It normalises them into `(1, 1, *spatial)` tensors plus `ImageSpace`s (`_as_image` / `_as_mask`; arrays without a space get 1 mm identity spacing) and delegates to `register_arrays`, which does the work. Keep the `Registration` protocol in `result.py` in sync with `register`'s signature.

1. **Intensity normalisation**: 0.5/99.5 percentiles (within the masks) are mapped to [0, 1].
2. **Affine** (`TorchReg._affine`):
   - Adam on the similarity over an image pyramid, started from aligning the centres of mass, with cosine LR decay (without the decay the result jitters by one Adam step).
   - Parametrised as `L (x - c) + c + t` around A's centre of mass. With `rigid_only`, `L` is the exponential of a skew-symmetric matrix.
   - Multi-start at the coarsest level, keeping the better similarity: identity scale, and the masks' per-axis spread ratio (moment matching). Identity alone failed a pair with a large size difference; moment matching alone was misled by arms in thorax masks.
3. **Resampling**: B is resampled onto A's grid with that affine (`preprocess.resample_to`).
4. **Non-rigid** (`_velocity` / `_optimise_level`):
   - The cubic B-spline control grid lives on **A's voxel grid**, not NiftyReg's world-aligned grid. Control-point spacing doubles per coarser level, and `_refine` re-expresses it exactly between levels (nested knots, `bspline.from_dense`).
   - Loss: symmetric similarity (`B ∘ exp(v)` vs A and `A ∘ exp(-v)` vs B) minus regularisation. NMI is evaluated on a random voxel subset (`max_samples`).
   - Bending energy and linear elasticity are evaluated at the control points with 3-tap B-spline derivative stencils, in units of the spacing, like NiftyReg's `-be`/`-le`. There is an optional log-Jacobian penalty.
   - Gradients come from autograd through scaling and squaring; early stop after 20 non-improving iterations.
5. **Result** (`result.RegistrationResult`): `affine`, a dense `velocity` `(1, D, *A.shape)`, `jacobian` stats and `qc`.

### Regularisation weights

The regularisation weights are normalised differently from `reg_f3d`'s. `NIFTYREG_WEIGHT_SCALE` / `from_niftyreg_weights` (be ×5, le ×5, jl ×50) convert NiftyReg weights.

- These multipliers were calibrated in drift_aug on 24 labelled Duke breast MRI pairs and 20 TotalSegmentator thorax pairs (Dice 0.924 vs NiftyReg 0.905 on breast).
- ×10/×100 was too stiff on the thorax; ×3/×30 gave det J up to ~20.
- Don't change them without re-validating on real data.

### NMI values

`nmi` is a Parzen (cubic B-spline window) joint histogram built with `index_add`. It peaks below 2 for identical images (≈1.55 at 64 bins), so `qc["nmi"]` is not comparable with NiftyReg's `reg_measure`.

### GPU memory

GPU memory grows ~0.3 GiB per million voxels (autograd through every squaring step). For full-resolution MRI, register on resampled images (`preprocess.resample` / `with_spacing`). Then move the velocity back onto the original grid with `preprocess.resample_field`.

## Conventions (likely sources of silent bugs)

- **Fields are `(B, D, *spatial)` in voxel units**; `D = v.shape[1]`. All code is dimension-agnostic, and 3D is primary. Math tests take the `ndim` fixture (`tests/conftest.py`) to run in both 2D and 3D. `tests/helpers.py` has synthetic fields and an `interior` crop: compare fields away from the border, because `compose` and `warp` use border padding.
- Field channel `i` is the component along array axis `i`, as is voxel axis `i` of an `ImageSpace` affine (as in a NIfTI sform). `grid_sample`'s coordinate conversion and `align_corners=True` live only in `grid.py` (`to_grid_sample` flips the axes to x/y/z order).
- **Pull-back warps**: `I_warped(x) = I(φ(x))`. A displacement `u` means `φ = id + u`, and `compose(u1, u2)` is `φ1∘φ2`.
- **Sign conventions**:
  - Affine: `Affine @ A_world = B_world`.
  - `exp(v)` pulls `B_aligned` back onto A (`warp(B_aligned, exp(v)) ≈ A`), so A's content moved onto B is `warp(A, exp(-v))`.
- Exponentiation is scaling and squaring, with the step count chosen so each step is < 0.5 voxel (`flow.auto_steps`).
- mm, orientation and spacing are `ImageSpace` metadata; they only matter at NIfTI I/O, at the affine stage, and in the NiftyReg spacing convention. That convention (`cp_spacing`, `lncc_sigma`) reads positive values as mm and negative values as voxels.
- `TorchReg.thread_safe = False`: `torch.linalg`'s lazy initialisation isn't thread-safe, so run registrations sequentially.
- NIfTI I/O uses nibabel's best affine (sform, else qform), the same priority as NiftyReg. 2D images are stored with a singleton third axis.
