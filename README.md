# DAEHN

DAEHN is a Python/JAX library for DAE-HardNet formulation, discretization,
neural-network training, and differentiable optimization/projection layers.

The library source lives directly in `src/`. Setuptools maps that directory
onto the Python package `daehn`, so imports remain `import daehn`.

## Install

Requires Python 3.10 or later.

```powershell
& D:\Projects\virtual_envs\sciml\env\Scripts\python.exe -m pip install -e D:\Projects\daehn --no-deps --no-build-isolation
```

For an environment without dependencies installed, use `python -m pip install -e .`.

## Source layout

- `src/dae`: continuous systems, discretization, projected models and training.
- `src/nn`: functional neural-network layers.
- `src/opt`: projection layers, solvers and implicit differentiation.
- `src/problems`: symbolic, functional and structured optimization problems.
- `src/losses` and `src/train`: losses and training utilities.
- `src/serialization` and `src/export`: predictor storage and deployment artifacts.
- `src/runtime` and `src/tree`: runtime and PyTree helpers.

Case-study scripts, datasets, saved models, results, analysis, and the manuscript
audit live in the separate sibling project `../daehn-benchmarks`.

```python
import daehn
from daehn.dae import DAESystem, TaylorNeighborhood, compile_dae
from daehn.opt.layers.projection import SparseProjection, LinearEqualityProjection
```


## Module organization and shared KKTHN backends

DAE problem creation, compilation, discretization, conditions, and model reconstruction remain in `daehn.dae`; optimization problem frontends remain in `daehn.problems`. Neural networks, training, losses, runtime, and persistence retain their existing APIs.

Projection implementations now live in `daehn.opt.linear_equality`, `daehn.opt.fischer_burmeister`, and `daehn.opt.primal_dual_interior_point`. Existing imports from `daehn.opt.layers` remain compatible. The historical `fb`, `sparse`, `linear`, and `auto` selections retain their original numerical implementations and defaults.

The explicit methods `fischer-burmeister-newton`, `fischer-burmeister-reduced`, and `primal-dual-interior-point` use the installed KKTHN package. Select them through `compiled.prepare_projection(method=..., batch_size=...)` for a compiled DAE, or `daehn.opt.api.prepare_projection` for a DAEHN functional/canonical problem. Native KKTHN problem definitions are also accepted by the latter. DAEHN canonical functions and sparsity are adapted through KKTHN's FunctionalProblem compiler; problem construction in DAEHN is unchanged. Selecting a new algorithm can change numerical results and is opt-in.

Use KKTHN's `FischerBurmeisterConfig` or `ProjectionConfig` for full control of the shared backends. When a DAEHN ProjectionConfig is provided, common tolerances, linear-solver settings, and gradient policy are transferred; shared FB-specific settings use KKTHN defaults. Shared sparse assembly, operators, MINRES, conversion, and types are imported through `daehn.utils.sparse`. Historical DAEHN sparse solver kernels remain available for existing results.

KKTHN 0.8.x is required for the shared interfaces. It must be installed before installing DAEHN if it is not available from your package index.

The Git tag `pre-kkthn-modularization` preserves the full library before this reorganization. To inspect it use `git show pre-kkthn-modularization:src/opt/layers/projection.py`; use a separate checkout of that tag when comparing old runs.

### Reorganization verification (2026-10-11)

Compared the saved DAEHN reference runs for Lotka-Volterra and CSTR against the pre-reorganization code. Predictions, full-model derivatives, differential/algebraic residuals, and solver residuals matched exactly (maximum absolute difference 0). Three projected optimizer updates from the same saved weights and batches also matched exactly, including all parameter leaves and training metrics. This is a short regression check, not a complete retraining. No benchmark result files were overwritten.

The three imported backends passed an analytical two-variable equality projection and finite-gradient check; full/reduced FB maximum errors were approximately 3.75e-10, and interior-point error was 0. Shared sparse matrix-vector multiplication and compatibility imports passed. The eight relocated solver definitions have identical parsed Python code to the Git snapshot.

All three shared backends also passed an equality-plus-nonnegativity check with converged statuses and finite gradients. Full/reduced FB agreed with the boundary optimum within 5.5e-11; the interior-point solver retained the expected finite-barrier offset of approximately 5e-5 at its default mu=1e-4. A compiled CSTR preparation/call with each new backend also returned converged status. These integration checks do not establish full benchmark equivalence when switching algorithms.
