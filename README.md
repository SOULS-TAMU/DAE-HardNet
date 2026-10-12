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
