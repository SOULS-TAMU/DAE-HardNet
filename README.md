# DAEHN — Differentiable DAE projection in JAX

[![Native CPU tests](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/ci.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/ci.yml?query=branch%3Ajax-library-ci)
[![Quality & coverage](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/quality.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/quality.yml?query=branch%3Ajax-library-ci)
[![CodeQL](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/codeql.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/codeql.yml?query=branch%3Ajax-library-ci)
[![Native dependency audit](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/dependency-audit.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/dependency-audit.yml?query=branch%3Ajax-library-ci)
[![Documentation](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/docs.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/docs.yml?query=branch%3Ajax-library-ci)
[![Release preflight](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/release.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/release.yml?query=branch%3Ajax-library-ci)
[![KKTHN integration](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/integration.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/integration.yml?query=branch%3Ajax-library-ci)
[![Full installation](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/package.yml/badge.svg?branch=jax-library-ci)](https://github.com/SOULS-TAMU/DAE-HardNet/actions/workflows/package.yml?query=branch%3Ajax-library-ci)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Status scope:** badges track the `jax-library-ci` review branch. Native checks are separate from KKTHN integration and full installation, which remain blocked by the unavailable compatible dependency. See [quality-check scope](docs/quality.md).

DAEHN is a Python/JAX library for DAE-HardNet formulation, discretization,
neural-network training, and differentiable optimization/projection layers.

The library source lives directly in `src/`. Setuptools maps that directory
onto the Python package `daehn`, so imports remain `import daehn`.

## Install from GitHub

This library is distributed through GitHub source, with no PyPI publishing workflow.
Use the revision containing this pyproject.toml and src/ layout; older upstream
research revisions have different installation instructions.

```bash
git clone --branch jax-library-ci https://github.com/SOULS-TAMU/DAE-HardNet.git daehn
cd daehn
python -m venv .venv
```

Activate `.venv` with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell,
or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install --upgrade pip
python -m pip install .
python -m pip check
python -c "import daehn; print(daehn.__version__)"
```

KKTHN is a required dependency: distribution `kkt-hardnet>=0.8.0,<0.9`, Python
import `kkthn`. If it is unavailable from your package index, install a compatible
KKTHN source checkout first with `python -m pip install ../kkthn` (replace the
path as needed). No dependency source URL is guessed here.

For development: `python -m pip install -e ".[dev,docs]"`.
The current metadata declares Python >=3.10; the initial CPU CI targets Python
3.11/3.12/3.13 on Linux, Windows and macOS. The preceding native-test campaign passed
all nine combinations; badges report the current revision. See [installation](docs/installation.md).

## Quick example

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from daehn.opt.linear_equality import LinearEqualityProjection

project = LinearEqualityProjection(A=np.array([[1., 1.]]), b=np.array([1.])).prepare(batch_size=1)
print(project(jnp.array([2., 0.])))  # [1.5, -0.5]
```

See [documentation](docs/index.md),
[contributing](CONTRIBUTING.md), [changelog](CHANGELOG.md), [security](SECURITY.md)
and [citation](CITATION.cff). Build docs locally with `python -m mkdocs serve`.

The repository workflows test CPU behavior, source installation and documentation.
The optional release workflow creates GitHub artifacts only. It never uploads to PyPI.

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

The `kkt-hardnet` distribution (KKTHN 0.8.x) is required for the shared interfaces. It must be installed before installing DAEHN if it is not available from your package index.

The Git tag `pre-kkthn-modularization` preserves the full library before this reorganization. To inspect it use `git show pre-kkthn-modularization:src/opt/layers/projection.py`; use a separate checkout of that tag when comparing old runs.

### Reorganization verification (2026-10-11)

Compared the saved DAEHN reference runs for Lotka-Volterra and CSTR against the pre-reorganization code. Predictions, full-model derivatives, differential/algebraic residuals, and solver residuals matched exactly (maximum absolute difference 0). Three projected optimizer updates from the same saved weights and batches also matched exactly, including all parameter leaves and training metrics. This is a short regression check, not a complete retraining. No benchmark result files were overwritten.

The three imported backends passed an analytical two-variable equality projection and finite-gradient check; full/reduced FB maximum errors were approximately 3.75e-10, and interior-point error was 0. Shared sparse matrix-vector multiplication and compatibility imports passed. The eight relocated solver definitions have identical parsed Python code to the Git snapshot.

All three shared backends also passed an equality-plus-nonnegativity check with converged statuses and finite gradients. Full/reduced FB agreed with the boundary optimum within 5.5e-11; the interior-point solver retained the expected finite-barrier offset of approximately 5e-5 at its default mu=1e-4. A compiled CSTR preparation/call with each new backend also returned converged status. These integration checks do not establish full benchmark equivalence when switching algorithms.

The CPU matrix runs native-library tests independently of KKTHN. A separate shared-backend job and the full installation job require the compatible kkt-hardnet 0.8 distribution or a configured KKTHN_SOURCE. Native-only installation uses --no-deps after installing other declared dependencies and is a CI isolation technique, not the recommended user installation. Until the compatible shared package is available, those dependency jobs remain blocked.
