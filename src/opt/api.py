"""Explicit backend selection; historical defaults remain unchanged."""
import jax.numpy as jnp
from .layers import LinearEqualityProjection, SparseProjection, FischerBurmeisterProjection

METHODS = ("linear", "sparse", "fb", "fischer-burmeister-newton", "fischer-burmeister-reduced", "primal-dual-interior-point")

def registered_methods():
    return METHODS

def _shared_problem(problem):
    from kkthn.problems.functional import FunctionalProblem
    canonical = problem.canonicalize() if hasattr(problem, "canonicalize") else problem
    return FunctionalProblem(parameter_size=canonical.parameter_size,
        variable_size=canonical.variable_size, objective=canonical.objective,
        equalities=canonical.equalities, inequalities=canonical.inequalities,
        equality_size=canonical.equality_size, inequality_size=canonical.inequality_size,
        equality_sparsity=canonical.equality_sparsity,
        inequality_sparsity=canonical.inequality_sparsity,
        name="daehn_shared_projection")

def prepare_projection(problem, *, method="fb", config=None, batch_size, dtype=jnp.float64, backend="cpu"):
    """Prepare DAEHN constraints using an explicitly selected backend.

    Shared backends accept DAEHN FunctionalProblem/canonical definitions or
    native KKTHN problems. Their configuration types belong to KKTHN.
    Legacy names retain the original DAEHN algorithms and configurations.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if method == "linear":
        return problem.prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    if method == "sparse":
        from .config import ProjectionConfig
        return SparseProjection(problem, config or ProjectionConfig()).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    if method == "fb":
        if config is not None:
            raise TypeError("Configure the legacy FischerBurmeisterProjection directly, or select a shared backend")
        return FischerBurmeisterProjection(problem).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    from kkthn.opt.api import prepare_projection as shared_prepare
    native = problem if type(problem).__module__.startswith("kkthn.") else _shared_problem(problem)
    return shared_prepare(native, method=method, config=config, batch_size=batch_size, dtype=dtype, backend=backend)
