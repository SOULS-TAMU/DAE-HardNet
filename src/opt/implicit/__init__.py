"""NumPy and JAX implicit-differentiation implementations."""
from . import numpy, jax
from .jax import (
    AdjointSolveResult,
    ImplicitVJPResult,
    MultiAdjointSolveResult,
    build_reduced_adjoint_rhs,
    compute_full_adjoint_residual,
    implicit_parameter_vjp,
    make_implicit_differentiable_solver,
    primal_output_cotangent,
    recover_bound_adjoint_blocks,
    solve_implicit_adjoint,
    solve_implicit_adjoint_multi_rhs,
    state_from_flat,
)

__all__ = [
    "numpy",
    "jax",
    "AdjointSolveResult",
    "ImplicitVJPResult",
    "MultiAdjointSolveResult",
    "build_reduced_adjoint_rhs",
    "compute_full_adjoint_residual",
    "implicit_parameter_vjp",
    "make_implicit_differentiable_solver",
    "primal_output_cotangent",
    "recover_bound_adjoint_blocks",
    "solve_implicit_adjoint",
    "solve_implicit_adjoint_multi_rhs",
    "state_from_flat",
]
