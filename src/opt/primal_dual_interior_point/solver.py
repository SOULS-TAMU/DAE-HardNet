"""Differentiable projection layers.

``SparseProjection`` wraps the unchanged sparse interior-point implementation.
The explicit linear-equality and Fischer--Burmeister layers are independent
alternatives and do not alter the existing sparse solver path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    import scipy.linalg as scipy_linalg
    import scipy.sparse as scipy_sparse
except Exception:  # pragma: no cover - SciPy is a declared dependency
    scipy_linalg = None
    scipy_sparse = None

from ..complementarity import fischer_burmeister
from ..config import ProjectionConfig
from ..semismooth import SemismoothNewtonConfig, semismooth_newton
from ..solvers.iteration import FixedMuOptions
from ..sparse.batch_iteration import (
    BATCH_STATUS_CONVERGED,
    BATCH_STATUS_MAXIMUM_ITERATIONS,
    BATCH_STATUS_NUMERICAL_FAILURE,
)
from ...problems.problem import Problem as CoreProblem
from ...train.projected_neural import make_implicit_projection_layer


FB_STATUS_CONVERGED = BATCH_STATUS_CONVERGED
FB_STATUS_MAX_ITERATIONS = BATCH_STATUS_MAXIMUM_ITERATIONS
FB_STATUS_NONFINITE = BATCH_STATUS_NUMERICAL_FAILURE


from ..types import ProjectionInfo

@dataclass(frozen=True, eq=False)
class PreparedProjection:
    _layer: Any
    problem: Any
    config: ProjectionConfig
    batch_size: int
    dtype: Any
    backend: str

    def __call__(self, y_hat, parameters=None):
        if parameters is None:
            parameters = jnp.empty(
                (self.batch_size, self._layer.physical_parameter_count),
                dtype=self.dtype,
            )
        return self._layer.project(parameters, y_hat)

    def with_info(self, y_hat, parameters=None):
        if parameters is None:
            parameters = jnp.empty(
                (self.batch_size, self._layer.physical_parameter_count),
                dtype=self.dtype,
            )
        out = self._layer.project_with_info(parameters, y_hat)
        return out.y_tilde, ProjectionInfo(
            out.forward_status_codes,
            out.forward_iterations,
            out.forward_residual_inf_norm,
            out.forward_state_finite,
            out.forward_gradient_eligible,
        )


@dataclass(frozen=True)
class SparseProjection:
    """The unchanged sparse interior-point projection exposed as a layer."""

    problem: Any
    config: ProjectionConfig = ProjectionConfig()

    def prepare(
        self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"
    ) -> PreparedProjection:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        problem = (
            self.problem.as_projection_problem()
            if hasattr(self.problem, "as_projection_problem")
            else self.problem
        )
        if not isinstance(problem, CoreProblem):
            if hasattr(problem, "to_symbolic_problem"):
                problem = problem.to_symbolic_problem()
            else:
                raise TypeError(
                    "problem must be a symbolic problem or provide "
                    "as_projection_problem()"
                )
        parameter_count = len(problem.parameter_names)
        cfg = self.config
        linear = cfg.linear_solver
        implicit = cfg.implicit_gradient
        prepared = problem.prepare_sparse_projection_batch_solver(
            parameters=np.zeros(
                (batch_size, parameter_count), dtype=np.float64
            ),
            batch_size=batch_size,
            options=FixedMuOptions(
                max_iterations=cfg.max_iterations,
                residual_tolerance=cfg.tolerance,
                line_search=None,
            ),
            interior_margin=cfg.interior_margin,
            gmres_tolerance=linear.tolerance,
            gmres_atol=linear.atol,
            gmres_restart=linear.restart,
            gmres_maxiter=linear.max_iterations,
            use_preconditioner=linear.use_preconditioner,
            backtracking=cfg.backtracking,
            max_backtracking_steps=cfg.max_backtracking_steps,
            backtracking_factor=cfg.backtracking_factor,
            backend=backend,
            transpose_mode=cfg.transpose_mode,
            store_residual_history=cfg.store_residual_history,
            adjoint_gmres_tolerance=implicit.adjoint_tolerance,
            adjoint_gmres_atol=linear.atol,
            adjoint_gmres_restart=linear.restart,
            adjoint_gmres_maxiter=linear.max_iterations,
            adjoint_solver_mode=linear.mode,
        )
        layer = make_implicit_projection_layer(
            prepared,
            mu=cfg.mu,
            differentiate_problem_parameters=(
                implicit.differentiate_problem_parameters
            ),
            gradient_policy=implicit.policy,
            inexact_gradient_tolerance=implicit.inexact_tolerance,
            adjoint_gradient_tolerance=implicit.adjoint_tolerance,
        )
        return PreparedProjection(
            layer, problem, cfg, batch_size, dtype, str(backend)
        )


