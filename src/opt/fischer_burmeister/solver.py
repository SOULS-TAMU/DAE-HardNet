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

class PreparedFischerBurmeisterProjection:
    def __init__(
        self,
        *,
        problem,
        batch_size: int,
        dtype,
        backend: str,
        tolerance: float,
        max_iterations: int,
        regularization: float,
        smoothing: float,
        line_search: bool,
        max_line_search_steps: int,
        line_search_factor: float,
        gradient_policy: str,
        inexact_gradient_tolerance: float,
        adjoint_gradient_tolerance: float,
        differentiate_problem_parameters: bool,
    ):
        canonical = (
            problem.canonicalize()
            if hasattr(problem, "canonicalize")
            else problem
        )
        required = (
            "parameter_size",
            "variable_size",
            "equality_size",
            "inequality_size",
            "equalities",
            "inequalities",
        )
        missing = [name for name in required if not hasattr(canonical, name)]
        if missing:
            raise TypeError(
                "FischerBurmeisterProjection requires a canonicalizable "
                f"functional problem; missing {missing}"
            )
        if gradient_policy not in {"strict", "residual_checked", "always"}:
            raise ValueError(
                "gradient_policy must be strict, residual_checked, or always"
            )
        self.problem = problem
        self.canonical = canonical
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.backend = str(backend)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.regularization = float(regularization)
        self.smoothing = float(smoothing)
        self.gradient_policy = gradient_policy
        self.inexact_gradient_tolerance = float(inexact_gradient_tolerance)
        self.adjoint_gradient_tolerance = float(adjoint_gradient_tolerance)
        self.differentiate_problem_parameters = bool(differentiate_problem_parameters)
        self.parameter_size = int(canonical.parameter_size)
        self.output_size = int(canonical.variable_size)
        self.equality_size = int(canonical.equality_size)
        self.inequality_size = int(canonical.inequality_size)
        self._newton_config = SemismoothNewtonConfig(
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            regularization=self.regularization,
            line_search=bool(line_search),
            max_line_search_steps=int(max_line_search_steps),
            line_search_factor=float(line_search_factor),
        )
        self._project_with_info = self._build_project_with_info()

    def _residual(self, z, y_hat, theta):
        n = self.output_size
        neq = self.equality_size
        y = z[:n]
        equality_multipliers = z[n : n + neq]
        inequality_multipliers = z[n + neq :]
        equalities = jnp.asarray(
            self.canonical.equalities(theta, y), dtype=self.dtype
        ).reshape((neq,))
        inequalities = jnp.asarray(
            self.canonical.inequalities(theta, y), dtype=self.dtype
        ).reshape((self.inequality_size,))
        if neq:
            equality_jacobian = jax.jacfwd(
                self.canonical.equalities, argnums=1
            )(theta, y)
        else:
            equality_jacobian = jnp.zeros((0, n), dtype=self.dtype)
        if self.inequality_size:
            inequality_jacobian = jax.jacfwd(
                self.canonical.inequalities, argnums=1
            )(theta, y)
        else:
            inequality_jacobian = jnp.zeros((0, n), dtype=self.dtype)
        stationarity = (
            y
            - y_hat
            + equality_jacobian.T @ equality_multipliers
            + inequality_jacobian.T @ inequality_multipliers
        )
        complementarity = fischer_burmeister(
            inequality_multipliers,
            -inequalities,
            smoothing=self.smoothing,
        )
        return jnp.concatenate(
            (stationarity, equalities, complementarity), axis=0
        )

    def _solve_one(self, y_hat, theta):
        initial = jnp.concatenate(
            (
                y_hat,
                jnp.zeros((self.equality_size,), dtype=self.dtype),
                jnp.ones((self.inequality_size,), dtype=self.dtype),
            )
        )
        z, iterations, residual, finite = semismooth_newton(
            self._residual,
            initial,
            y_hat,
            theta,
            config=self._newton_config,
        )
        converged = finite & (residual <= self.tolerance)
        status = jnp.where(
            ~finite,
            FB_STATUS_NONFINITE,
            jnp.where(
                converged,
                FB_STATUS_CONVERGED,
                FB_STATUS_MAX_ITERATIONS,
            ),
        ).astype(jnp.int32)
        if self.gradient_policy == "strict":
            eligible = converged
        elif self.gradient_policy == "residual_checked":
            eligible = finite & (
                converged | (residual <= self.inexact_gradient_tolerance)
            )
        else:
            eligible = finite
        return z, status, iterations, residual, finite, eligible

    def _build_project_with_info(self):
        solve_batch = jax.vmap(self._solve_one, in_axes=(0, 0))

        @jax.custom_vjp
        def project_with_info(y_hat, theta):
            z, status, iterations, residual, finite, eligible = solve_batch(
                y_hat, theta
            )
            return (
                z[:, : self.output_size],
                status,
                iterations,
                residual,
                finite,
                eligible,
            )

        def forward(y_hat, theta):
            z, status, iterations, residual, finite, eligible = solve_batch(
                y_hat, theta
            )
            output = (
                z[:, : self.output_size],
                status,
                iterations,
                residual,
                finite,
                eligible,
            )
            return output, (z, y_hat, theta, eligible)

        def backward(saved, cotangents):
            z, y_hat, theta, eligible = saved
            cotangent_y = cotangents[0]

            def one_backward(z_i, yhat_i, theta_i, cot_i, eligible_i):
                jac_z = jax.jacfwd(self._residual, argnums=0)(
                    z_i, yhat_i, theta_i
                )
                rhs = jnp.concatenate(
                    (
                        cot_i,
                        jnp.zeros(
                            (
                                self.equality_size
                                + self.inequality_size,
                            ),
                            dtype=cot_i.dtype,
                        ),
                    )
                )
                system = jac_z.T + self.regularization * jnp.eye(
                    jac_z.shape[0], dtype=jac_z.dtype
                )
                adjoint = jnp.linalg.solve(system, rhs)
                adjoint = jnp.nan_to_num(
                    adjoint, nan=0.0, posinf=0.0, neginf=0.0
                )
                adjoint_residual = jnp.max(
                    jnp.abs(jac_z.T @ adjoint - rhs)
                )
                adjoint_finite = jnp.all(jnp.isfinite(adjoint))
                if self.gradient_policy == "always":
                    adjoint_valid = adjoint_finite
                else:
                    adjoint_valid = adjoint_finite & (
                        adjoint_residual <= self.adjoint_gradient_tolerance
                    )
                valid = eligible_i & adjoint_valid
                # Contract the adjoint with residual derivatives directly.
                # Building full Jacobians here is needlessly expensive because
                # the backward rule only requires J^T @ adjoint. VJPs preserve
                # the exact implicit-gradient formula and avoid materializing
                # dense theta/y_hat Jacobians (especially important for DAE
                # neighborhood contexts).
                _value, pullback_yhat = jax.vjp(
                    lambda value: self._residual(z_i, value, theta_i), yhat_i
                )
                grad_yhat = -pullback_yhat(adjoint)[0]
                if self.parameter_size and self.differentiate_problem_parameters:
                    _value, pullback_theta = jax.vjp(
                        lambda value: self._residual(z_i, yhat_i, value), theta_i
                    )
                    grad_theta = -pullback_theta(adjoint)[0]
                else:
                    grad_theta = jnp.zeros((self.parameter_size,), dtype=grad_yhat.dtype)
                grad_yhat = jnp.where(valid, grad_yhat, 0.0)
                grad_theta = jnp.where(valid, grad_theta, 0.0)
                return grad_yhat, grad_theta

            grad_yhat, grad_theta = jax.vmap(one_backward)(
                z, y_hat, theta, cotangent_y, eligible
            )
            return grad_yhat, grad_theta

        project_with_info.defvjp(forward, backward)
        return project_with_info

    def _parameters(self, parameters, batch):
        if self.parameter_size == 0:
            return jnp.zeros((batch, 0), dtype=self.dtype)
        if parameters is None:
            raise ValueError(
                "parameters are required for this Fischer--Burmeister layer"
            )
        parameters = jnp.asarray(parameters, dtype=self.dtype)
        if parameters.shape != (batch, self.parameter_size):
            raise ValueError(
                f"parameters must have shape {(batch, self.parameter_size)}, "
                f"got {parameters.shape}"
            )
        return parameters

    def __call__(self, y_hat, parameters=None):
        y_hat = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = y_hat.ndim == 1
        if squeeze:
            y_hat = y_hat[None, :]
        if y_hat.shape[0] != self.batch_size:
            raise ValueError(
                f"prepared projection requires batch size {self.batch_size}; "
                f"got {y_hat.shape[0]}"
            )
        theta = self._parameters(parameters, y_hat.shape[0])
        output = self._project_with_info(y_hat, theta)[0]
        return output[0] if squeeze else output

    def with_info(self, y_hat, parameters=None):
        y_hat = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = y_hat.ndim == 1
        if squeeze:
            y_hat = y_hat[None, :]
        if y_hat.shape[0] != self.batch_size:
            raise ValueError(
                f"prepared projection requires batch size {self.batch_size}; "
                f"got {y_hat.shape[0]}"
            )
        theta = self._parameters(parameters, y_hat.shape[0])
        projected, status, iterations, residual, finite, eligible = (
            self._project_with_info(y_hat, theta)
        )
        info = ProjectionInfo(
            forward_status_codes=status,
            forward_iterations=iterations,
            forward_residual_inf_norm=residual,
            forward_state_finite=finite,
            gradient_eligible=eligible,
        )
        return (projected[0] if squeeze else projected), info


@dataclass(frozen=True)
class FischerBurmeisterProjection:
    """Semismooth projection layer based on Fischer--Burmeister equations."""

    problem: Any
    tolerance: float = 1e-9
    max_iterations: int = 100
    regularization: float = 1e-8
    smoothing: float = 0.0
    line_search: bool = True
    max_line_search_steps: int = 12
    line_search_factor: float = 0.5
    gradient_policy: str = "residual_checked"
    inexact_gradient_tolerance: float = 1e-6
    adjoint_gradient_tolerance: float = 1e-8
    differentiate_problem_parameters: bool = True

    def prepare(
        self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"
    ) -> PreparedFischerBurmeisterProjection:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        return PreparedFischerBurmeisterProjection(
            problem=self.problem,
            batch_size=int(batch_size),
            dtype=dtype,
            backend=backend,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            regularization=self.regularization,
            smoothing=self.smoothing,
            line_search=self.line_search,
            max_line_search_steps=self.max_line_search_steps,
            line_search_factor=self.line_search_factor,
            gradient_policy=self.gradient_policy,
            inexact_gradient_tolerance=self.inexact_gradient_tolerance,
            adjoint_gradient_tolerance=self.adjoint_gradient_tolerance,
            differentiate_problem_parameters=self.differentiate_problem_parameters,
        )


