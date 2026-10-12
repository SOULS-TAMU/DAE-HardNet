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


@dataclass(frozen=True)
class ProjectionInfo:
    forward_status_codes: Any
    forward_iterations: Any
    forward_residual_inf_norm: Any
    forward_state_finite: Any
    gradient_eligible: Any

    @property
    def forward_converged(self):
        return self.forward_status_codes == BATCH_STATUS_CONVERGED


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


def _as_numpy_matrix(value, *, name: str):
    if scipy_sparse is not None and scipy_sparse.issparse(value):
        matrix = value.astype(np.float64).tocsr()
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        return matrix
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return matrix


def _dense(value):
    if scipy_sparse is not None and scipy_sparse.issparse(value):
        return value.toarray()
    return np.asarray(value)


@dataclass(frozen=True, eq=False)
class PreparedLinearEqualityProjection:
    A: Any
    b0: Any
    b_parameter_map: Any
    correction: Any
    batch_size: int
    dtype: Any
    backend: str
    regularization: float

    @property
    def parameter_size(self) -> int:
        return int(self.b_parameter_map.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.A.shape[1])

    def _rhs(self, parameters, batch_size):
        if self.parameter_size == 0:
            return jnp.broadcast_to(self.b0, (batch_size, self.b0.shape[0]))
        if parameters is None:
            raise ValueError(
                "parameters are required because b_parameter_map is nonempty"
            )
        parameters = jnp.asarray(parameters, dtype=self.dtype)
        return self.b0 + parameters @ self.b_parameter_map.T

    def __call__(self, y_hat, parameters=None):
        y_hat = jnp.asarray(y_hat, dtype=self.dtype)
        squeeze = y_hat.ndim == 1
        if squeeze:
            y_hat = y_hat[None, :]
        if y_hat.shape[-1] != self.output_size:
            raise ValueError(
                f"y_hat must end in dimension {self.output_size}; "
                f"got {y_hat.shape}"
            )
        if y_hat.shape[0] != self.batch_size:
            raise ValueError(
                f"prepared projection requires batch size {self.batch_size}; "
                f"got {y_hat.shape[0]}"
            )
        rhs = self._rhs(parameters, y_hat.shape[0])
        residual = y_hat @ self.A.T - rhs
        projected = y_hat - residual @ self.correction.T
        return projected[0] if squeeze else projected

    def with_info(self, y_hat, parameters=None):
        projected = self(y_hat, parameters)
        values = projected[None, :] if projected.ndim == 1 else projected
        rhs = self._rhs(parameters, values.shape[0])
        residual = jnp.max(jnp.abs(values @ self.A.T - rhs), axis=-1)
        finite = jnp.all(jnp.isfinite(values), axis=-1) & jnp.isfinite(residual)
        status = jnp.where(
            finite, FB_STATUS_CONVERGED, FB_STATUS_NONFINITE
        ).astype(jnp.int32)
        info = ProjectionInfo(
            forward_status_codes=status,
            forward_iterations=jnp.zeros_like(status),
            forward_residual_inf_norm=residual,
            forward_state_finite=finite,
            gradient_eligible=finite,
        )
        return projected, info


@dataclass(frozen=True)
class LinearEqualityProjection:
    """Explicit Euclidean projection onto ``A y = b(theta)``.

    Preparation computes the correction operator once. Runtime consists only of
    matrix multiplications and therefore has no iterative-solver overhead.
    """

    A: Any
    b: Any | None = None
    b0: Any | None = None
    b_parameter_map: Any | None = None
    regularization: float = 0.0

    def prepare(
        self, *, batch_size: int, dtype=jnp.float64, backend: str = "cpu"
    ) -> PreparedLinearEqualityProjection:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if self.regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        A = _as_numpy_matrix(self.A, name="A")
        neq, _n = A.shape
        source_b = self.b0 if self.b0 is not None else self.b
        if source_b is None:
            source_b = np.zeros((neq,), dtype=np.float64)
        b0 = np.asarray(source_b, dtype=np.float64)
        if b0.shape != (neq,):
            raise ValueError(f"b must have shape {(neq,)}, got {b0.shape}")
        if self.b_parameter_map is None:
            B = np.zeros((neq, 0), dtype=np.float64)
        else:
            B = np.asarray(self.b_parameter_map, dtype=np.float64)
            if B.ndim != 2 or B.shape[0] != neq:
                raise ValueError(
                    "b_parameter_map must have shape (number of equalities, m)"
                )
        A_dense = _dense(A)
        if neq == 0:
            correction = np.zeros((_n, 0), dtype=np.float64)
        else:
            gram = A_dense @ A_dense.T
            if self.regularization == 0.0:
                rank = np.linalg.matrix_rank(gram)
                if rank != neq:
                    raise ValueError(
                        "A must have full row rank when regularization is zero"
                    )
            gram = gram + self.regularization * np.eye(neq)
            if scipy_linalg is not None:
                factor, lower = scipy_linalg.cho_factor(
                    gram, lower=True, check_finite=True
                )
                solved = scipy_linalg.cho_solve(
                    (factor, lower), A_dense, check_finite=True
                )
            else:  # pragma: no cover
                solved = np.linalg.solve(gram, A_dense)
            correction = solved.T
        return PreparedLinearEqualityProjection(
            A=jnp.asarray(A_dense, dtype=dtype),
            b0=jnp.asarray(b0, dtype=dtype),
            b_parameter_map=jnp.asarray(B, dtype=dtype),
            correction=jnp.asarray(correction, dtype=dtype),
            batch_size=int(batch_size),
            dtype=dtype,
            backend=str(backend),
            regularization=float(self.regularization),
        )


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


# The older name remains as a direct alias, not an additional runtime wrapper.
ExplicitLinearProjection = LinearEqualityProjection
