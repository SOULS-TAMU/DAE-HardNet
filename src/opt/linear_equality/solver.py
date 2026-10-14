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


