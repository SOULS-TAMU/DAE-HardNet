"""Implicit differentiation for the final fixed-barrier KKT system.

This module differentiates the final barrier equations

    F(w, theta; mu) = 0

at a fixed final barrier parameter ``mu``.  It does not differentiate through
the iteration history, filter line search, or barrier update.

For an output cotangent ``g_w``, the adjoint solves

    F_w(w, theta; mu).T @ lambda = g_w.

The full primal-dual KKT Jacobian is nonsymmetric, but transpose elimination of
the four bound-adjoint blocks produces the same symmetric reduced KKT matrix
used by the forward Newton direction.  Only the reduced right-hand side and
bound-block recovery formulas differ.

NumPy has no native automatic-differentiation or VJP API.  The module therefore
provides the same adjoint solve and supports parameter gradients through either

1. an explicitly supplied residual Jacobian ``F_theta``, or
2. a user-supplied residual VJP callback.

The JAX backend provides automatic residual VJPs and a custom-VJP solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from ..solvers.bounds import BoundIndices, BoundSlacks, BoundValues, compute_bound_slacks
from ..solvers.newton import (
    NewtonRegularization,
    build_full_kkt_matrix,
    build_reduced_kkt_matrix,
    compute_sigma,
    split_reduced_solution,
)
from ..solvers.residuals import NLPQuantities
from ..solvers.state import FloatArray, PrimalDualState


ResidualParameterVJP = Callable[[FloatArray], FloatArray]
ResidualFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class AdjointSolveResult:
    """Result of one transpose-KKT adjoint solve."""

    adjoint: PrimalDualState
    output_cotangent: PrimalDualState
    slacks: BoundSlacks
    reduced_matrix: np.ndarray
    reduced_rhs: FloatArray
    reduced_solution: FloatArray
    reduced_residual_inf_norm: float
    full_residual_inf_norm: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.reduced_matrix, dtype=np.float64)
        rhs = np.asarray(self.reduced_rhs, dtype=np.float64)
        solution = np.asarray(self.reduced_solution, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "reduced_matrix must be square; "
                f"received shape {matrix.shape}."
            )
        if rhs.ndim != 1 or rhs.size != matrix.shape[0]:
            raise ValueError(
                "reduced_rhs must match reduced_matrix; "
                f"received {rhs.shape} and {matrix.shape}."
            )
        if solution.shape != rhs.shape:
            raise ValueError(
                "reduced_solution must match reduced_rhs; "
                f"received {solution.shape} and {rhs.shape}."
            )

        object.__setattr__(self, "reduced_matrix", matrix)
        object.__setattr__(self, "reduced_rhs", rhs)
        object.__setattr__(self, "reduced_solution", solution)
        object.__setattr__(
            self,
            "reduced_residual_inf_norm",
            float(self.reduced_residual_inf_norm),
        )
        object.__setattr__(
            self,
            "full_residual_inf_norm",
            float(self.full_residual_inf_norm),
        )


@dataclass(frozen=True)
class MultiAdjointSolveResult:
    """Multiple adjoints solved with one matrix and multiple right-hand sides."""

    adjoint_matrix: np.ndarray
    output_cotangent_matrix: np.ndarray
    reduced_matrix: np.ndarray
    reduced_rhs_matrix: np.ndarray
    reduced_solution_matrix: np.ndarray
    reduced_residual_inf_norm: float
    full_residual_inf_norm: float

    def __post_init__(self) -> None:
        for name in (
            "adjoint_matrix",
            "output_cotangent_matrix",
            "reduced_matrix",
            "reduced_rhs_matrix",
            "reduced_solution_matrix",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, value)


def primal_output_cotangent(
    state: PrimalDualState,
    grad_x: FloatArray,
) -> PrimalDualState:
    """Create a full output cotangent from a gradient with respect to ``x``."""
    grad_x = np.asarray(grad_x, dtype=np.float64)
    if grad_x.shape != state.x.shape:
        raise ValueError(
            f"grad_x must have shape {state.x.shape}; received {grad_x.shape}."
        )

    result = PrimalDualState.zeros_like(state)
    result.x = grad_x.copy()
    return result


def _validate_output_cotangent(
    state: PrimalDualState,
    output_cotangent: PrimalDualState,
) -> None:
    if state.block_sizes != output_cotangent.block_sizes:
        raise ValueError(
            "State and output cotangent block sizes must match: "
            f"{state.block_sizes} != {output_cotangent.block_sizes}."
        )
    output_cotangent.validate_finite()


def build_reduced_adjoint_rhs(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
    output_cotangent: PrimalDualState,
) -> FloatArray:
    r"""Build the reduced right-hand side for ``K.T lambda = g``.

    After eliminating the bound-adjoint blocks,

    .. math::

        b_x &=
        g_x
        -P_{xL}\operatorname{diag}(z_L/a_{xL})g_{zL}
        +P_{xU}\operatorname{diag}(z_U/a_{xU})g_{zU},\\

        b_s &=
        g_s
        -P_{dL}\operatorname{diag}(v_L/a_{sL})g_{vL}
        +P_{dU}\operatorname{diag}(v_U/a_{sU})g_{vU}.
    """
    _validate_output_cotangent(state, output_cotangent)
    indices.validate_against_state(state)

    if slacks.sizes != indices.sizes:
        raise ValueError(
            "Bound slack lengths must match bound index lengths: "
            f"{slacks.sizes} != {indices.sizes}."
        )
    slacks.validate_strictly_positive()

    b_x = output_cotangent.x.copy()
    b_s = output_cotangent.s.copy()

    np.add.at(
        b_x,
        indices.x_lower,
        -(state.z_L / slacks.x_lower) * output_cotangent.z_L,
    )
    np.add.at(
        b_x,
        indices.x_upper,
        (state.z_U / slacks.x_upper) * output_cotangent.z_U,
    )
    np.add.at(
        b_s,
        indices.s_lower,
        -(state.v_L / slacks.s_lower) * output_cotangent.v_L,
    )
    np.add.at(
        b_s,
        indices.s_upper,
        (state.v_U / slacks.s_upper) * output_cotangent.v_U,
    )

    return np.concatenate(
        (
            b_x,
            b_s,
            output_cotangent.y_c,
            output_cotangent.y_d,
        )
    )


def recover_bound_adjoint_blocks(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
    output_cotangent: PrimalDualState,
    lambda_x: FloatArray,
    lambda_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Recover ``lambda_zL, lambda_zU, lambda_vL, lambda_vU``."""
    lambda_x = np.asarray(lambda_x, dtype=np.float64)
    lambda_s = np.asarray(lambda_s, dtype=np.float64)

    if lambda_x.shape != state.x.shape:
        raise ValueError(
            f"lambda_x must have shape {state.x.shape}; "
            f"received {lambda_x.shape}."
        )
    if lambda_s.shape != state.s.shape:
        raise ValueError(
            f"lambda_s must have shape {state.s.shape}; "
            f"received {lambda_s.shape}."
        )

    slacks.validate_strictly_positive()

    lambda_z_L = (
        output_cotangent.z_L + lambda_x[indices.x_lower]
    ) / slacks.x_lower
    lambda_z_U = (
        output_cotangent.z_U - lambda_x[indices.x_upper]
    ) / slacks.x_upper
    lambda_v_L = (
        output_cotangent.v_L + lambda_s[indices.s_lower]
    ) / slacks.s_lower
    lambda_v_U = (
        output_cotangent.v_U - lambda_s[indices.s_upper]
    ) / slacks.s_upper

    return lambda_z_L, lambda_z_U, lambda_v_L, lambda_v_U


def compute_full_adjoint_residual(
    state: PrimalDualState,
    adjoint: PrimalDualState,
    output_cotangent: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    *,
    regularization: NewtonRegularization | None = None,
) -> FloatArray:
    """Return ``K.T @ lambda - g`` using the same full KKT matrix as Newton."""
    _validate_output_cotangent(state, output_cotangent)

    matrix = build_full_kkt_matrix(
        state,
        nlp,
        indices,
        slacks,
        regularization=regularization,
    )
    return matrix.T @ adjoint.flatten() - output_cotangent.flatten()


def solve_implicit_adjoint(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    output_cotangent: PrimalDualState,
    *,
    regularization: NewtonRegularization | None = None,
    check_full_residual: bool = True,
) -> AdjointSolveResult:
    """Solve the final fixed-``mu`` transpose KKT system.

    ``mu`` does not appear explicitly because the Jacobian is determined by the
    final state, final slacks, exact final Lagrangian Hessian, and Jacobians.
    The caller is responsible for supplying quantities evaluated at the final
    state for the fixed final barrier problem.
    """
    state.validate_finite()
    nlp.validate_against_state(state)
    nlp.validate_hessian_symmetry()
    _validate_output_cotangent(state, output_cotangent)
    indices.validate_against_state(state)
    bounds.validate_against_indices(indices)

    slacks = compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
    )
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    reduced_matrix = build_reduced_kkt_matrix(
        state,
        nlp,
        sigma_x,
        sigma_s,
        regularization=regularization,
    )
    reduced_rhs = build_reduced_adjoint_rhs(
        state,
        indices,
        slacks,
        output_cotangent,
    )

    try:
        reduced_solution = np.linalg.solve(reduced_matrix, reduced_rhs)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The reduced transpose KKT system could not be solved. "
            "The final KKT Jacobian may be singular or ill-conditioned."
        ) from error

    lambda_x, lambda_s, lambda_c, lambda_d = split_reduced_solution(
        reduced_solution,
        state,
    )
    lambda_z_L, lambda_z_U, lambda_v_L, lambda_v_U = (
        recover_bound_adjoint_blocks(
            state,
            indices,
            slacks,
            output_cotangent,
            lambda_x,
            lambda_s,
        )
    )
    adjoint = PrimalDualState(
        x=lambda_x,
        s=lambda_s,
        y_c=lambda_c,
        y_d=lambda_d,
        z_L=lambda_z_L,
        z_U=lambda_z_U,
        v_L=lambda_v_L,
        v_U=lambda_v_U,
    )

    reduced_residual = reduced_matrix @ reduced_solution - reduced_rhs
    reduced_norm = float(
        np.max(np.abs(reduced_residual))
        if reduced_residual.size
        else 0.0
    )

    if check_full_residual:
        full_residual = compute_full_adjoint_residual(
            state,
            adjoint,
            output_cotangent,
            nlp,
            indices,
            slacks,
            regularization=regularization,
        )
        full_norm = float(
            np.max(np.abs(full_residual))
            if full_residual.size
            else 0.0
        )
    else:
        full_norm = float("nan")

    return AdjointSolveResult(
        adjoint=adjoint,
        output_cotangent=output_cotangent,
        slacks=slacks,
        reduced_matrix=reduced_matrix,
        reduced_rhs=reduced_rhs,
        reduced_solution=reduced_solution,
        reduced_residual_inf_norm=reduced_norm,
        full_residual_inf_norm=full_norm,
    )


def solve_implicit_adjoint_multi_rhs(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    output_cotangents: Iterable[PrimalDualState],
    *,
    regularization: NewtonRegularization | None = None,
) -> MultiAdjointSolveResult:
    """Solve several adjoints in one dense multi-right-hand-side solve."""
    cotangents = tuple(output_cotangents)
    if not cotangents:
        raise ValueError("At least one output cotangent is required.")

    for cotangent in cotangents:
        _validate_output_cotangent(state, cotangent)

    slacks = compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
    )
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    reduced_matrix = build_reduced_kkt_matrix(
        state,
        nlp,
        sigma_x,
        sigma_s,
        regularization=regularization,
    )
    reduced_rhs_matrix = np.column_stack(
        [
            build_reduced_adjoint_rhs(
                state,
                indices,
                slacks,
                cotangent,
            )
            for cotangent in cotangents
        ]
    )

    try:
        reduced_solution_matrix = np.linalg.solve(
            reduced_matrix,
            reduced_rhs_matrix,
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The reduced transpose KKT system could not be solved for the "
            "multiple right-hand sides."
        ) from error

    adjoint_columns: list[FloatArray] = []
    for column, cotangent in enumerate(cotangents):
        reduced_solution = reduced_solution_matrix[:, column]
        lambda_x, lambda_s, lambda_c, lambda_d = split_reduced_solution(
            reduced_solution,
            state,
        )
        lambda_z_L, lambda_z_U, lambda_v_L, lambda_v_U = (
            recover_bound_adjoint_blocks(
                state,
                indices,
                slacks,
                cotangent,
                lambda_x,
                lambda_s,
            )
        )
        adjoint_columns.append(
            PrimalDualState(
                lambda_x,
                lambda_s,
                lambda_c,
                lambda_d,
                lambda_z_L,
                lambda_z_U,
                lambda_v_L,
                lambda_v_U,
            ).flatten()
        )

    adjoint_matrix = np.column_stack(adjoint_columns)
    output_matrix = np.column_stack(
        [cotangent.flatten() for cotangent in cotangents]
    )
    reduced_residual = (
        reduced_matrix @ reduced_solution_matrix - reduced_rhs_matrix
    )

    full_matrix = build_full_kkt_matrix(
        state,
        nlp,
        indices,
        slacks,
        regularization=regularization,
    )
    full_residual = full_matrix.T @ adjoint_matrix - output_matrix

    return MultiAdjointSolveResult(
        adjoint_matrix=adjoint_matrix,
        output_cotangent_matrix=output_matrix,
        reduced_matrix=reduced_matrix,
        reduced_rhs_matrix=reduced_rhs_matrix,
        reduced_solution_matrix=reduced_solution_matrix,
        reduced_residual_inf_norm=float(
            np.max(np.abs(reduced_residual))
        ),
        full_residual_inf_norm=float(np.max(np.abs(full_residual))),
    )


def solve_full_adjoint_direct(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    output_cotangent: PrimalDualState,
    *,
    regularization: NewtonRegularization | None = None,
) -> PrimalDualState:
    """Dense full-space reference solve of ``K.T lambda = g``."""
    slacks = compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
    )
    matrix = build_full_kkt_matrix(
        state,
        nlp,
        indices,
        slacks,
        regularization=regularization,
    )
    try:
        flat = np.linalg.solve(matrix.T, output_cotangent.flatten())
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The full transpose KKT system could not be solved."
        ) from error
    return PrimalDualState.from_flat(flat, template=state)


def parameter_vjp_from_jacobian(
    adjoint: PrimalDualState,
    residual_parameter_jacobian: np.ndarray,
    *,
    direct_parameter_gradient: FloatArray | None = None,
) -> FloatArray:
    r"""Compute ``L_theta - F_theta.T @ lambda`` from a dense Jacobian."""
    jacobian = np.asarray(
        residual_parameter_jacobian,
        dtype=np.float64,
    )
    if jacobian.ndim != 2:
        raise ValueError(
            "residual_parameter_jacobian must be two-dimensional; "
            f"received shape {jacobian.shape}."
        )
    if jacobian.shape[0] != adjoint.size:
        raise ValueError(
            "The first Jacobian dimension must equal the KKT residual size: "
            f"{jacobian.shape[0]} != {adjoint.size}."
        )

    gradient = -(jacobian.T @ adjoint.flatten())
    if direct_parameter_gradient is not None:
        direct = np.asarray(
            direct_parameter_gradient,
            dtype=np.float64,
        )
        if direct.shape != gradient.shape:
            raise ValueError(
                "direct_parameter_gradient must have shape "
                f"{gradient.shape}; received {direct.shape}."
            )
        gradient = gradient + direct
    return gradient


def parameter_vjp_from_callback(
    adjoint: PrimalDualState,
    residual_vjp: ResidualParameterVJP,
    *,
    direct_parameter_gradient: FloatArray | None = None,
) -> FloatArray:
    r"""Compute ``L_theta - F_theta.T @ lambda`` from a VJP callback.

    The callback must return ``F_theta.T @ seed`` for the supplied flat KKT
    seed.  This is the NumPy extension point corresponding to ``jax.vjp``.
    """
    if not callable(residual_vjp):
        raise TypeError("residual_vjp must be callable.")

    contraction = np.asarray(
        residual_vjp(adjoint.flatten()),
        dtype=np.float64,
    )
    if contraction.ndim != 1:
        raise ValueError(
            "The residual VJP callback must return a one-dimensional array; "
            f"received shape {contraction.shape}."
        )

    gradient = -contraction
    if direct_parameter_gradient is not None:
        direct = np.asarray(
            direct_parameter_gradient,
            dtype=np.float64,
        )
        if direct.shape != gradient.shape:
            raise ValueError(
                "direct_parameter_gradient must have shape "
                f"{gradient.shape}; received {direct.shape}."
            )
        gradient = gradient + direct
    return gradient


def finite_difference_residual_parameter_jacobian(
    residual_function: ResidualFunction,
    parameters: FloatArray,
    *,
    relative_step: float = 1e-6,
) -> np.ndarray:
    """Debug-only centered finite-difference approximation of ``F_theta``.

    This helper is useful for validating a user-supplied NumPy VJP.  It is not
    intended as a production replacement for analytic derivatives.
    """
    if not callable(residual_function):
        raise TypeError("residual_function must be callable.")

    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 1:
        raise ValueError(
            f"parameters must be one-dimensional; received {parameters.shape}."
        )
    relative_step = float(relative_step)
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError(
            "relative_step must be finite and positive; "
            f"received {relative_step}."
        )

    base = np.asarray(residual_function(parameters), dtype=np.float64)
    if base.ndim != 1:
        raise ValueError(
            "residual_function must return a one-dimensional vector; "
            f"received {base.shape}."
        )

    jacobian = np.empty((base.size, parameters.size), dtype=np.float64)
    for column in range(parameters.size):
        step = relative_step * max(1.0, abs(parameters[column]))
        perturbation = np.zeros_like(parameters)
        perturbation[column] = step
        plus = np.asarray(
            residual_function(parameters + perturbation),
            dtype=np.float64,
        )
        minus = np.asarray(
            residual_function(parameters - perturbation),
            dtype=np.float64,
        )
        jacobian[:, column] = (plus - minus) / (2.0 * step)

    return jacobian
