"""Implicit differentiation of the final JAX KKT system.

The backward pass solves the final fixed-barrier adjoint equation

    K.T @ lambda = output_cotangent

using the same reduced symmetric KKT matrix as the forward Newton solve.  A
parameter gradient is then formed with a residual VJP:

    dL/dtheta = direct_term - F_theta.T @ lambda.

The custom-VJP solver factory differentiates the final KKT root, not the
unrolled optimization trajectory, filter decisions, or barrier update.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp

from ..sparse.bounds import compute_bound_slacks
from ..sparse.linear_solve import dense_linear_solve
from ..sparse.newton import (
    build_full_kkt_matrix,
    build_reduced_kkt_matrix,
    compute_sigma,
    split_reduced_solution,
)
from ..sparse.residuals import compute_kkt_residual
from ..sparse.types import (
    BoundIndices,
    BoundSlacks,
    BoundValues,
    NLPQuantities,
    NewtonRegularization,
    PrimalDualState,
)


Parameters = TypeVar("Parameters")
KKTDataFunction = Callable[
    [PrimalDualState, Parameters],
    tuple[NLPQuantities, BoundValues],
]
ForwardSolver = Callable[[Parameters], PrimalDualState]


class AdjointSolveResult(NamedTuple):
    adjoint: PrimalDualState
    output_cotangent: PrimalDualState
    slacks: BoundSlacks
    reduced_matrix: jax.Array
    reduced_rhs: jax.Array
    reduced_solution: jax.Array
    reduced_residual_inf_norm: jax.Array
    full_residual_inf_norm: jax.Array


class MultiAdjointSolveResult(NamedTuple):
    adjoint_matrix: jax.Array
    output_cotangent_matrix: jax.Array
    reduced_matrix: jax.Array
    reduced_rhs_matrix: jax.Array
    reduced_solution_matrix: jax.Array
    reduced_residual_inf_norm: jax.Array
    full_residual_inf_norm: jax.Array


class ImplicitVJPResult(NamedTuple):
    parameter_gradient: object
    adjoint_result: AdjointSolveResult


def state_from_flat(
    flat,
    template: PrimalDualState,
) -> PrimalDualState:
    """Split a flat JAX vector using a template state's static block sizes."""
    flat = jnp.asarray(flat)
    blocks = []
    start = 0
    for block in template:
        stop = start + block.shape[0]
        blocks.append(flat[start:stop])
        start = stop
    return PrimalDualState(*blocks)


def primal_output_cotangent(
    state: PrimalDualState,
    grad_x,
) -> PrimalDualState:
    """Create a full KKT cotangent from a gradient with respect to ``x``."""
    return PrimalDualState(
        x=jnp.asarray(grad_x),
        s=jnp.zeros_like(state.s),
        y_c=jnp.zeros_like(state.y_c),
        y_d=jnp.zeros_like(state.y_d),
        z_L=jnp.zeros_like(state.z_L),
        z_U=jnp.zeros_like(state.z_U),
        v_L=jnp.zeros_like(state.v_L),
        v_U=jnp.zeros_like(state.v_U),
    )


def build_reduced_adjoint_rhs(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
    output_cotangent: PrimalDualState,
):
    """Build the transpose-eliminated right-hand side."""
    b_x = output_cotangent.x
    b_s = output_cotangent.s

    b_x = b_x.at[indices.x_lower].add(
        -(state.z_L / slacks.x_lower) * output_cotangent.z_L
    )
    b_x = b_x.at[indices.x_upper].add(
        (state.z_U / slacks.x_upper) * output_cotangent.z_U
    )
    b_s = b_s.at[indices.s_lower].add(
        -(state.v_L / slacks.s_lower) * output_cotangent.v_L
    )
    b_s = b_s.at[indices.s_upper].add(
        (state.v_U / slacks.s_upper) * output_cotangent.v_U
    )

    return jnp.concatenate(
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
    lambda_x,
    lambda_s,
):
    """Recover the four eliminated bound-adjoint blocks."""
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
):
    """Return ``K.T @ lambda - g``."""
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
    """Solve one final fixed-barrier adjoint system."""
    slacks = compute_bound_slacks(state, indices, bounds)
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
    reduced_solution = dense_linear_solve(
        reduced_matrix,
        reduced_rhs,
        symmetric=True,
    )

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
        lambda_x,
        lambda_s,
        lambda_c,
        lambda_d,
        lambda_z_L,
        lambda_z_U,
        lambda_v_L,
        lambda_v_U,
    )

    reduced_residual = reduced_matrix @ reduced_solution - reduced_rhs
    reduced_norm = jnp.max(jnp.abs(reduced_residual), initial=0.0)

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
        full_norm = jnp.max(jnp.abs(full_residual), initial=0.0)
    else:
        full_norm = jnp.asarray(jnp.nan, dtype=state.x.dtype)

    return AdjointSolveResult(
        adjoint,
        output_cotangent,
        slacks,
        reduced_matrix,
        reduced_rhs,
        reduced_solution,
        reduced_norm,
        full_norm,
    )


def solve_implicit_adjoint_multi_rhs(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    output_cotangent_matrix,
    *,
    regularization: NewtonRegularization | None = None,
) -> MultiAdjointSolveResult:
    """Solve multiple flat KKT cotangents with one matrix factorization call."""
    output_cotangent_matrix = jnp.asarray(output_cotangent_matrix)
    slacks = compute_bound_slacks(state, indices, bounds)
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    reduced_matrix = build_reduced_kkt_matrix(
        state,
        nlp,
        sigma_x,
        sigma_s,
        regularization=regularization,
    )

    def one_reduced_rhs(flat_cotangent):
        return build_reduced_adjoint_rhs(
            state,
            indices,
            slacks,
            state_from_flat(flat_cotangent, state),
        )

    reduced_rhs_matrix = jax.vmap(
        one_reduced_rhs,
        in_axes=1,
        out_axes=1,
    )(output_cotangent_matrix)
    reduced_solution_matrix = dense_linear_solve(
        reduced_matrix,
        reduced_rhs_matrix,
        symmetric=True,
    )

    def recover_one(reduced_solution, flat_cotangent):
        cotangent = state_from_flat(flat_cotangent, state)
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
        return PrimalDualState(
            lambda_x,
            lambda_s,
            lambda_c,
            lambda_d,
            lambda_z_L,
            lambda_z_U,
            lambda_v_L,
            lambda_v_U,
        ).flatten()

    adjoint_matrix = jax.vmap(
        recover_one,
        in_axes=(1, 1),
        out_axes=1,
    )(reduced_solution_matrix, output_cotangent_matrix)

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
    full_residual = (
        full_matrix.T @ adjoint_matrix - output_cotangent_matrix
    )

    return MultiAdjointSolveResult(
        adjoint_matrix,
        output_cotangent_matrix,
        reduced_matrix,
        reduced_rhs_matrix,
        reduced_solution_matrix,
        jnp.max(jnp.abs(reduced_residual), initial=0.0),
        jnp.max(jnp.abs(full_residual), initial=0.0),
    )


def _default_residual_function(
    state: PrimalDualState,
    parameters: Parameters,
    *,
    kkt_data_function: KKTDataFunction,
    indices: BoundIndices,
    mu: float,
):
    nlp, bounds = kkt_data_function(state, parameters)
    slacks = compute_bound_slacks(state, indices, bounds)
    return compute_kkt_residual(
        state,
        nlp,
        indices,
        slacks,
        mu,
    ).flatten()


def implicit_parameter_vjp(
    state: PrimalDualState,
    parameters: Parameters,
    output_cotangent: PrimalDualState,
    *,
    kkt_data_function: KKTDataFunction,
    indices: BoundIndices,
    mu: float,
    regularization: NewtonRegularization | None = None,
) -> ImplicitVJPResult:
    r"""Compute ``-F_theta.T @ lambda`` with a JAX residual VJP."""
    stopped_state = jax.lax.stop_gradient(state)
    nlp, bounds = kkt_data_function(stopped_state, parameters)
    adjoint_result = solve_implicit_adjoint(
        stopped_state,
        nlp,
        indices,
        bounds,
        output_cotangent,
        regularization=regularization,
    )

    def residual_at_parameters(current_parameters):
        return _default_residual_function(
            stopped_state,
            current_parameters,
            kkt_data_function=kkt_data_function,
            indices=indices,
            mu=mu,
        )

    _, residual_pullback = jax.vjp(
        residual_at_parameters,
        parameters,
    )
    parameter_contraction = residual_pullback(
        adjoint_result.adjoint.flatten()
    )[0]
    parameter_gradient = jax.tree_util.tree_map(
        lambda value: -value,
        parameter_contraction,
    )
    return ImplicitVJPResult(parameter_gradient, adjoint_result)


def make_implicit_differentiable_solver(
    forward_solver: ForwardSolver,
    *,
    kkt_data_function: KKTDataFunction,
    indices: BoundIndices,
    mu: float,
    regularization: NewtonRegularization | None = None,
):
    """Wrap a forward solver with a final-KKT custom VJP.

    The forward function may contain a fixed number of unrolled iterations,
    ``lax.scan``, or another solver implementation.  Reverse-mode
    differentiation does not trace through those iterations.  Instead, it
    performs one final transpose-KKT solve and one residual VJP.
    """
    if not callable(forward_solver):
        raise TypeError("forward_solver must be callable.")
    if not callable(kkt_data_function):
        raise TypeError("kkt_data_function must be callable.")

    @jax.custom_vjp
    def implicit_solver(parameters):
        return forward_solver(parameters)

    def forward_rule(parameters):
        final_state = forward_solver(parameters)
        return final_state, (final_state, parameters)

    def backward_rule(saved, output_cotangent):
        final_state, parameters = saved
        result = implicit_parameter_vjp(
            final_state,
            parameters,
            output_cotangent,
            kkt_data_function=kkt_data_function,
            indices=indices,
            mu=mu,
            regularization=regularization,
        )
        return (result.parameter_gradient,)

    implicit_solver.defvjp(forward_rule, backward_rule)
    return implicit_solver
