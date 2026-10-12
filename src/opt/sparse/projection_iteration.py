"""Projection-specialized batched primal--dual solvers.

The objective is

    0.5 * ||x - target(parameters)||_2**2,

so its gradient is ``x - target`` and its Hessian is the identity.  The dense
path is provided by :mod:`batch_iteration`; this module implements:

* a matrix-free sparse iterative Newton path using Jacobian products, HVPs, and GMRES;
* a shared-linear equality path that factorizes one KKT matrix for all RHS.

Both paths are pure JAX and execute on the active CPU or CUDA device.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from .batch_iteration import (
    BATCH_STATUS_ACTIVE,
    BATCH_STATUS_CONVERGED,
    BATCH_STATUS_MAXIMUM_ITERATIONS,
    BATCH_STATUS_NUMERICAL_FAILURE,
    BATCH_STATUS_STALLED,
    BatchedFixedMuResult,
    _state_rows_are_finite,
    _tree_batch_where,
)
from .bounds import compute_bound_slacks
from .newton import (
    build_reduced_kkt_rhs,
    compute_sigma,
    recover_bound_multiplier_directions,
    split_reduced_solution,
)
from .step import apply_primal_dual_step, compute_fraction_to_boundary_steps
from .types import (
    BoundIndices,
    BoundValues,
    KKTResidual,
    NewtonRegularization,
    PrimalDualState,
)

VectorFunction = Callable[[jax.Array, jax.Array], jax.Array]
SparseValuesFunction = Callable[[jax.Array, jax.Array], jax.Array]
ConstraintHVPFunction = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def _projection_target(parameters: jax.Array, target_parameter_indices: jax.Array) -> jax.Array:
    return parameters[target_parameter_indices]


def _sparse_jacobian_matvec(
    values: jax.Array,
    rows: jax.Array,
    cols: jax.Array,
    vector: jax.Array,
    n_rows: int,
) -> jax.Array:
    result = jnp.zeros((n_rows,), dtype=vector.dtype)
    return result.at[rows].add(values * vector[cols])


def _sparse_jacobian_transpose_matvec(
    values: jax.Array,
    rows: jax.Array,
    cols: jax.Array,
    vector: jax.Array,
    n_cols: int,
) -> jax.Array:
    result = jnp.zeros((n_cols,), dtype=vector.dtype)
    return result.at[cols].add(values * vector[rows])


def _projection_residual(
    state: PrimalDualState,
    parameters: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_sparse_values_function: SparseValuesFunction,
    inequality_sparse_values_function: SparseValuesFunction,
    equality_rows: jax.Array,
    equality_cols: jax.Array,
    inequality_rows: jax.Array,
    inequality_cols: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> tuple[KKTResidual, object]:
    slacks = compute_bound_slacks(state, indices, bounds)
    c = equality_function(state.x, parameters)
    d = inequality_function(state.x, parameters)
    equality_values = equality_sparse_values_function(state.x, parameters)
    inequality_values = inequality_sparse_values_function(state.x, parameters)
    target = _projection_target(parameters, target_parameter_indices)

    r_x = state.x - target
    r_x = r_x + _sparse_jacobian_transpose_matvec(
        equality_values, equality_rows, equality_cols, state.y_c, state.x.shape[0]
    )
    r_x = r_x + _sparse_jacobian_transpose_matvec(
        inequality_values, inequality_rows, inequality_cols, state.y_d, state.x.shape[0]
    )
    r_x = r_x.at[indices.x_lower].add(-state.z_L)
    r_x = r_x.at[indices.x_upper].add(state.z_U)

    r_s = -state.y_d
    r_s = r_s.at[indices.s_lower].add(-state.v_L)
    r_s = r_s.at[indices.s_upper].add(state.v_U)

    residual = KKTResidual(
        x=r_x,
        s=r_s,
        y_c=c,
        y_d=d - state.s,
        z_L=slacks.x_lower * state.z_L - mu,
        z_U=slacks.x_upper * state.z_U - mu,
        v_L=slacks.s_lower * state.v_L - mu,
        v_U=slacks.s_upper * state.v_U - mu,
    )
    return residual, slacks

def _residual_infinity_norm(residual: KKTResidual) -> jax.Array:
    return jnp.max(jnp.abs(residual.flatten()), initial=0.0)


def _matrix_free_reduced_direction(
    state: PrimalDualState,
    parameters: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_sparse_values_function: SparseValuesFunction,
    inequality_sparse_values_function: SparseValuesFunction,
    constraint_hvp_function: ConstraintHVPFunction,
    equality_rows: jax.Array,
    equality_cols: jax.Array,
    inequality_rows: jax.Array,
    inequality_cols: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
    regularization: NewtonRegularization,
    *,
    gmres_tolerance: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int,
    use_preconditioner: bool,
    constraints_affine_in_variables: bool,
) -> tuple[PrimalDualState, jax.Array, jax.Array]:
    residual, slacks = _projection_residual(
        state,
        parameters,
        equality_function,
        inequality_function,
        equality_sparse_values_function,
        inequality_sparse_values_function,
        equality_rows,
        equality_cols,
        inequality_rows,
        inequality_cols,
        target_parameter_indices,
        indices,
        bounds,
        mu,
    )
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    rhs = build_reduced_kkt_rhs(residual, indices, slacks)
    equality_values = equality_sparse_values_function(state.x, parameters)
    inequality_values = inequality_sparse_values_function(state.x, parameters)

    n_x = state.x.shape[0]
    n_c = state.y_c.shape[0]
    n_d = state.y_d.shape[0]

    def reduced_matvec(vector):
        vector_x, vector_s, vector_c, vector_d = split_reduced_solution(vector, state)

        constraint_hvp = (
            jnp.zeros_like(vector_x)
            if constraints_affine_in_variables
            else constraint_hvp_function(
                state.x,
                parameters,
                state.y_c,
                state.y_d,
                vector_x,
            )
        )
        jac_c_vector = _sparse_jacobian_matvec(
            equality_values, equality_rows, equality_cols, vector_x, n_c
        )
        jac_d_vector = _sparse_jacobian_matvec(
            inequality_values, inequality_rows, inequality_cols, vector_x, n_d
        )
        jac_c_transpose = _sparse_jacobian_transpose_matvec(
            equality_values, equality_rows, equality_cols, vector_c, n_x
        )
        jac_d_transpose = _sparse_jacobian_transpose_matvec(
            inequality_values, inequality_rows, inequality_cols, vector_d, n_x
        )

        row_x = (
            vector_x
            + constraint_hvp
            + (sigma_x + regularization.delta_x) * vector_x
            + jac_c_transpose
            + jac_d_transpose
        )
        row_s = (sigma_s + regularization.delta_s) * vector_s - vector_d
        row_c = jac_c_vector - regularization.delta_c * vector_c
        row_d = jac_d_vector - vector_s - regularization.delta_d * vector_d
        return jnp.concatenate((row_x, row_s, row_c, row_d))

    preconditioner = None
    if use_preconditioner:
        primal_diag = jnp.maximum(
            jnp.abs(1.0 + sigma_x + regularization.delta_x), 1e-8
        )
        slack_diag = jnp.maximum(
            jnp.abs(sigma_s + regularization.delta_s), 1e-8
        )
        equality_diag = jnp.full(
            (n_c,),
            max(abs(float(regularization.delta_c)), 1.0),
            dtype=state.x.dtype,
        )
        inequality_diag = jnp.full(
            (n_d,),
            max(abs(float(regularization.delta_d)), 1.0),
            dtype=state.x.dtype,
        )
        diagonal = jnp.concatenate(
            (primal_diag, slack_diag, equality_diag, inequality_diag)
        )
        preconditioner = lambda vector: vector / diagonal

    reduced_solution, info = gmres(
        reduced_matvec,
        rhs,
        tol=gmres_tolerance,
        atol=gmres_atol,
        restart=gmres_restart,
        maxiter=gmres_maxiter,
        M=preconditioner,
        solve_method="batched",
    )
    delta_x, delta_s, delta_y_c, delta_y_d = split_reduced_solution(
        reduced_solution, state
    )
    delta_z_L, delta_z_U, delta_v_L, delta_v_U = recover_bound_multiplier_directions(
        state,
        indices,
        slacks,
        residual,
        delta_x,
        delta_s,
    )
    direction = PrimalDualState(
        delta_x,
        delta_s,
        delta_y_c,
        delta_y_d,
        delta_z_L,
        delta_z_U,
        delta_v_L,
        delta_v_U,
    )
    linear_residual = jnp.max(
        jnp.abs(reduced_matvec(reduced_solution) - rhs), initial=0.0
    )
    return direction, jnp.asarray(info), linear_residual

def run_batched_projection_sparse_iterations(
    initial_state: PrimalDualState,
    parameter_values: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_sparse_values_function: SparseValuesFunction,
    inequality_sparse_values_function: SparseValuesFunction,
    constraint_hvp_function: ConstraintHVPFunction,
    equality_rows: jax.Array,
    equality_cols: jax.Array,
    inequality_rows: jax.Array,
    inequality_cols: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu,
    *,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-9,
    tau: float = 0.995,
    step_scale: float = 1.0,
    minimum_step: float = 1e-14,
    regularization: NewtonRegularization | None = None,
    gmres_tolerance: float = 1e-9,
    gmres_atol: float = 0.0,
    gmres_restart: int = 30,
    gmres_maxiter: int = 50,
    use_preconditioner: bool = True,
    constraints_affine_in_variables: bool = False,
    jit: bool = True,
) -> BatchedFixedMuResult:
    """Run matrix-free projection Newton iterations over a parameter batch."""
    parameters = jnp.asarray(parameter_values)
    batch_size = parameters.shape[0]
    mu_values = jnp.asarray(mu, dtype=initial_state.x.dtype)
    if mu_values.ndim == 0:
        mu_values = jnp.broadcast_to(mu_values, (batch_size,))
    if mu_values.shape != (batch_size,):
        raise ValueError(f"mu must be scalar or have shape ({batch_size},).")
    reg = regularization or NewtonRegularization()
    target_indices = jnp.asarray(target_parameter_indices, dtype=jnp.int32)
    equality_rows = jnp.asarray(equality_rows, dtype=jnp.int32)
    equality_cols = jnp.asarray(equality_cols, dtype=jnp.int32)
    inequality_rows = jnp.asarray(inequality_rows, dtype=jnp.int32)
    inequality_cols = jnp.asarray(inequality_cols, dtype=jnp.int32)

    residual_batch = jax.vmap(
        lambda state, params, mu_i: _projection_residual(
            state,
            params,
            equality_function,
            inequality_function,
            equality_sparse_values_function,
            inequality_sparse_values_function,
            equality_rows,
            equality_cols,
            inequality_rows,
            inequality_cols,
            target_indices,
            indices,
            bounds,
            mu_i,
        )[0],
        in_axes=(0, 0, 0),
    )
    residual_norm_batch = jax.vmap(_residual_infinity_norm)
    solve_batch = jax.vmap(
        lambda state, params, mu_i: _matrix_free_reduced_direction(
            state,
            params,
            equality_function,
            inequality_function,
            equality_sparse_values_function,
            inequality_sparse_values_function,
            constraint_hvp_function,
            equality_rows,
            equality_cols,
            inequality_rows,
            inequality_cols,
            target_indices,
            indices,
            bounds,
            mu_i,
            reg,
            gmres_tolerance=gmres_tolerance,
            gmres_atol=gmres_atol,
            gmres_restart=gmres_restart,
            gmres_maxiter=gmres_maxiter,
            use_preconditioner=use_preconditioner,
            constraints_affine_in_variables=constraints_affine_in_variables,
        ),
        in_axes=(0, 0, 0),
    )
    step_limit_batch = jax.vmap(
        lambda state, direction: compute_fraction_to_boundary_steps(
            state, direction, indices, bounds, tau
        )
    )
    update_batch = jax.vmap(
        lambda state, direction, alpha_primal, alpha_dual: apply_primal_dual_step(
            state,
            direction,
            alpha_primal=alpha_primal,
            alpha_dual=alpha_dual,
        )
    )

    def solve_all(state0, params, mus):
        status0 = jnp.full((batch_size,), BATCH_STATUS_ACTIVE, dtype=jnp.int32)
        iteration0 = jnp.full((batch_size,), max_iterations, dtype=jnp.int32)

        def iteration_step(carry, iteration):
            state, status, stopping_iterations = carry
            residual = residual_batch(state, params, mus)
            residual_norm = residual_norm_batch(residual)

            active = status == BATCH_STATUS_ACTIVE
            finite = jnp.isfinite(residual_norm) & _state_rows_are_finite(state)
            failed = active & ~finite
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(failed, iteration, stopping_iterations)

            active = status == BATCH_STATUS_ACTIVE
            converged = active & (residual_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            stopping_iterations = jnp.where(converged, iteration, stopping_iterations)

            direction, info, linear_residual = solve_batch(state, params, mus)
            active = status == BATCH_STATUS_ACTIVE
            direction_ok = (
                _state_rows_are_finite(direction)
                & jnp.isfinite(linear_residual)
                & (linear_residual <= jnp.maximum(1e-5, 10.0 * gmres_tolerance))
            )
            # Some JAX backends return an experimental nonzero info code even
            # when the requested residual tolerance is reached; the explicit
            # residual above is the authoritative convergence check.
            failed = active & ~direction_ok
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(failed, iteration, stopping_iterations)

            active = status == BATCH_STATUS_ACTIVE
            limits = step_limit_batch(state, direction)
            alpha_primal = step_scale * limits.alpha_primal
            alpha_dual = step_scale * limits.alpha_dual
            stalled = active & (
                jnp.minimum(alpha_primal, alpha_dual) <= minimum_step
            )
            status = jnp.where(stalled, BATCH_STATUS_STALLED, status)
            stopping_iterations = jnp.where(stalled, iteration, stopping_iterations)

            should_update = status == BATCH_STATUS_ACTIVE
            candidate = update_batch(
                state, direction, alpha_primal, alpha_dual
            )
            update_failed = should_update & ~_state_rows_are_finite(candidate)
            status = jnp.where(
                update_failed, BATCH_STATUS_NUMERICAL_FAILURE, status
            )
            stopping_iterations = jnp.where(
                update_failed, iteration, stopping_iterations
            )
            should_update = status == BATCH_STATUS_ACTIVE
            state = _tree_batch_where(should_update, candidate, state)
            return (state, status, stopping_iterations), residual_norm

        iteration_indices = jnp.arange(max_iterations, dtype=jnp.int32)
        (state, status, stopping_iterations), history = jax.lax.scan(
            iteration_step,
            (state0, status0, iteration0),
            iteration_indices,
        )
        final_residual = residual_batch(state, params, mus)
        final_norm = residual_norm_batch(final_residual)
        active = status == BATCH_STATUS_ACTIVE
        finite = jnp.isfinite(final_norm) & _state_rows_are_finite(state)
        status = jnp.where(
            active & ~finite, BATCH_STATUS_NUMERICAL_FAILURE, status
        )
        active = status == BATCH_STATUS_ACTIVE
        converged = active & (final_norm <= residual_tolerance)
        status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
        status = jnp.where(
            status == BATCH_STATUS_ACTIVE,
            BATCH_STATUS_MAXIMUM_ITERATIONS,
            status,
        )
        stopping_iterations = jnp.where(
            converged, max_iterations, stopping_iterations
        )
        return BatchedFixedMuResult(
            state=state,
            status_codes=status,
            iterations=stopping_iterations,
            final_residual_inf_norm=final_norm,
            residual_history=jnp.concatenate((history, final_norm[None, :]), axis=0),
        )

    runner = jax.jit(solve_all) if jit else solve_all
    return runner(initial_state, parameters, mu_values)


def run_shared_linear_projection_batch(
    initial_state: PrimalDualState,
    parameter_values: jax.Array,
    equality_function: VectorFunction,
    equality_jacobian: jax.Array,
    target_parameter_indices: jax.Array,
    *,
    residual_tolerance: float = 1e-9,
    jit: bool = True,
) -> BatchedFixedMuResult:
    """Project a batch onto one shared affine equality system.

    The common KKT matrix is solved once with a matrix right-hand side.  This
    mode intentionally supports equality-only projection problems.
    """
    parameters = jnp.asarray(parameter_values)
    batch_size = parameters.shape[0]
    target_indices = jnp.asarray(target_parameter_indices, dtype=jnp.int32)
    matrix_a = jnp.asarray(equality_jacobian, dtype=initial_state.x.dtype)
    n_x = initial_state.x.shape[1]
    n_c = matrix_a.shape[0]

    def solve_all(params):
        zero_x = jnp.zeros((n_x,), dtype=initial_state.x.dtype)
        offsets = jax.vmap(lambda p: equality_function(zero_x, p))(params)
        targets = params[:, target_indices]
        identity = jnp.eye(n_x, dtype=initial_state.x.dtype)
        zero_cc = jnp.zeros((n_c, n_c), dtype=initial_state.x.dtype)
        kkt = jnp.block([[identity, matrix_a.T], [matrix_a, zero_cc]])
        rhs = jnp.concatenate((targets, -offsets), axis=1)
        solution = jnp.linalg.solve(kkt, rhs.T).T
        x = solution[:, :n_x]
        y_c = solution[:, n_x:]

        c = jax.vmap(equality_function)(x, params)
        stationarity = x - targets + y_c @ matrix_a
        residual_norm = jnp.maximum(
            jnp.max(jnp.abs(stationarity), axis=1, initial=0.0),
            jnp.max(jnp.abs(c), axis=1, initial=0.0),
        )
        finite = (
            jnp.all(jnp.isfinite(x), axis=1)
            & jnp.all(jnp.isfinite(y_c), axis=1)
            & jnp.isfinite(residual_norm)
        )
        converged = finite & (residual_norm <= residual_tolerance)
        status = jnp.where(
            converged,
            BATCH_STATUS_CONVERGED,
            BATCH_STATUS_NUMERICAL_FAILURE,
        ).astype(jnp.int32)

        empty = jnp.empty((batch_size, 0), dtype=initial_state.x.dtype)
        state = PrimalDualState(
            x=x,
            s=empty,
            y_c=y_c,
            y_d=empty,
            z_L=empty,
            z_U=empty,
            v_L=empty,
            v_U=empty,
        )

        initial_c = jax.vmap(equality_function)(initial_state.x, params)
        initial_stationarity = initial_state.x - targets + initial_state.y_c @ matrix_a
        initial_norm = jnp.maximum(
            jnp.max(jnp.abs(initial_stationarity), axis=1, initial=0.0),
            jnp.max(jnp.abs(initial_c), axis=1, initial=0.0),
        )
        return BatchedFixedMuResult(
            state=state,
            status_codes=status,
            iterations=jnp.where(converged, 1, 0).astype(jnp.int32),
            final_residual_inf_norm=residual_norm,
            residual_history=jnp.stack((initial_norm, residual_norm), axis=0),
        )

    runner = jax.jit(solve_all) if jit else solve_all
    return runner(parameters)
