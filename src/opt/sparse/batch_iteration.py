"""True batched fixed-barrier primal-dual iterations in JAX.

The batch dimension is mapped with :func:`jax.vmap`; Newton iterations are
executed with :func:`jax.lax.scan`.  There is no Python loop over individual
problem instances.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .bounds import compute_bound_slacks
from .newton import solve_primal_dual_newton_system
from .residuals import compute_kkt_residual, residual_infinity_norm
from .step import apply_primal_dual_step, compute_fraction_to_boundary_steps
from .types import (
    BoundIndices,
    BoundValues,
    NLPQuantities,
    NewtonRegularization,
    PrimalDualState,
)


BATCH_STATUS_ACTIVE = 0
BATCH_STATUS_CONVERGED = 1
BATCH_STATUS_MAXIMUM_ITERATIONS = 2
BATCH_STATUS_STALLED = 3
BATCH_STATUS_NUMERICAL_FAILURE = 4

BATCH_STATUS_NAMES = (
    "active",
    "converged",
    "maximum_iterations",
    "stalled",
    "numerical_failure",
)


class BatchedFixedMuResult(NamedTuple):
    """JAX result for a batch of fixed-barrier solves."""

    state: PrimalDualState
    status_codes: jax.Array
    iterations: jax.Array
    final_residual_inf_norm: jax.Array
    residual_history: jax.Array

    @property
    def converged(self):
        return self.status_codes == BATCH_STATUS_CONVERGED


BatchNLPEvaluator = Callable[[PrimalDualState, jax.Array], NLPQuantities]


def _is_jax_array(value) -> bool:
    return isinstance(value, jax.Array)


def _tree_batch_where(mask, new_tree, old_tree):
    """Select complete batch rows from two identically shaped pytrees."""

    def select(new, old):
        expanded = mask.reshape((mask.shape[0],) + (1,) * (new.ndim - 1))
        return jnp.where(expanded, new, old)

    return jax.tree_util.tree_map(select, new_tree, old_tree)


def _state_rows_are_finite(state: PrimalDualState):
    finite = jnp.ones((state.x.shape[0],), dtype=bool)
    for block in state:
        axes = tuple(range(1, block.ndim))
        block_finite = jnp.all(jnp.isfinite(block), axis=axes) if axes else jnp.isfinite(block)
        finite = finite & block_finite
    return finite


def run_batched_fixed_mu_iterations(
    initial_state: PrimalDualState,
    parameter_values,
    evaluate_nlp: BatchNLPEvaluator,
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
    check_newton_residual: bool = False,
    jit: bool = True,
) -> BatchedFixedMuResult:
    """Solve a same-structure batch without looping over problems in Python.

    Parameters have shape ``(batch, n_parameter)`` and every state block has a
    leading batch dimension.  The direct fraction-to-boundary update is used;
    discrete filter-line-search state is intentionally not part of this pure
    batched kernel.
    """

    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    residual_tolerance = float(residual_tolerance)
    tau = float(tau)
    step_scale = float(step_scale)
    minimum_step = float(minimum_step)
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must satisfy 0 < tau < 1.")
    if not 0.0 < step_scale <= 1.0:
        raise ValueError("step_scale must satisfy 0 < step_scale <= 1.")
    if minimum_step < 0.0:
        raise ValueError("minimum_step must be nonnegative.")

    if _is_jax_array(parameter_values):
        parameters = jnp.asarray(parameter_values, dtype=initial_state.x.dtype)
    else:
        parameter_host = np.asarray(parameter_values, dtype=np.float64)
        if not np.all(np.isfinite(parameter_host)):
            raise ValueError("parameter_values must be finite.")
        parameters = jnp.asarray(parameter_host, dtype=initial_state.x.dtype)
    if parameters.ndim != 2:
        raise ValueError(
            "parameter_values must have shape (batch, n_parameter); "
            f"received {parameters.shape}."
        )
    batch_size = parameters.shape[0]
    for block_name, block in zip(
        ("x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U"),
        initial_state,
        strict=True,
    ):
        if block.ndim != 2 or block.shape[0] != batch_size:
            raise ValueError(
                f"Initial-state block {block_name} must have shape (batch, width); "
                f"received {block.shape}."
            )

    if _is_jax_array(mu):
        mu_values = jnp.asarray(mu, dtype=initial_state.x.dtype)
    else:
        mu_host = np.asarray(mu, dtype=np.float64)
        if not np.all(np.isfinite(mu_host)) or np.any(mu_host < 0.0):
            raise ValueError("mu must contain finite nonnegative values.")
        mu_values = jnp.asarray(mu_host, dtype=initial_state.x.dtype)
    if mu_values.ndim == 0:
        mu_values = jnp.broadcast_to(mu_values, (batch_size,))
    elif mu_values.shape != (batch_size,):
        raise ValueError(f"mu must be scalar or have shape ({batch_size},); got {mu_values.shape}.")

    reg = regularization or NewtonRegularization()

    evaluate_batch = jax.vmap(evaluate_nlp, in_axes=(0, 0))
    solve_batch = jax.vmap(
        lambda state, nlp, mu_i: solve_primal_dual_newton_system(
            state,
            nlp,
            indices,
            bounds,
            mu_i,
            regularization=reg,
            check_full_residual=check_newton_residual,
        ),
        in_axes=(0, 0, 0),
    )
    step_limit_batch = jax.vmap(
        lambda state, direction: compute_fraction_to_boundary_steps(
            state,
            direction,
            indices,
            bounds,
            tau,
        ),
        in_axes=(0, 0),
    )
    update_batch = jax.vmap(
        lambda state, direction, alpha_primal, alpha_dual: apply_primal_dual_step(
            state,
            direction,
            alpha_primal=alpha_primal,
            alpha_dual=alpha_dual,
        ),
        in_axes=(0, 0, 0, 0),
    )
    slack_batch = jax.vmap(
        lambda state: compute_bound_slacks(state, indices, bounds),
        in_axes=0,
    )
    residual_batch = jax.vmap(
        lambda state, nlp, slacks, mu_i: compute_kkt_residual(
            state, nlp, indices, slacks, mu_i
        ),
        in_axes=(0, 0, 0, 0),
    )
    residual_norm_batch = jax.vmap(residual_infinity_norm, in_axes=0)

    def solve_all(state0, params, mus):
        status0 = jnp.full((batch_size,), BATCH_STATUS_ACTIVE, dtype=jnp.int32)
        iteration0 = jnp.full((batch_size,), max_iterations, dtype=jnp.int32)

        def iteration_step(carry, iteration):
            state, status, stopping_iterations = carry
            nlp = evaluate_batch(state, params)
            slacks = slack_batch(state)
            residual = residual_batch(state, nlp, slacks, mus)
            residual_norm = residual_norm_batch(residual)

            active = status == BATCH_STATUS_ACTIVE
            residual_finite = jnp.isfinite(residual_norm) & _state_rows_are_finite(state)
            residual_failure = active & ~residual_finite
            status = jnp.where(
                residual_failure,
                BATCH_STATUS_NUMERICAL_FAILURE,
                status,
            )
            stopping_iterations = jnp.where(
                residual_failure,
                iteration,
                stopping_iterations,
            )

            active = status == BATCH_STATUS_ACTIVE
            converged = active & (residual_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            stopping_iterations = jnp.where(converged, iteration, stopping_iterations)

            newton = solve_batch(state, nlp, mus)
            active = status == BATCH_STATUS_ACTIVE
            direction_finite = _state_rows_are_finite(newton.direction)
            numerical_failure = active & ~direction_finite
            status = jnp.where(
                numerical_failure,
                BATCH_STATUS_NUMERICAL_FAILURE,
                status,
            )
            stopping_iterations = jnp.where(
                numerical_failure,
                iteration,
                stopping_iterations,
            )

            active = status == BATCH_STATUS_ACTIVE
            limits = step_limit_batch(state, newton.direction)
            alpha_primal = step_scale * limits.alpha_primal
            alpha_dual = step_scale * limits.alpha_dual
            too_small = jnp.minimum(alpha_primal, alpha_dual) <= minimum_step
            stalled = active & too_small
            status = jnp.where(stalled, BATCH_STATUS_STALLED, status)
            stopping_iterations = jnp.where(stalled, iteration, stopping_iterations)

            should_update = status == BATCH_STATUS_ACTIVE
            candidate = update_batch(
                state,
                newton.direction,
                alpha_primal,
                alpha_dual,
            )
            candidate_finite = _state_rows_are_finite(candidate)
            update_failure = should_update & ~candidate_finite
            status = jnp.where(
                update_failure,
                BATCH_STATUS_NUMERICAL_FAILURE,
                status,
            )
            stopping_iterations = jnp.where(
                update_failure,
                iteration,
                stopping_iterations,
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

        final_nlp = evaluate_batch(state, params)
        final_slacks = slack_batch(state)
        final_residual = residual_batch(state, final_nlp, final_slacks, mus)
        final_norm = residual_norm_batch(final_residual)

        active = status == BATCH_STATUS_ACTIVE
        final_finite = jnp.isfinite(final_norm) & _state_rows_are_finite(state)
        status = jnp.where(
            active & ~final_finite,
            BATCH_STATUS_NUMERICAL_FAILURE,
            status,
        )
        active = status == BATCH_STATUS_ACTIVE
        final_converged = active & (final_norm <= residual_tolerance)
        status = jnp.where(final_converged, BATCH_STATUS_CONVERGED, status)
        status = jnp.where(
            status == BATCH_STATUS_ACTIVE,
            BATCH_STATUS_MAXIMUM_ITERATIONS,
            status,
        )
        stopping_iterations = jnp.where(
            final_converged,
            max_iterations,
            stopping_iterations,
        )
        full_history = jnp.concatenate((history, final_norm[None, :]), axis=0)
        return BatchedFixedMuResult(
            state=state,
            status_codes=status,
            iterations=stopping_iterations,
            final_residual_inf_norm=final_norm,
            residual_history=full_history,
        )

    runner = jax.jit(solve_all) if jit else solve_all
    return runner(initial_state, parameters, mu_values)
