"""Pure JAX filter metrics.

The mutable nondominance filter and Python backtracking loop remain in
``src.filter`` and ``src.line_search``.  This module ports the differentiable
array calculations used by those orchestration layers.
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import (
    BoundIndices,
    BoundSlacks,
    NLPQuantities,
    PrimalDualState,
)


def compute_constraint_violation(
    state: PrimalDualState,
    nlp: NLPQuantities,
):
    return (
        jnp.sum(jnp.abs(nlp.c))
        + jnp.sum(jnp.abs(nlp.d - state.s))
    )


def compute_barrier_objective(
    nlp: NLPQuantities,
    slacks: BoundSlacks,
    mu: float,
):
    if nlp.objective_value is None:
        raise ValueError(
            "objective_value is required for the JAX barrier objective."
        )
    barrier_sum = sum(jnp.sum(jnp.log(block)) for block in slacks.blocks())
    return nlp.objective_value - mu * barrier_sum


def compute_barrier_gradient(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    mu: float,
):
    grad_x = nlp.grad_f
    grad_s = jnp.zeros_like(state.s)

    grad_x = grad_x.at[indices.x_lower].add(-mu / slacks.x_lower)
    grad_x = grad_x.at[indices.x_upper].add(mu / slacks.x_upper)
    grad_s = grad_s.at[indices.s_lower].add(-mu / slacks.s_lower)
    grad_s = grad_s.at[indices.s_upper].add(mu / slacks.s_upper)
    return grad_x, grad_s


def compute_barrier_directional_derivative(
    state: PrimalDualState,
    direction: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    mu: float,
):
    grad_x, grad_s = compute_barrier_gradient(
        state,
        nlp,
        indices,
        slacks,
        mu,
    )
    return grad_x @ direction.x + grad_s @ direction.s
