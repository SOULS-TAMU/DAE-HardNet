"""Pure JAX fraction-to-the-boundary and state-update kernels."""

from __future__ import annotations

import jax.numpy as jnp

from .bounds import compute_bound_slacks
from .types import (
    BoundIndices,
    BoundValues,
    FractionToBoundarySteps,
    PrimalDualState,
)


def fraction_to_boundary(values, directions, tau: float):
    """Largest step in ``[0, 1]`` retaining a ``1 - tau`` fraction."""
    moving = directions < 0.0
    safe_denominator = jnp.where(moving, directions, -jnp.ones_like(directions))
    candidates = jnp.where(
        moving,
        -tau * values / safe_denominator,
        jnp.inf,
    )
    return jnp.minimum(1.0, jnp.min(candidates, initial=jnp.inf))


def compute_fraction_to_boundary_steps(
    state: PrimalDualState,
    direction: PrimalDualState,
    indices: BoundIndices,
    bounds: BoundValues,
    tau: float,
) -> FractionToBoundarySteps:
    slacks = compute_bound_slacks(state, indices, bounds)
    return FractionToBoundarySteps(
        x_lower=fraction_to_boundary(
            slacks.x_lower,
            direction.x[indices.x_lower],
            tau,
        ),
        x_upper=fraction_to_boundary(
            slacks.x_upper,
            -direction.x[indices.x_upper],
            tau,
        ),
        s_lower=fraction_to_boundary(
            slacks.s_lower,
            direction.s[indices.s_lower],
            tau,
        ),
        s_upper=fraction_to_boundary(
            slacks.s_upper,
            -direction.s[indices.s_upper],
            tau,
        ),
        z_L=fraction_to_boundary(state.z_L, direction.z_L, tau),
        z_U=fraction_to_boundary(state.z_U, direction.z_U, tau),
        v_L=fraction_to_boundary(state.v_L, direction.v_L, tau),
        v_U=fraction_to_boundary(state.v_U, direction.v_U, tau),
    )


def apply_primal_dual_step(
    state: PrimalDualState,
    direction: PrimalDualState,
    *,
    alpha_primal,
    alpha_dual,
    alpha_equality=None,
) -> PrimalDualState:
    if alpha_equality is None:
        alpha_equality = alpha_dual

    return PrimalDualState(
        x=state.x + alpha_primal * direction.x,
        s=state.s + alpha_primal * direction.s,
        y_c=state.y_c + alpha_equality * direction.y_c,
        y_d=state.y_d + alpha_equality * direction.y_d,
        z_L=state.z_L + alpha_dual * direction.z_L,
        z_U=state.z_U + alpha_dual * direction.z_U,
        v_L=state.v_L + alpha_dual * direction.v_L,
        v_U=state.v_U + alpha_dual * direction.v_U,
    )
