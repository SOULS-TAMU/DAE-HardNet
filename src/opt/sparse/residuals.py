"""Pure JAX assembly of primal-dual KKT residuals."""

from __future__ import annotations

import jax.numpy as jnp

from .types import (
    BoundIndices,
    BoundSlacks,
    KKTResidual,
    NLPQuantities,
    PrimalDualState,
)


def compute_kkt_residual(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    mu: float,
) -> KKTResidual:
    """Assemble the same eight residual blocks as the NumPy reference."""
    r_x = (
        nlp.grad_f
        + nlp.jac_c.T @ state.y_c
        + nlp.jac_d.T @ state.y_d
    )
    r_x = r_x.at[indices.x_lower].add(-state.z_L)
    r_x = r_x.at[indices.x_upper].add(state.z_U)

    r_s = -state.y_d
    r_s = r_s.at[indices.s_lower].add(-state.v_L)
    r_s = r_s.at[indices.s_upper].add(state.v_U)

    return KKTResidual(
        x=r_x,
        s=r_s,
        y_c=nlp.c,
        y_d=nlp.d - state.s,
        z_L=slacks.x_lower * state.z_L - mu,
        z_U=slacks.x_upper * state.z_U - mu,
        v_L=slacks.s_lower * state.v_L - mu,
        v_U=slacks.s_upper * state.v_U - mu,
    )


def residual_infinity_norm(residual: KKTResidual):
    """Return the infinity norm of a complete KKT residual."""
    return jnp.max(jnp.abs(residual.flatten()), initial=0.0)


def residual_squared_l2_norm(residual: KKTResidual):
    """Return the squared Euclidean norm of a complete KKT residual."""
    flat = residual.flatten()
    return flat @ flat
