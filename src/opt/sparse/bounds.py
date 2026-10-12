"""Pure JAX bound-slack calculations."""

from __future__ import annotations

from .types import BoundIndices, BoundSlacks, BoundValues, PrimalDualState


def compute_bound_slacks(
    state: PrimalDualState,
    indices: BoundIndices,
    bounds: BoundValues,
) -> BoundSlacks:
    """Compute distances from ``x`` and ``s`` to all finite bounds."""
    return BoundSlacks(
        x_lower=state.x[indices.x_lower] - bounds.x_lower,
        x_upper=bounds.x_upper - state.x[indices.x_upper],
        s_lower=state.s[indices.s_lower] - bounds.s_lower,
        s_upper=bounds.s_upper - state.s[indices.s_upper],
    )
