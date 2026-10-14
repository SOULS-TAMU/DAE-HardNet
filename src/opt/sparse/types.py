"""JAX-compatible immutable containers for the primal-dual kernels.

The containers are ``NamedTuple`` pytrees, so they can be passed directly to
``jax.jit``, ``jax.vmap``, and differentiation transforms.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class PrimalDualState(NamedTuple):
    x: Array
    s: Array
    y_c: Array
    y_d: Array
    z_L: Array
    z_U: Array
    v_L: Array
    v_U: Array

    def blocks(self) -> tuple[Array, ...]:
        return tuple(self)

    def flatten(self) -> Array:
        return jnp.concatenate(self.blocks())

    def axpy(self, alpha: Array | float, direction: "PrimalDualState") -> "PrimalDualState":
        return PrimalDualState(
            *(block + alpha * delta for block, delta in zip(self, direction, strict=True))
        )


class BoundIndices(NamedTuple):
    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array


class BoundValues(NamedTuple):
    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array


class BoundSlacks(NamedTuple):
    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array

    def blocks(self) -> tuple[Array, ...]:
        return tuple(self)


class NLPQuantities(NamedTuple):
    grad_f: Array
    c: Array
    d: Array
    jac_c: Array
    jac_d: Array
    hessian_lagrangian: Array
    objective_value: Array | None = None


class KKTResidual(NamedTuple):
    x: Array
    s: Array
    y_c: Array
    y_d: Array
    z_L: Array
    z_U: Array
    v_L: Array
    v_U: Array

    def blocks(self) -> tuple[Array, ...]:
        return tuple(self)

    def flatten(self) -> Array:
        return jnp.concatenate(self.blocks())


class NewtonRegularization(NamedTuple):
    delta_x: float = 0.0
    delta_s: float = 0.0
    delta_c: float = 0.0
    delta_d: float = 0.0


class ReducedKKTSystem(NamedTuple):
    matrix: Array
    rhs: Array
    sigma_x: Array
    sigma_s: Array


class NewtonSolveResult(NamedTuple):
    direction: PrimalDualState
    residual: KKTResidual
    slacks: BoundSlacks
    reduced_system: ReducedKKTSystem
    reduced_solution: Array
    reduced_residual_inf_norm: Array
    full_residual_inf_norm: Array


class FractionToBoundarySteps(NamedTuple):
    x_lower: Array
    x_upper: Array
    s_lower: Array
    s_upper: Array
    z_L: Array
    z_U: Array
    v_L: Array
    v_U: Array

    @property
    def alpha_primal(self) -> Array:
        return jnp.minimum(
            jnp.minimum(self.x_lower, self.x_upper),
            jnp.minimum(self.s_lower, self.s_upper),
        )

    @property
    def alpha_dual(self) -> Array:
        return jnp.minimum(
            jnp.minimum(self.z_L, self.z_U),
            jnp.minimum(self.v_L, self.v_U),
        )
