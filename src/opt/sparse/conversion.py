"""Conversions between the NumPy reference containers and JAX pytrees."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..solvers.bounds import BoundIndices as NumpyBoundIndices
from ..solvers.bounds import BoundSlacks as NumpyBoundSlacks
from ..solvers.bounds import BoundValues as NumpyBoundValues
from ..solvers.residuals import KKTResidual as NumpyKKTResidual
from ..solvers.residuals import NLPQuantities as NumpyNLPQuantities
from ..solvers.state import PrimalDualState as NumpyPrimalDualState
from .types import (
    BoundIndices,
    BoundSlacks,
    BoundValues,
    KKTResidual,
    NLPQuantities,
    PrimalDualState,
)


def state_from_numpy(state: NumpyPrimalDualState) -> PrimalDualState:
    return PrimalDualState(*(jnp.asarray(block) for block in state.blocks()))


def state_to_numpy(state: PrimalDualState) -> NumpyPrimalDualState:
    return NumpyPrimalDualState(*(np.asarray(block) for block in state.blocks()))


def indices_from_numpy(indices: NumpyBoundIndices) -> BoundIndices:
    return BoundIndices(
        *(jnp.asarray(block, dtype=jnp.int32) for _, block in indices.named_blocks())
    )


def bounds_from_numpy(bounds: NumpyBoundValues) -> BoundValues:
    return BoundValues(
        jnp.asarray(bounds.x_lower),
        jnp.asarray(bounds.x_upper),
        jnp.asarray(bounds.s_lower),
        jnp.asarray(bounds.s_upper),
    )


def slacks_from_numpy(slacks: NumpyBoundSlacks) -> BoundSlacks:
    return BoundSlacks(
        jnp.asarray(slacks.x_lower),
        jnp.asarray(slacks.x_upper),
        jnp.asarray(slacks.s_lower),
        jnp.asarray(slacks.s_upper),
    )


def nlp_from_numpy(nlp: NumpyNLPQuantities) -> NLPQuantities:
    objective = (
        None
        if nlp.objective_value is None
        else jnp.asarray(nlp.objective_value)
    )
    return NLPQuantities(
        grad_f=jnp.asarray(nlp.grad_f),
        c=jnp.asarray(nlp.c),
        d=jnp.asarray(nlp.d),
        jac_c=jnp.asarray(nlp.jac_c),
        jac_d=jnp.asarray(nlp.jac_d),
        hessian_lagrangian=jnp.asarray(nlp.hessian_lagrangian),
        objective_value=objective,
    )


def residual_from_numpy(residual: NumpyKKTResidual) -> KKTResidual:
    return KKTResidual(*(jnp.asarray(block) for block in residual.blocks()))
