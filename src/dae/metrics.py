"""Metrics evaluated on the original continuous DAE equations."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def physical_residual_blocks(system, coordinates, states, derivatives, parameters):
    """Evaluate original differential/algebraic equations for a batch.

    This function intentionally bypasses scaled KKT residuals and any internal
    algebraic solver representation. ``states`` and ``derivatives`` must be the
    decoded physical outputs.
    """

    coordinates = jnp.asarray(coordinates)
    states = jnp.asarray(states)
    derivatives = jnp.asarray(derivatives)
    parameters = jnp.asarray(parameters)
    differential = jax.vmap(system.differential_equations.function)(
        coordinates, states, derivatives, parameters
    ).reshape((coordinates.shape[0], -1))
    algebraic = jax.vmap(system.algebraic_equalities.function)(
        coordinates, states, derivatives, parameters
    ).reshape((coordinates.shape[0], -1))
    inequalities = jax.vmap(system.algebraic_inequalities.function)(
        coordinates, states, derivatives, parameters
    ).reshape((coordinates.shape[0], -1))
    return differential, algebraic, inequalities


def physical_violation_metrics(
    system,
    coordinates,
    states,
    derivatives,
    parameters,
    mask=None,
):
    """Return mean-L2 physical violation and per-block diagnostics."""

    differential, algebraic, inequalities = physical_residual_blocks(
        system, coordinates, states, derivatives, parameters
    )
    positive_inequalities = jnp.maximum(inequalities, 0.0)
    combined = jnp.concatenate(
        (differential, algebraic, positive_inequalities), axis=-1
    )
    if mask is None:
        mask = jnp.ones((combined.shape[0],), dtype=jnp.float64)
    else:
        mask = jnp.asarray(mask, dtype=jnp.float64)

    def masked_mean(values):
        return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def mean_norm(values):
        if values.shape[-1] == 0:
            return jnp.asarray(0.0, dtype=states.dtype)
        return masked_mean(jnp.linalg.norm(values, axis=-1))

    return {
        "violation": mean_norm(combined),
        "differential_violation": mean_norm(differential),
        "algebraic_violation": mean_norm(algebraic),
        "inequality_violation": mean_norm(positive_inequalities),
        "differential_residuals": differential,
        "algebraic_residuals": algebraic,
        "inequality_residuals": inequalities,
    }


def dae_metrics(compiled, context, normalized_variables, targets=None):
    """Evaluate decoded states against the original continuous DAE."""

    decoded = jax.vmap(compiled.decode)(context, normalized_variables)
    coordinates, parameters, _center, _neighbors = jax.vmap(
        compiled.context_layout.unpack
    )(context)
    values = physical_violation_metrics(
        compiled.system,
        coordinates,
        decoded["states"],
        decoded["derivatives"],
        parameters,
    )
    output = {
        "physical_equality_inf": jnp.maximum(
            jnp.max(jnp.abs(values["differential_residuals"]))
            if values["differential_residuals"].size
            else jnp.asarray(0.0),
            jnp.max(jnp.abs(values["algebraic_residuals"]))
            if values["algebraic_residuals"].size
            else jnp.asarray(0.0),
        ),
        "physical_inequality_inf": (
            jnp.max(jnp.maximum(values["inequality_residuals"], 0.0))
            if values["inequality_residuals"].size
            else jnp.asarray(0.0)
        ),
        "physical_violation": values["violation"],
    }
    if targets is not None:
        output["state_mse"] = jnp.mean((decoded["states"] - targets) ** 2)
    return output
