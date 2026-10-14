"""Condition losses independent of the projection system."""
from __future__ import annotations
import jax.numpy as jnp


def condition_mse(predictions, targets):
    predictions=jnp.asarray(predictions); targets=jnp.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("condition prediction and target shapes must match")
    return jnp.mean((predictions-targets)**2)
