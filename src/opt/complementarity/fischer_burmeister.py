"""Fischer--Burmeister complementarity mappings."""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _fischer_burmeister(a, b, smoothing):
    return jnp.sqrt(a * a + b * b + 2.0 * smoothing * smoothing) - a - b


@_fischer_burmeister.defjvp
def _fischer_burmeister_jvp(primals, tangents):
    a, b, smoothing = primals
    da_tangent, db_tangent, smoothing_tangent = tangents
    radius = jnp.sqrt(a * a + b * b + 2.0 * smoothing * smoothing)
    safe = radius > 0.0
    # At the nonsmooth origin choose the symmetric generalized-Jacobian
    # element (-1, -1). This keeps Newton and implicit-adjoint traces finite.
    partial_a = jnp.where(safe, a / radius - 1.0, -1.0)
    partial_b = jnp.where(safe, b / radius - 1.0, -1.0)
    partial_smoothing = jnp.where(
        safe, 2.0 * smoothing / radius, jnp.asarray(0.0, radius.dtype)
    )
    primal = _fischer_burmeister(a, b, smoothing)
    tangent = (
        partial_a * da_tangent
        + partial_b * db_tangent
        + partial_smoothing * smoothing_tangent
    )
    return primal, tangent


def fischer_burmeister(a, b, *, smoothing: float = 0.0):
    """Return the (optionally smoothed) Fischer--Burmeister mapping.

    ``phi(a, b) == 0`` represents ``a >= 0``, ``b >= 0`` and ``a*b == 0``
    when ``smoothing == 0``. Its custom JVP selects a finite generalized
    derivative at the nonsmooth origin.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    smoothing = jnp.asarray(smoothing, dtype=jnp.result_type(a, b))
    return _fischer_burmeister(a, b, smoothing)


def fischer_burmeister_partials(
    a,
    b,
    *,
    smoothing: float = 0.0,
    origin_value: float = 0.0,
):
    """Return one valid generalized-Jacobian selection ``(dphi/da, dphi/db)``."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    smoothing = jnp.asarray(smoothing, dtype=jnp.result_type(a, b))
    radius = jnp.sqrt(a * a + b * b + 2.0 * smoothing * smoothing)
    safe = radius > 0.0
    origin = jnp.asarray(origin_value, dtype=radius.dtype)
    da = jnp.where(safe, a / radius - 1.0, origin - 1.0)
    db = jnp.where(safe, b / radius - 1.0, origin - 1.0)
    return da, db
