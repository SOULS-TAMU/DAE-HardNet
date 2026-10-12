"""Small pure-JAX semismooth Newton driver used by the FB projection layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SemismoothNewtonConfig:
    tolerance: float = 1e-9
    max_iterations: int = 100
    regularization: float = 1e-8
    line_search: bool = True
    max_line_search_steps: int = 12
    line_search_factor: float = 0.5

    def __post_init__(self):
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        if self.max_line_search_steps < 1:
            raise ValueError("max_line_search_steps must be positive")
        if not 0.0 < self.line_search_factor < 1.0:
            raise ValueError("line_search_factor must satisfy 0 < factor < 1")


def _inf_norm(value):
    return jnp.max(jnp.abs(value))


def semismooth_newton(
    residual_fn: Callable,
    initial,
    *args,
    config: SemismoothNewtonConfig = SemismoothNewtonConfig(),
):
    """Run a fixed-shape semismooth Newton solve.

    The returned tuple is ``(solution, iterations, residual_inf_norm, finite)``.
    All branching is expressed with JAX control flow so this function is safe to
    use inside one larger jitted graph.
    """
    z0 = jnp.asarray(initial)
    identity = jnp.eye(z0.shape[0], dtype=z0.dtype)

    def body(index, carry):
        z, iterations, done = carry
        residual = residual_fn(z, *args)
        residual_norm = _inf_norm(residual)
        finite = jnp.all(jnp.isfinite(z)) & jnp.all(jnp.isfinite(residual))
        active = (~done) & finite & (residual_norm > config.tolerance)
        jacobian = jax.jacfwd(residual_fn, argnums=0)(z, *args)
        system = jacobian + config.regularization * identity
        step = jnp.linalg.solve(system, -residual)
        step = jnp.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)

        if config.line_search:
            def ls_body(_ls_index, ls_carry):
                alpha, best_z, best_norm, accepted = ls_carry
                candidate = z + alpha * step
                candidate_residual = residual_fn(candidate, *args)
                candidate_norm = _inf_norm(candidate_residual)
                candidate_finite = jnp.all(jnp.isfinite(candidate_residual)) & jnp.all(jnp.isfinite(candidate))
                improve = (~accepted) & candidate_finite & (candidate_norm < best_norm)
                best_z = jnp.where(improve, candidate, best_z)
                best_norm = jnp.where(improve, candidate_norm, best_norm)
                accepted = accepted | improve
                alpha = alpha * config.line_search_factor
                return alpha, best_z, best_norm, accepted

            _, proposed, _, accepted = jax.lax.fori_loop(
                0,
                config.max_line_search_steps,
                ls_body,
                (jnp.asarray(1.0, z.dtype), z, residual_norm, jnp.asarray(False)),
            )
            proposed = jnp.where(accepted, proposed, z + config.line_search_factor ** config.max_line_search_steps * step)
        else:
            proposed = z + step

        z = jnp.where(active, proposed, z)
        iterations = iterations + active.astype(jnp.int32)
        next_residual = residual_fn(z, *args)
        next_done = done | (~finite) | (_inf_norm(next_residual) <= config.tolerance)
        return z, iterations, next_done

    z, iterations, _ = jax.lax.fori_loop(
        0,
        config.max_iterations,
        body,
        (z0, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)),
    )
    residual = residual_fn(z, *args)
    residual_norm = _inf_norm(residual)
    finite = jnp.all(jnp.isfinite(z)) & jnp.all(jnp.isfinite(residual))
    return z, iterations, residual_norm, finite
