"""Closed-form projection layers."""
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True, eq=False)
class BoxProjection:
    lower: object
    upper: object
    def prepare(self, **_kwargs): return self
    def __call__(self, y_hat, parameters=None):
        del parameters
        return jnp.clip(y_hat, self.lower, self.upper)
    def with_info(self, y_hat, parameters=None):
        y = self(y_hat, parameters)
        batch = y.shape[0] if y.ndim > 1 else 1
        return y, {'forward_converged': jnp.ones((batch,), dtype=bool), 'gradient_eligible': jnp.ones((batch,), dtype=bool)}

@dataclass(frozen=True, eq=False)
class AffineProjection:
    A: object
    b: object
    regularization: float = 0.0
    def prepare(self, **_kwargs): return self
    def __call__(self, y_hat, parameters=None):
        del parameters
        A=jnp.asarray(self.A); b=jnp.asarray(self.b)
        rhs = y_hat @ A.T - b
        gram = A @ A.T + self.regularization * jnp.eye(A.shape[0], dtype=A.dtype)
        multipliers = jnp.linalg.solve(gram, rhs.T).T
        return y_hat - multipliers @ A
    def with_info(self, y_hat, parameters=None):
        y=self(y_hat, parameters); r=jnp.max(jnp.abs(y @ jnp.asarray(self.A).T-jnp.asarray(self.b)),axis=-1)
        return y, {'forward_converged': jnp.isfinite(r), 'forward_residual_inf_norm': r, 'gradient_eligible': jnp.isfinite(r)}
