"""Equation blocks for DAE systems."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import jax.numpy as jnp

ResidualCallable = Callable[[Any, Any, Any, Any], Any]


def _empty(coords, states, derivatives, parameters):
    del coords, states, derivatives, parameters
    return jnp.empty((0,), dtype=jnp.float64)


@dataclass(frozen=True)
class EquationBlock:
    """A vector-valued residual block.

    Functions use the signature ``(coordinates, states, derivatives,
    parameters) -> residual``. ``dependencies`` contains projection-variable
    indices per row when exact sparsity is known; ``None`` requests dense
    support.
    """

    function: ResidualCallable = _empty
    size: int = 0
    dependencies: tuple[tuple[int, ...], ...] | None = None
    affine_in_projection_variables: bool = False
    name: str = "equations"

    def __post_init__(self):
        if self.size < 0:
            raise ValueError("equation block size must be nonnegative")
        if not callable(self.function):
            raise TypeError("equation block function must be callable")
        if self.dependencies is not None and len(self.dependencies) != self.size:
            raise ValueError("dependency row count must match equation block size")
