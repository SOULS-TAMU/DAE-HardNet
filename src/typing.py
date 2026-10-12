"""Shared type aliases for DAEHN's functional JAX APIs."""
from __future__ import annotations
from typing import Any, Callable, NamedTuple
import jax

Array = jax.Array
PyTree = Any
Shape = tuple[int, ...]
InitFn = Callable[[Array, Shape], tuple[Shape, PyTree]]
ApplyFn = Callable[..., Array]

class Layer(NamedTuple):
    init: InitFn
    apply: ApplyFn
