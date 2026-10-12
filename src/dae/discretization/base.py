"""Base types for local DAE discretization strategies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompiledDiscretization:
    name: str
    offsets: Any
    derivatives: tuple
    derivative_physical_scales: Any
    reconstruct_differential_states: Any
    estimate_derivatives: Any
    signature: str


class DiscretizationStrategy:
    def compile(self, system, scaling):
        raise NotImplementedError
