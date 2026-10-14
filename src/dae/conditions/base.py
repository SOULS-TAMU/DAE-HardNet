from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Condition:
    coordinate_index: int
    location: float
    target: Callable
    kind: str = "dirichlet"
