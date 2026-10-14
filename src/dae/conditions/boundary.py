from dataclasses import dataclass
from .base import Condition

@dataclass(frozen=True)
class BoundaryCondition(Condition):
    kind: str = "dirichlet"
