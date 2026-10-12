from dataclasses import dataclass
from .base import Condition

@dataclass(frozen=True)
class InitialCondition(Condition):
    kind: str = "initial"
