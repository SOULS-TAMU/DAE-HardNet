"""Canonical optimization-problem data structures."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class CanonicalProblem:
    parameter_size: int
    variable_size: int
    equality_size: int
    inequality_size: int
    objective: Callable
    equalities: Callable
    inequalities: Callable
    equality_sparsity: Any = None
    inequality_sparsity: Any = None
    metadata: Any = None

@dataclass(frozen=True)
class CanonicalQP(CanonicalProblem):
    P: Any = None
    q0: Any = None
    q_parameter_map: Any = None
    A_eq: Any = None
    b_eq0: Any = None
    b_eq_parameter_map: Any = None
    A_ineq: Any = None
    b_ineq0: Any = None
    b_ineq_parameter_map: Any = None
