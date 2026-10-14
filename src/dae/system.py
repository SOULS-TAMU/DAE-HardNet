"""Continuous differential-algebraic system definitions."""
from __future__ import annotations
from dataclasses import dataclass
from .symbols import Coordinate, DifferentialState, AlgebraicState, Parameter
from .derivatives import DerivativeSpec, derivative_closure
from .equations import EquationBlock


@dataclass(frozen=True)
class DAESystem:
    coordinates: tuple[Coordinate, ...]
    differential_states: tuple[DifferentialState, ...]
    algebraic_states: tuple[AlgebraicState, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    derivatives: tuple[DerivativeSpec, ...] = ()
    differential_equations: EquationBlock = EquationBlock(name="differential")
    algebraic_equalities: EquationBlock = EquationBlock(name="algebraic_equalities")
    algebraic_inequalities: EquationBlock = EquationBlock(name="algebraic_inequalities")
    name: str = "dae_system"

    def __post_init__(self):
        if not self.coordinates:
            raise ValueError("a DAE system requires at least one coordinate")
        if not self.differential_states and not self.algebraic_states:
            raise ValueError("a DAE system requires at least one state")
        all_names = [x.name for x in self.coordinates]
        all_names += [x.name for x in self.differential_states]
        all_names += [x.name for x in self.algebraic_states]
        all_names += [x.name for x in self.parameters]
        if len(set(all_names)) != len(all_names):
            raise ValueError("coordinate, state, and parameter names must be unique")
        valid_states = set(self.differential_states)
        for derivative in self.derivatives:
            if derivative.state not in valid_states:
                raise ValueError("derivatives may only reference differential states")
            if len(derivative.multi_index) != len(self.coordinates):
                raise ValueError("derivative dimension does not match coordinates")

    @property
    def derivative_closure(self) -> tuple[DerivativeSpec, ...]:
        return derivative_closure(self.derivatives, coordinate_count=len(self.coordinates))

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in (*self.differential_states, *self.algebraic_states))

    @property
    def equality_size(self) -> int:
        return self.differential_equations.size + self.algebraic_equalities.size
