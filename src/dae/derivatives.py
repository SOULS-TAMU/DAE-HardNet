"""Derivative metadata and static derivative utilities."""
from __future__ import annotations
from dataclasses import dataclass
from math import factorial
from typing import Iterable
from .symbols import DifferentialState


@dataclass(frozen=True, order=True)
class DerivativeSpec:
    """A derivative of a differential state represented by a multi-index."""

    state: DifferentialState
    multi_index: tuple[int, ...]
    physics_required: bool = True

    def __post_init__(self):
        mi = tuple(int(v) for v in self.multi_index)
        if not mi or any(v < 0 for v in mi) or sum(mi) < 1:
            raise ValueError("multi_index must be nonnegative and have positive order")
        object.__setattr__(self, "multi_index", mi)

    @property
    def order(self) -> int:
        return int(sum(self.multi_index))

    @property
    def multi_factorial(self) -> int:
        out = 1
        for value in self.multi_index:
            out *= factorial(value)
        return out

    @property
    def pure_coordinate(self) -> int | None:
        nz = [i for i, value in enumerate(self.multi_index) if value]
        return nz[0] if len(nz) == 1 else None


def derivative_closure(
    derivatives: Iterable[DerivativeSpec], *, coordinate_count: int
) -> tuple[DerivativeSpec, ...]:
    """Return deterministic Taylor-required lower-order derivative closure.

    Pure derivatives add all lower orders on the same coordinate. Mixed
    derivatives are retained as declared; strategy-specific mixed closure can
    be added later without changing the public representation.
    """

    result: dict[tuple[str, tuple[int, ...]], DerivativeSpec] = {}
    for spec in derivatives:
        if len(spec.multi_index) != coordinate_count:
            raise ValueError("derivative multi_index dimension does not match coordinates")
        key = (spec.state.name, spec.multi_index)
        result[key] = spec
        coord = spec.pure_coordinate
        if coord is not None:
            for order in range(1, spec.order):
                mi = [0] * coordinate_count
                mi[coord] = order
                lower = DerivativeSpec(spec.state, tuple(mi), physics_required=False)
                result.setdefault((spec.state.name, lower.multi_index), lower)
    return tuple(sorted(result.values(), key=lambda s: (s.state.name, s.order, s.multi_index)))
