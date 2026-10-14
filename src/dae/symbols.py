"""Typed symbols used by continuous DAE definitions.

The symbols are lightweight immutable metadata. Runtime JAX code uses integer
indices and arrays produced by :mod:`daehn.dae.compiler`.
"""
from __future__ import annotations
from dataclasses import dataclass


def _validate_name(name: str) -> str:
    value = str(name).strip()
    if not value or not value.isidentifier():
        raise ValueError(f"invalid symbol name: {name!r}")
    return value


@dataclass(frozen=True, order=True)
class Coordinate:
    name: str
    lower: float | None = None
    upper: float | None = None
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "name", _validate_name(self.name))
        if self.scale <= 0:
            raise ValueError("coordinate scale must be positive")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("coordinate lower bound must be smaller than upper bound")


@dataclass(frozen=True, order=True)
class DifferentialState:
    name: str
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "name", _validate_name(self.name))
        if self.scale <= 0:
            raise ValueError("state scale must be positive")


@dataclass(frozen=True, order=True)
class AlgebraicState:
    name: str
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "name", _validate_name(self.name))
        if self.scale <= 0:
            raise ValueError("state scale must be positive")


@dataclass(frozen=True, order=True)
class Parameter:
    name: str
    scale: float = 1.0
    trainable: bool = False

    def __post_init__(self):
        object.__setattr__(self, "name", _validate_name(self.name))
        if self.scale <= 0:
            raise ValueError("parameter scale must be positive")
