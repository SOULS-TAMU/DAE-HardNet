"""Differentiable optimization layers."""
from .projection import (
    ExplicitLinearProjection,
    FischerBurmeisterProjection,
    LinearEqualityProjection,
    PreparedFischerBurmeisterProjection,
    PreparedLinearEqualityProjection,
    PreparedProjection,
    ProjectionInfo,
    SparseProjection,
)
from .simple import BoxProjection, AffineProjection

PolyhedralProjection = SparseProjection
NonlinearProjection = SparseProjection

__all__ = [
    "SparseProjection",
    "PreparedProjection",
    "ProjectionInfo",
    "FischerBurmeisterProjection",
    "PreparedFischerBurmeisterProjection",
    "LinearEqualityProjection",
    "ExplicitLinearProjection",
    "PreparedLinearEqualityProjection",
    "BoxProjection",
    "AffineProjection",
    "PolyhedralProjection",
    "NonlinearProjection",
]
