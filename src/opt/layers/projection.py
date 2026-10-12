"""Compatibility imports for existing model and checkpoint code."""
from ..types import ProjectionInfo
from ..linear_equality.solver import LinearEqualityProjection, PreparedLinearEqualityProjection
from ..fischer_burmeister.solver import FischerBurmeisterProjection, PreparedFischerBurmeisterProjection
from ..primal_dual_interior_point.solver import SparseProjection, PreparedProjection
ExplicitLinearProjection = LinearEqualityProjection

from ..fischer_burmeister.solver import FB_STATUS_CONVERGED, FB_STATUS_MAX_ITERATIONS, FB_STATUS_NONFINITE
