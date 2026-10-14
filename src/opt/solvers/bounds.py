"""Bound indexing and slack calculations for the NumPy IPOPT reference solver.

IPOPT stores multipliers only for finite bounds.  This module mirrors that
layout using index arrays rather than explicit selection matrices.

For example, ``indices.x_lower`` represents the columns selected by
``P_xL.T @ x`` in IPOPT notation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .state import FloatArray, PrimalDualState


IntArray = NDArray[np.int64]


def _as_index_vector(value: IntArray, name: str) -> IntArray:
    """Return *value* as a validated one-dimensional int64 array."""
    array = np.asarray(value)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional index array; "
            f"received shape {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{name} must contain integer indices; received dtype {array.dtype}."
        )

    array = array.astype(np.int64, copy=False)

    if np.any(array < 0):
        bad = array[array < 0]
        raise ValueError(
            f"{name} contains negative indices: {bad.tolist()}."
        )

    if np.unique(array).size != array.size:
        raise ValueError(
            f"{name} contains duplicate indices: {array.tolist()}."
        )

    return array


def _as_float_vector(value: FloatArray, name: str) -> FloatArray:
    """Return *value* as a validated one-dimensional float64 array."""
    array = np.asarray(value, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array; received shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        bad_indices = np.flatnonzero(~np.isfinite(array))
        raise ValueError(
            f"{name} contains non-finite values at indices "
            f"{bad_indices.tolist()}."
        )

    return array


@dataclass(frozen=True)
class BoundIndices:
    """Indices of variables having finite lower or upper bounds.

    A variable may appear in both a lower-bound and an upper-bound index array.
    This is how a two-sided finite bound is represented.

    Attributes
    ----------
    x_lower
        Indices of ``x`` with finite lower bounds.
    x_upper
        Indices of ``x`` with finite upper bounds.
    s_lower
        Indices of ``s`` with finite lower bounds.
    s_upper
        Indices of ``s`` with finite upper bounds.
    """

    x_lower: IntArray
    x_upper: IntArray
    s_lower: IntArray
    s_upper: IntArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_lower", _as_index_vector(self.x_lower, "x_lower")
        )
        object.__setattr__(
            self, "x_upper", _as_index_vector(self.x_upper, "x_upper")
        )
        object.__setattr__(
            self, "s_lower", _as_index_vector(self.s_lower, "s_lower")
        )
        object.__setattr__(
            self, "s_upper", _as_index_vector(self.s_upper, "s_upper")
        )

    @classmethod
    def empty(cls) -> "BoundIndices":
        """Create an object with no finite bounds."""
        empty = np.empty(0, dtype=np.int64)
        return cls(
            x_lower=empty.copy(),
            x_upper=empty.copy(),
            s_lower=empty.copy(),
            s_upper=empty.copy(),
        )

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        """Numbers of lower/upper bounds on ``x`` and ``s``."""
        return (
            int(self.x_lower.size),
            int(self.x_upper.size),
            int(self.s_lower.size),
            int(self.s_upper.size),
        )

    def named_blocks(self) -> Iterator[tuple[str, IntArray]]:
        """Iterate over index arrays in canonical order."""
        return iter(
            (
                ("x_lower", self.x_lower),
                ("x_upper", self.x_upper),
                ("s_lower", self.s_lower),
                ("s_upper", self.s_upper),
            )
        )

    def validate_against_state(self, state: PrimalDualState) -> None:
        """Validate index ranges and multiplier dimensions for ``state``."""
        for name, indices, dimension in (
            ("x_lower", self.x_lower, state.x.size),
            ("x_upper", self.x_upper, state.x.size),
            ("s_lower", self.s_lower, state.s.size),
            ("s_upper", self.s_upper, state.s.size),
        ):
            if indices.size and int(indices.max()) >= dimension:
                raise IndexError(
                    f"{name} contains index {int(indices.max())}, but the "
                    f"corresponding vector has length {dimension}."
                )

        expected = self.sizes
        actual = (
            state.z_L.size,
            state.z_U.size,
            state.v_L.size,
            state.v_U.size,
        )

        if actual != expected:
            raise ValueError(
                "Bound multiplier dimensions must match the finite-bound index "
                "arrays. Expected "
                f"(z_L, z_U, v_L, v_U)={expected}, received {actual}."
            )


@dataclass(frozen=True)
class BoundValues:
    """Finite bound values aligned with :class:`BoundIndices`.

    The arrays contain only finite bounds and follow the same order as their
    corresponding index arrays.
    """

    x_lower: FloatArray
    x_upper: FloatArray
    s_lower: FloatArray
    s_upper: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_lower", _as_float_vector(self.x_lower, "x_lower")
        )
        object.__setattr__(
            self, "x_upper", _as_float_vector(self.x_upper, "x_upper")
        )
        object.__setattr__(
            self, "s_lower", _as_float_vector(self.s_lower, "s_lower")
        )
        object.__setattr__(
            self, "s_upper", _as_float_vector(self.s_upper, "s_upper")
        )

    @classmethod
    def empty(cls) -> "BoundValues":
        """Create an object containing no finite bounds."""
        empty = np.empty(0, dtype=np.float64)
        return cls(
            x_lower=empty.copy(),
            x_upper=empty.copy(),
            s_lower=empty.copy(),
            s_upper=empty.copy(),
        )

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        """Lengths of the four finite-bound value arrays."""
        return (
            int(self.x_lower.size),
            int(self.x_upper.size),
            int(self.s_lower.size),
            int(self.s_upper.size),
        )

    def validate_against_indices(self, indices: BoundIndices) -> None:
        """Validate that bound values align with their index arrays."""
        if self.sizes != indices.sizes:
            raise ValueError(
                "Bound value lengths must match the corresponding index lengths. "
                f"Expected {indices.sizes}, received {self.sizes}."
            )

    def validate_two_sided_bounds(self, indices: BoundIndices) -> None:
        """Check that lower bounds do not exceed upper bounds.

        Only variables appearing in both the lower- and upper-bound index arrays
        are checked.
        """
        self.validate_against_indices(indices)

        x_lower_by_index = dict(
            zip(indices.x_lower.tolist(), self.x_lower.tolist(), strict=True)
        )
        x_upper_by_index = dict(
            zip(indices.x_upper.tolist(), self.x_upper.tolist(), strict=True)
        )

        for index in x_lower_by_index.keys() & x_upper_by_index.keys():
            lower = x_lower_by_index[index]
            upper = x_upper_by_index[index]
            if lower > upper:
                raise ValueError(
                    f"x[{index}] has lower bound {lower} greater than "
                    f"upper bound {upper}."
                )

        s_lower_by_index = dict(
            zip(indices.s_lower.tolist(), self.s_lower.tolist(), strict=True)
        )
        s_upper_by_index = dict(
            zip(indices.s_upper.tolist(), self.s_upper.tolist(), strict=True)
        )

        for index in s_lower_by_index.keys() & s_upper_by_index.keys():
            lower = s_lower_by_index[index]
            upper = s_upper_by_index[index]
            if lower > upper:
                raise ValueError(
                    f"s[{index}] has lower bound {lower} greater than "
                    f"upper bound {upper}."
                )


@dataclass(frozen=True)
class BoundSlacks:
    """Distances from the current primal variables to their finite bounds.

    Definitions
    -----------
    ``x_lower``
        ``x[indices.x_lower] - bounds.x_lower``.
    ``x_upper``
        ``bounds.x_upper - x[indices.x_upper]``.
    ``s_lower``
        ``s[indices.s_lower] - bounds.s_lower``.
    ``s_upper``
        ``bounds.s_upper - s[indices.s_upper]``.
    """

    x_lower: FloatArray
    x_upper: FloatArray
    s_lower: FloatArray
    s_upper: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x_lower", _as_float_vector(self.x_lower, "x_lower")
        )
        object.__setattr__(
            self, "x_upper", _as_float_vector(self.x_upper, "x_upper")
        )
        object.__setattr__(
            self, "s_lower", _as_float_vector(self.s_lower, "s_lower")
        )
        object.__setattr__(
            self, "s_upper", _as_float_vector(self.s_upper, "s_upper")
        )

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        """Lengths of the four bound-slack arrays."""
        return (
            int(self.x_lower.size),
            int(self.x_upper.size),
            int(self.s_lower.size),
            int(self.s_upper.size),
        )

    def named_blocks(self) -> Iterator[tuple[str, FloatArray]]:
        """Iterate over slack arrays in canonical order."""
        return iter(
            (
                ("x_lower", self.x_lower),
                ("x_upper", self.x_upper),
                ("s_lower", self.s_lower),
                ("s_upper", self.s_upper),
            )
        )

    def validate_strictly_positive(self, *, tolerance: float = 0.0) -> None:
        """Require all bound slacks to be greater than ``tolerance``.

        The primal-dual barrier equations divide by these values, so they must
        remain strictly positive.
        """
        tolerance = float(tolerance)

        if tolerance < 0.0:
            raise ValueError(
                f"tolerance must be nonnegative; received {tolerance}."
            )

        for name, slack in self.named_blocks():
            bad_indices = np.flatnonzero(slack <= tolerance)
            if bad_indices.size:
                bad_values = slack[bad_indices]
                raise ValueError(
                    f"{name} must be strictly greater than {tolerance}; "
                    f"violations at indices {bad_indices.tolist()} with values "
                    f"{bad_values.tolist()}."
                )

    @property
    def minimum(self) -> float:
        """Smallest finite-bound slack, or positive infinity if none exist."""
        nonempty = [block for _, block in self.named_blocks() if block.size]
        if not nonempty:
            return float("inf")
        return float(min(np.min(block) for block in nonempty))


def compute_bound_slacks(
    state: PrimalDualState,
    indices: BoundIndices,
    bounds: BoundValues,
    *,
    require_strict_interior: bool = True,
    positivity_tolerance: float = 0.0,
) -> BoundSlacks:
    """Compute IPOPT-style bound distances for the current primal state.

    Parameters
    ----------
    state
        Current primal-dual state.
    indices
        Locations of finite bounds.
    bounds
        Finite bound values aligned with ``indices``.
    require_strict_interior
        If true, raise ``ValueError`` when any bound slack is nonpositive.
    positivity_tolerance
        Minimum accepted slack when ``require_strict_interior`` is true.

    Returns
    -------
    BoundSlacks
        Distances from ``x`` and ``s`` to their finite lower and upper bounds.
    """
    indices.validate_against_state(state)
    bounds.validate_against_indices(indices)
    bounds.validate_two_sided_bounds(indices)

    slacks = BoundSlacks(
        x_lower=state.x[indices.x_lower] - bounds.x_lower,
        x_upper=bounds.x_upper - state.x[indices.x_upper],
        s_lower=state.s[indices.s_lower] - bounds.s_lower,
        s_upper=bounds.s_upper - state.s[indices.s_upper],
    )

    if require_strict_interior:
        slacks.validate_strictly_positive(
            tolerance=positivity_tolerance
        )

    return slacks
