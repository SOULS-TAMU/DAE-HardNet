"""Fraction-to-the-boundary step calculations for the NumPy IPOPT solver.

A primal-dual interior-point method must keep every bound slack and every bound
multiplier strictly positive.  Given a current positive vector ``v`` and a
Newton direction ``dv``, the fraction-to-the-boundary rule chooses

    alpha = min(1, min(-tau * v[i] / dv[i] for dv[i] < 0)),

where ``0 < tau < 1``.

At that step length,

    v[i] + alpha * dv[i] >= (1 - tau) * v[i] > 0

for every component that moves toward the boundary.

The primal step protects the bound slacks of ``x`` and ``s``.  The dual step
protects the bound multipliers ``z_L``, ``z_U``, ``v_L``, and ``v_U``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .bounds import BoundIndices, BoundValues, compute_bound_slacks
from .state import FloatArray, PrimalDualState


def _as_float_vector(value: FloatArray, name: str) -> FloatArray:
    """Return ``value`` as a finite one-dimensional float64 array."""
    array = np.asarray(value, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; received shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        bad_indices = np.flatnonzero(~np.isfinite(array))
        raise ValueError(
            f"{name} contains non-finite values at indices "
            f"{bad_indices.tolist()}."
        )

    return array


def _validate_tau(tau: float) -> float:
    """Validate and return the fraction-to-the-boundary parameter."""
    tau = float(tau)

    if not np.isfinite(tau):
        raise ValueError(f"tau must be finite; received {tau}.")

    if not 0.0 < tau < 1.0:
        raise ValueError(
            "tau must satisfy 0 < tau < 1 so the trial point remains "
            f"strictly interior; received {tau}."
        )

    return tau


def fraction_to_boundary(
    values: FloatArray,
    directions: FloatArray,
    tau: float,
) -> float:
    """Return the largest safe step in ``[0, 1]`` for positive variables.

    Parameters
    ----------
    values
        Current strictly positive quantities.
    directions
        Newton directions for those quantities.
    tau
        Fraction-to-the-boundary parameter satisfying ``0 < tau < 1``.

    Returns
    -------
    float
        The maximum step no greater than one that preserves a positive
        fraction of every current value.
    """
    values = _as_float_vector(values, "values")
    directions = _as_float_vector(directions, "directions")
    tau = _validate_tau(tau)

    if values.shape != directions.shape:
        raise ValueError(
            "values and directions must have the same shape; "
            f"received {values.shape} and {directions.shape}."
        )

    bad_values = np.flatnonzero(values <= 0.0)
    if bad_values.size:
        raise ValueError(
            "values must be strictly positive; violations at indices "
            f"{bad_values.tolist()} with values "
            f"{values[bad_values].tolist()}."
        )

    moving_toward_boundary = directions < 0.0

    if not np.any(moving_toward_boundary):
        return 1.0

    candidate_steps = (
        -tau
        * values[moving_toward_boundary]
        / directions[moving_toward_boundary]
    )

    alpha = min(1.0, float(np.min(candidate_steps)))

    # Numerical roundoff should not produce a negative step, but clipping makes
    # the public contract explicit.
    return max(0.0, alpha)


@dataclass(frozen=True)
class FractionToBoundarySteps:
    """Blockwise and aggregate primal-dual step limits."""

    x_lower: float
    x_upper: float
    s_lower: float
    s_upper: float

    z_L: float
    z_U: float
    v_L: float
    v_U: float

    def __post_init__(self) -> None:
        for name in (
            "x_lower",
            "x_upper",
            "s_lower",
            "s_upper",
            "z_L",
            "z_U",
            "v_L",
            "v_U",
        ):
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} step limit must be finite; received {value}."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} step limit must lie in [0, 1]; received {value}."
                )

            object.__setattr__(self, name, value)

    @property
    def alpha_primal(self) -> float:
        """Largest common step preserving all primal bound slacks."""
        return min(
            self.x_lower,
            self.x_upper,
            self.s_lower,
            self.s_upper,
        )

    @property
    def alpha_dual(self) -> float:
        """Largest common step preserving all bound multipliers."""
        return min(
            self.z_L,
            self.z_U,
            self.v_L,
            self.v_U,
        )

    def primal_limits(self) -> dict[str, float]:
        """Return the four primal block limits."""
        return {
            "x_lower": self.x_lower,
            "x_upper": self.x_upper,
            "s_lower": self.s_lower,
            "s_upper": self.s_upper,
        }

    def dual_limits(self) -> dict[str, float]:
        """Return the four dual block limits."""
        return {
            "z_L": self.z_L,
            "z_U": self.z_U,
            "v_L": self.v_L,
            "v_U": self.v_U,
        }

    def named_limits(self) -> Iterator[tuple[str, float]]:
        """Iterate over all blockwise limits in canonical order."""
        return iter(
            (
                ("x_lower", self.x_lower),
                ("x_upper", self.x_upper),
                ("s_lower", self.s_lower),
                ("s_upper", self.s_upper),
                ("z_L", self.z_L),
                ("z_U", self.z_U),
                ("v_L", self.v_L),
                ("v_U", self.v_U),
            )
        )


def compute_fraction_to_boundary_steps(
    state: PrimalDualState,
    direction: PrimalDualState,
    indices: BoundIndices,
    bounds: BoundValues,
    tau: float,
) -> FractionToBoundarySteps:
    """Compute IPOPT-style primal and dual fraction-to-boundary limits.

    For the primal variables, the directions of the bound slacks are

    .. code-block:: text

        lower x slack:  +dx[index]
        upper x slack:  -dx[index]
        lower s slack:  +ds[index]
        upper s slack:  -ds[index]

    The dual limits are computed directly from the bound multipliers and their
    Newton directions.
    """
    tau = _validate_tau(tau)

    state.validate_finite()
    direction.validate_finite()

    if state.block_sizes != direction.block_sizes:
        raise ValueError(
            "State and direction block sizes must match: "
            f"{state.block_sizes} != {direction.block_sizes}."
        )

    indices.validate_against_state(state)
    bounds.validate_against_indices(indices)

    slacks = compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
    )

    return FractionToBoundarySteps(
        x_lower=fraction_to_boundary(
            slacks.x_lower,
            direction.x[indices.x_lower],
            tau,
        ),
        x_upper=fraction_to_boundary(
            slacks.x_upper,
            -direction.x[indices.x_upper],
            tau,
        ),
        s_lower=fraction_to_boundary(
            slacks.s_lower,
            direction.s[indices.s_lower],
            tau,
        ),
        s_upper=fraction_to_boundary(
            slacks.s_upper,
            -direction.s[indices.s_upper],
            tau,
        ),
        z_L=fraction_to_boundary(
            state.z_L,
            direction.z_L,
            tau,
        ),
        z_U=fraction_to_boundary(
            state.z_U,
            direction.z_U,
            tau,
        ),
        v_L=fraction_to_boundary(
            state.v_L,
            direction.v_L,
            tau,
        ),
        v_U=fraction_to_boundary(
            state.v_U,
            direction.v_U,
            tau,
        ),
    )


def apply_primal_dual_step(
    state: PrimalDualState,
    direction: PrimalDualState,
    *,
    alpha_primal: float,
    alpha_dual: float,
    alpha_equality: float | None = None,
) -> PrimalDualState:
    """Apply primal, bound-dual, and equality-multiplier step lengths.

    ``alpha_primal`` is applied to ``x`` and ``s``.

    ``alpha_dual`` is applied to the bound multipliers ``z_L``, ``z_U``,
    ``v_L``, and ``v_U``.

    ``alpha_equality`` is applied to ``y_c`` and ``y_d``.  When omitted it
    defaults to ``alpha_dual`` for backward compatibility.
    """
    if state.block_sizes != direction.block_sizes:
        raise ValueError(
            "State and direction block sizes must match: "
            f"{state.block_sizes} != {direction.block_sizes}."
        )

    alpha_primal = float(alpha_primal)
    alpha_dual = float(alpha_dual)
    alpha_equality = (
        alpha_dual
        if alpha_equality is None
        else float(alpha_equality)
    )

    for name, value in (
        ("alpha_primal", alpha_primal),
        ("alpha_dual", alpha_dual),
        ("alpha_equality", alpha_equality),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value}.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must lie in [0, 1]; received {value}."
            )

    return PrimalDualState(
        x=state.x + alpha_primal * direction.x,
        s=state.s + alpha_primal * direction.s,
        y_c=state.y_c + alpha_equality * direction.y_c,
        y_d=state.y_d + alpha_equality * direction.y_d,
        z_L=state.z_L + alpha_dual * direction.z_L,
        z_U=state.z_U + alpha_dual * direction.z_U,
        v_L=state.v_L + alpha_dual * direction.v_L,
        v_U=state.v_U + alpha_dual * direction.v_U,
    )
