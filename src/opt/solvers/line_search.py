"""Backtracking filter line search for the fixed-barrier solver.

This is a compact implementation of the main IPOPT filter ideas:

- barrier objective ``phi_mu``,
- one-norm constraint violation ``theta``,
- sufficient reduction with filter margins,
- switching to an Armijo test for objective-type iterations,
- rejection of points dominated by the existing filter,
- geometric backtracking of the primal step.

The implementation intentionally omits watchdog steps, restoration, second
order correction, magic steps, and filter reset heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .bounds import BoundIndices, BoundValues, compute_bound_slacks
from .filter import (
    Filter,
    compute_barrier_directional_derivative,
    compute_barrier_objective,
    compute_constraint_violation,
)
from .residuals import NLPQuantities
from .state import PrimalDualState
from .step import apply_primal_dual_step


NLPQuantityEvaluator = Callable[[PrimalDualState], NLPQuantities]
LineSearchStatus = Literal[
    "accepted",
    "minimum_step",
    "maximum_backtracks",
]


@dataclass(frozen=True)
class FilterLineSearchOptions:
    """Options for the simplified IPOPT-style filter line search."""

    alpha_reduction_factor: float = 0.5
    maximum_backtracks: int = 30
    minimum_alpha: float = 1e-16

    gamma_phi: float = 1e-8
    gamma_theta: float = 1e-5
    eta_phi: float = 1e-8

    delta: float = 1.0
    s_phi: float = 2.3
    s_theta: float = 1.1

    theta_min_factor: float = 1e-4
    theta_max_factor: float = 1e4
    objective_max_increase_orders: float = 5.0

    equality_multiplier_step: Literal[
        "primal",
        "dual",
        "minimum",
        "maximum",
    ] = "primal"

    def __post_init__(self) -> None:
        numeric = (
            "alpha_reduction_factor",
            "minimum_alpha",
            "gamma_phi",
            "gamma_theta",
            "eta_phi",
            "delta",
            "s_phi",
            "s_theta",
            "theta_min_factor",
            "theta_max_factor",
            "objective_max_increase_orders",
        )

        for name in numeric:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value}.")
            object.__setattr__(self, name, value)

        if not 0.0 < self.alpha_reduction_factor < 1.0:
            raise ValueError(
                "alpha_reduction_factor must lie in (0, 1); "
                f"received {self.alpha_reduction_factor}."
            )
        if not isinstance(self.maximum_backtracks, int):
            raise TypeError("maximum_backtracks must be an integer.")
        if self.maximum_backtracks < 0:
            raise ValueError("maximum_backtracks must be nonnegative.")
        if self.minimum_alpha < 0.0:
            raise ValueError("minimum_alpha must be nonnegative.")

        for name in ("gamma_phi", "gamma_theta", "eta_phi"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must lie in (0, 1); received {value}."
                )

        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")
        if self.s_phi <= 1.0:
            raise ValueError("s_phi must be greater than 1.")
        if self.s_theta <= 1.0:
            raise ValueError("s_theta must be greater than 1.")
        if self.theta_min_factor <= 0.0:
            raise ValueError("theta_min_factor must be positive.")
        if self.theta_max_factor <= self.theta_min_factor:
            raise ValueError(
                "theta_max_factor must exceed theta_min_factor."
            )
        if self.objective_max_increase_orders < 1.0:
            raise ValueError(
                "objective_max_increase_orders must be at least 1."
            )


@dataclass(frozen=True)
class LineSearchTrial:
    """Diagnostics for one tested primal step."""

    alpha_primal: float
    barrier_objective: float
    constraint_violation: float
    accepted_by_current: bool
    accepted_by_filter: bool
    objective_type: bool
    armijo_holds: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class FilterLineSearchResult:
    """Result of one backtracking filter line search."""

    status: LineSearchStatus
    state: PrimalDualState
    nlp: NLPQuantities
    alpha_primal: float
    alpha_dual: float
    alpha_equality: float
    backtracks: int
    barrier_objective: float
    constraint_violation: float
    acceptance_type: Literal["f", "h", "none"]
    trials: tuple[LineSearchTrial, ...]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _validate_step(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; received {value}.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; received {value}.")
    return value


def _equality_step(
    mode: str,
    alpha_primal: float,
    alpha_dual: float,
) -> float:
    if mode == "primal":
        return alpha_primal
    if mode == "dual":
        return alpha_dual
    if mode == "minimum":
        return min(alpha_primal, alpha_dual)
    if mode == "maximum":
        return max(alpha_primal, alpha_dual)
    raise ValueError(f"Unsupported equality multiplier step mode: {mode}.")


def _objective_increase_is_acceptable(
    current_phi: float,
    trial_phi: float,
    maximum_orders: float,
) -> bool:
    """Mirror IPOPT's guard against explosive barrier-objective growth."""
    if trial_phi <= current_phi:
        return True

    increase = trial_phi - current_phi
    if increase <= 0.0:
        return True

    base_value = 1.0
    if abs(current_phi) > 10.0:
        base_value = np.log10(abs(current_phi))

    return np.log10(increase) <= maximum_orders + base_value


def _is_objective_type(
    alpha_primal: float,
    theta: float,
    directional_derivative: float,
    options: FilterLineSearchOptions,
) -> bool:
    """IPOPT switching rule used to classify an f-type trial."""
    if directional_derivative >= 0.0:
        return False

    return (
        alpha_primal
        * (-directional_derivative) ** options.s_phi
        >
        options.delta * theta ** options.s_theta
    )


def _sufficient_reduction(
    current_phi: float,
    current_theta: float,
    trial_phi: float,
    trial_theta: float,
    options: FilterLineSearchOptions,
) -> bool:
    return (
        trial_theta
        <= (1.0 - options.gamma_theta) * current_theta
        or trial_phi - current_phi
        <= -options.gamma_phi * current_theta
    )


def _armijo_holds(
    current_phi: float,
    trial_phi: float,
    alpha_primal: float,
    directional_derivative: float,
    options: FilterLineSearchOptions,
) -> bool:
    return (
        trial_phi - current_phi
        <= options.eta_phi
        * alpha_primal
        * directional_derivative
    )


def backtracking_filter_line_search(
    *,
    state: PrimalDualState,
    current_nlp: NLPQuantities,
    direction: PrimalDualState,
    evaluate_nlp: NLPQuantityEvaluator,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: float,
    alpha_primal_max: float,
    alpha_dual_max: float,
    filter_: Filter,
    iteration: int,
    initial_constraint_violation: float,
    options: FilterLineSearchOptions | None = None,
) -> FilterLineSearchResult:
    """Backtrack the primal step until the trial point is filter acceptable."""
    opts = options or FilterLineSearchOptions()
    alpha_primal_max = _validate_step(
        "alpha_primal_max",
        alpha_primal_max,
    )
    alpha_dual_max = _validate_step(
        "alpha_dual_max",
        alpha_dual_max,
    )

    if not callable(evaluate_nlp):
        raise TypeError("evaluate_nlp must be callable.")

    current_slacks = compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
    )
    current_phi = compute_barrier_objective(
        current_nlp,
        current_slacks,
        mu,
    )
    current_theta = compute_constraint_violation(
        state,
        current_nlp,
    )
    directional_derivative = compute_barrier_directional_derivative(
        state,
        direction,
        current_nlp,
        indices,
        current_slacks,
        mu,
    )

    initial_constraint_violation = float(
        initial_constraint_violation
    )
    theta_min = (
        opts.theta_min_factor
        * max(1.0, initial_constraint_violation)
    )
    theta_max = (
        opts.theta_max_factor
        * max(1.0, initial_constraint_violation)
    )

    alpha = alpha_primal_max
    trials: list[LineSearchTrial] = []
    last_state = state
    last_nlp = current_nlp
    last_phi = current_phi
    last_theta = current_theta

    for backtracks in range(opts.maximum_backtracks + 1):
        if alpha < opts.minimum_alpha:
            return FilterLineSearchResult(
                status="minimum_step",
                state=last_state,
                nlp=last_nlp,
                alpha_primal=alpha,
                alpha_dual=alpha_dual_max,
                alpha_equality=_equality_step(
                    opts.equality_multiplier_step,
                    alpha,
                    alpha_dual_max,
                ),
                backtracks=backtracks,
                barrier_objective=last_phi,
                constraint_violation=last_theta,
                acceptance_type="none",
                trials=tuple(trials),
            )

        # Filter quantities depend only on the primal variables.  Keep all
        # multipliers fixed while evaluating trial objective and feasibility.
        primal_trial_state = apply_primal_dual_step(
            state,
            direction,
            alpha_primal=alpha,
            alpha_dual=0.0,
            alpha_equality=0.0,
        )

        rejection_reason: str | None = None
        try:
            trial_slacks = compute_bound_slacks(
                primal_trial_state,
                indices,
                bounds,
                require_strict_interior=True,
            )
            trial_nlp = evaluate_nlp(primal_trial_state)
            if not isinstance(trial_nlp, NLPQuantities):
                raise TypeError(
                    "evaluate_nlp must return NLPQuantities; "
                    f"received {type(trial_nlp).__name__}."
                )
            trial_nlp.validate_against_state(primal_trial_state)

            trial_phi = compute_barrier_objective(
                trial_nlp,
                trial_slacks,
                mu,
            )
            trial_theta = compute_constraint_violation(
                primal_trial_state,
                trial_nlp,
            )
        except (ValueError, FloatingPointError):
            trial_nlp = current_nlp
            trial_phi = float("inf")
            trial_theta = float("inf")
            rejection_reason = "invalid_trial"

        objective_type = _is_objective_type(
            alpha,
            current_theta,
            directional_derivative,
            opts,
        )
        armijo = (
            np.isfinite(trial_phi)
            and _armijo_holds(
                current_phi,
                trial_phi,
                alpha,
                directional_derivative,
                opts,
            )
        )

        accepted_by_current = False
        accepted_by_filter = False

        if rejection_reason is None and trial_theta > theta_max:
            rejection_reason = "theta_max"
        elif (
            rejection_reason is None
            and not _objective_increase_is_acceptable(
                current_phi,
                trial_phi,
                opts.objective_max_increase_orders,
            )
        ):
            rejection_reason = "objective_increase"
        elif rejection_reason is None:
            if objective_type and current_theta <= theta_min:
                accepted_by_current = armijo
            else:
                accepted_by_current = _sufficient_reduction(
                    current_phi,
                    current_theta,
                    trial_phi,
                    trial_theta,
                    opts,
                )

            if not accepted_by_current:
                rejection_reason = "current_iterate"
            else:
                accepted_by_filter = filter_.acceptable(
                    trial_phi,
                    trial_theta,
                )
                if not accepted_by_filter:
                    rejection_reason = "filter"

        trials.append(
            LineSearchTrial(
                alpha_primal=alpha,
                barrier_objective=trial_phi,
                constraint_violation=trial_theta,
                accepted_by_current=accepted_by_current,
                accepted_by_filter=accepted_by_filter,
                objective_type=objective_type,
                armijo_holds=armijo,
                rejection_reason=rejection_reason,
            )
        )

        if (
            rejection_reason is None
            and accepted_by_current
            and accepted_by_filter
        ):
            alpha_equality = _equality_step(
                opts.equality_multiplier_step,
                alpha,
                alpha_dual_max,
            )
            accepted_state = apply_primal_dual_step(
                state,
                direction,
                alpha_primal=alpha,
                alpha_dual=alpha_dual_max,
                alpha_equality=alpha_equality,
            )

            # IPOPT augments the filter for h-type steps and for steps that do
            # not satisfy the barrier-objective Armijo condition.
            if not objective_type or not armijo:
                filter_.add_reference_with_margin(
                    barrier_objective=current_phi,
                    constraint_violation=current_theta,
                    gamma_phi=opts.gamma_phi,
                    gamma_theta=opts.gamma_theta,
                    iteration=iteration,
                )
                acceptance_type: Literal["f", "h", "none"] = "h"
            else:
                acceptance_type = "f"

            return FilterLineSearchResult(
                status="accepted",
                state=accepted_state,
                nlp=trial_nlp,
                alpha_primal=alpha,
                alpha_dual=alpha_dual_max,
                alpha_equality=alpha_equality,
                backtracks=backtracks,
                barrier_objective=trial_phi,
                constraint_violation=trial_theta,
                acceptance_type=acceptance_type,
                trials=tuple(trials),
            )

        last_state = primal_trial_state
        last_nlp = trial_nlp
        last_phi = trial_phi
        last_theta = trial_theta
        alpha *= opts.alpha_reduction_factor

    return FilterLineSearchResult(
        status="maximum_backtracks",
        state=last_state,
        nlp=last_nlp,
        alpha_primal=alpha,
        alpha_dual=alpha_dual_max,
        alpha_equality=_equality_step(
            opts.equality_multiplier_step,
            alpha,
            alpha_dual_max,
        ),
        backtracks=opts.maximum_backtracks + 1,
        barrier_objective=last_phi,
        constraint_violation=last_theta,
        acceptance_type="none",
        trials=tuple(trials),
    )
