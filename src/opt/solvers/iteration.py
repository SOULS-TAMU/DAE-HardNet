"""Fixed-barrier primal-dual iteration loop with optional filter line search.

The loop keeps ``mu`` fixed and repeatedly performs:

1. evaluate the nonlinear problem,
2. assemble the KKT residual,
3. compute a primal-dual Newton direction,
4. compute fraction-to-the-boundary limits,
5. optionally backtrack with an IPOPT-style filter,
6. accept the primal-dual step.

Adaptive barrier updates, restoration, watchdogs, and second-order correction
are intentionally left for later modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .bounds import BoundIndices, BoundValues, compute_bound_slacks
from .filter import Filter, FilterEntry, compute_constraint_violation
from .line_search import (
    FilterLineSearchOptions,
    backtracking_filter_line_search,
)
from .newton import (
    NewtonRegularization,
    solve_primal_dual_newton_system,
)
from .residuals import NLPQuantities, compute_kkt_residual
from .state import PrimalDualState
from .step import (
    apply_primal_dual_step,
    compute_fraction_to_boundary_steps,
)


NLPQuantityEvaluator = Callable[[PrimalDualState], NLPQuantities]
FixedMuStatus = Literal[
    "converged",
    "maximum_iterations",
    "stalled",
    "line_search_failed",
]


@dataclass(frozen=True)
class FixedMuOptions:
    """Options for the fixed-``mu`` primal-dual loop.

    Set ``line_search`` to ``FilterLineSearchOptions()`` to enable the filter
    line search.  Leave it as ``None`` to retain the earlier direct
    fraction-to-the-boundary step behavior.
    """

    max_iterations: int = 50
    residual_tolerance: float = 1e-9
    tau: float = 0.995
    step_scale: float = 1.0
    minimum_step: float = 1e-14
    regularization: NewtonRegularization | None = None
    require_positive_multipliers: bool = True
    positivity_tolerance: float = 0.0
    check_newton_residual: bool = False
    line_search: FilterLineSearchOptions | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_iterations, int):
            raise TypeError(
                "max_iterations must be an integer; "
                f"received {type(self.max_iterations).__name__}."
            )
        if self.max_iterations < 0:
            raise ValueError(
                "max_iterations must be nonnegative; "
                f"received {self.max_iterations}."
            )
        for name in (
            "residual_tolerance",
            "tau",
            "step_scale",
            "minimum_step",
            "positivity_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value}.")
            object.__setattr__(self, name, value)

        if self.residual_tolerance <= 0.0:
            raise ValueError(
                "residual_tolerance must be positive; "
                f"received {self.residual_tolerance}."
            )
        if not 0.0 < self.tau < 1.0:
            raise ValueError(
                f"tau must satisfy 0 < tau < 1; received {self.tau}."
            )
        if not 0.0 < self.step_scale <= 1.0:
            raise ValueError(
                "step_scale must satisfy 0 < step_scale <= 1; "
                f"received {self.step_scale}."
            )
        if self.minimum_step < 0.0:
            raise ValueError(
                "minimum_step must be nonnegative; "
                f"received {self.minimum_step}."
            )
        if self.positivity_tolerance < 0.0:
            raise ValueError(
                "positivity_tolerance must be nonnegative; "
                f"received {self.positivity_tolerance}."
            )


@dataclass(frozen=True)
class FixedMuIterationRecord:
    """Diagnostics for one Newton direction and attempted step."""

    iteration: int
    residual_inf_norm: float
    direction_inf_norm: float
    alpha_primal_max: float
    alpha_dual_max: float
    alpha_primal: float
    alpha_dual: float
    reduced_newton_residual_inf_norm: float
    full_newton_residual_inf_norm: float
    backtracks: int = 0
    barrier_objective: float = float("nan")
    constraint_violation: float = float("nan")
    acceptance_type: Literal["f", "h", "none"] = "none"

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                f"iteration must be nonnegative; received {self.iteration}."
            )
        if self.backtracks < 0:
            raise ValueError(
                f"backtracks must be nonnegative; received {self.backtracks}."
            )

        finite_names = (
            "residual_inf_norm",
            "direction_inf_norm",
            "alpha_primal_max",
            "alpha_dual_max",
            "alpha_primal",
            "alpha_dual",
            "reduced_newton_residual_inf_norm",
        )
        for name in finite_names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value}.")
            object.__setattr__(self, name, value)

        for name in (
            "full_newton_residual_inf_norm",
            "barrier_objective",
            "constraint_violation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) and not np.isnan(value):
                raise ValueError(
                    f"{name} must be finite or NaN; received {value}."
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FixedMuResult:
    """Final state and diagnostics from the fixed-barrier loop."""

    state: PrimalDualState
    status: FixedMuStatus
    mu: float
    iterations: int
    final_residual_inf_norm: float
    residual_history: tuple[float, ...]
    records: tuple[FixedMuIterationRecord, ...]
    filter_entries: tuple[FilterEntry, ...]

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _validate_mu(mu: float) -> float:
    mu = float(mu)
    if not np.isfinite(mu):
        raise ValueError(f"mu must be finite; received {mu}.")
    if mu < 0.0:
        raise ValueError(f"mu must be nonnegative; received {mu}.")
    return mu


def _direction_infinity_norm(direction: PrimalDualState) -> float:
    if direction.size == 0:
        return 0.0
    return float(np.max(np.abs(direction.flatten())))


def _validate_updated_interior(
    state: PrimalDualState,
    indices: BoundIndices,
    bounds: BoundValues,
    *,
    require_positive_multipliers: bool,
    positivity_tolerance: float,
) -> None:
    compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
        positivity_tolerance=positivity_tolerance,
    )

    if not require_positive_multipliers:
        return

    for name in ("z_L", "z_U", "v_L", "v_U"):
        values = getattr(state, name)
        bad = np.flatnonzero(values <= positivity_tolerance)
        if bad.size:
            raise RuntimeError(
                f"The accepted step left {name} outside the strict dual "
                f"interior at indices {bad.tolist()}, with values "
                f"{values[bad].tolist()}."
            )


def _make_result(
    *,
    state: PrimalDualState,
    status: FixedMuStatus,
    mu: float,
    iterations: int,
    final_residual_inf_norm: float,
    residual_history: list[float],
    records: list[FixedMuIterationRecord],
    filter_: Filter,
) -> FixedMuResult:
    return FixedMuResult(
        state=state,
        status=status,
        mu=mu,
        iterations=iterations,
        final_residual_inf_norm=final_residual_inf_norm,
        residual_history=tuple(residual_history),
        records=tuple(records),
        filter_entries=filter_.entries,
    )


def run_fixed_mu_iterations(
    initial_state: PrimalDualState,
    evaluate_nlp: NLPQuantityEvaluator,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: float,
    *,
    options: FixedMuOptions | None = None,
) -> FixedMuResult:
    """Run primal-dual Newton iterations at a fixed barrier parameter.

    When ``options.line_search`` is enabled, every NLP evaluation must include
    ``objective_value`` because the filter uses the logarithmic barrier
    objective.
    """
    if not callable(evaluate_nlp):
        raise TypeError("evaluate_nlp must be callable.")

    mu = _validate_mu(mu)
    opts = options or FixedMuOptions()

    state = initial_state.copy()
    state.validate_finite()
    indices.validate_against_state(state)
    bounds.validate_against_indices(indices)

    _validate_updated_interior(
        state,
        indices,
        bounds,
        require_positive_multipliers=opts.require_positive_multipliers,
        positivity_tolerance=opts.positivity_tolerance,
    )

    filter_ = Filter()
    initial_constraint_violation: float | None = None
    residual_history: list[float] = []
    records: list[FixedMuIterationRecord] = []

    for iteration in range(opts.max_iterations + 1):
        nlp = evaluate_nlp(state)
        if not isinstance(nlp, NLPQuantities):
            raise TypeError(
                "evaluate_nlp must return NLPQuantities; "
                f"received {type(nlp).__name__}."
            )

        nlp.validate_against_state(state)
        nlp.validate_hessian_symmetry()

        slacks = compute_bound_slacks(
            state,
            indices,
            bounds,
            require_strict_interior=True,
            positivity_tolerance=opts.positivity_tolerance,
        )
        residual = compute_kkt_residual(
            state,
            nlp,
            indices,
            slacks,
            mu,
        )
        residual_norm = residual.infinity_norm()
        residual_history.append(residual_norm)

        if initial_constraint_violation is None:
            initial_constraint_violation = compute_constraint_violation(
                state,
                nlp,
            )

        if residual_norm <= opts.residual_tolerance:
            return _make_result(
                state=state,
                status="converged",
                mu=mu,
                iterations=iteration,
                final_residual_inf_norm=residual_norm,
                residual_history=residual_history,
                records=records,
                filter_=filter_,
            )

        if iteration == opts.max_iterations:
            return _make_result(
                state=state,
                status="maximum_iterations",
                mu=mu,
                iterations=iteration,
                final_residual_inf_norm=residual_norm,
                residual_history=residual_history,
                records=records,
                filter_=filter_,
            )

        newton_result = solve_primal_dual_newton_system(
            state=state,
            nlp=nlp,
            indices=indices,
            bounds=bounds,
            mu=mu,
            regularization=opts.regularization,
            require_positive_multipliers=opts.require_positive_multipliers,
            positivity_tolerance=opts.positivity_tolerance,
            check_full_residual=opts.check_newton_residual,
        )
        direction = newton_result.direction

        step_limits = compute_fraction_to_boundary_steps(
            state=state,
            direction=direction,
            indices=indices,
            bounds=bounds,
            tau=opts.tau,
        )

        alpha_primal_max = (
            opts.step_scale * step_limits.alpha_primal
        )
        alpha_dual_max = (
            opts.step_scale * step_limits.alpha_dual
        )

        if min(alpha_primal_max, alpha_dual_max) <= opts.minimum_step:
            records.append(
                FixedMuIterationRecord(
                    iteration=iteration,
                    residual_inf_norm=residual_norm,
                    direction_inf_norm=_direction_infinity_norm(direction),
                    alpha_primal_max=alpha_primal_max,
                    alpha_dual_max=alpha_dual_max,
                    alpha_primal=alpha_primal_max,
                    alpha_dual=alpha_dual_max,
                    reduced_newton_residual_inf_norm=(
                        newton_result.reduced_residual_inf_norm
                    ),
                    full_newton_residual_inf_norm=(
                        newton_result.full_residual_inf_norm
                    ),
                )
            )
            return _make_result(
                state=state,
                status="stalled",
                mu=mu,
                iterations=iteration,
                final_residual_inf_norm=residual_norm,
                residual_history=residual_history,
                records=records,
                filter_=filter_,
            )

        if opts.line_search is None:
            alpha_primal = alpha_primal_max
            alpha_dual = alpha_dual_max
            state = apply_primal_dual_step(
                state,
                direction,
                alpha_primal=alpha_primal,
                alpha_dual=alpha_dual,
            )

            record = FixedMuIterationRecord(
                iteration=iteration,
                residual_inf_norm=residual_norm,
                direction_inf_norm=_direction_infinity_norm(direction),
                alpha_primal_max=alpha_primal_max,
                alpha_dual_max=alpha_dual_max,
                alpha_primal=alpha_primal,
                alpha_dual=alpha_dual,
                reduced_newton_residual_inf_norm=(
                    newton_result.reduced_residual_inf_norm
                ),
                full_newton_residual_inf_norm=(
                    newton_result.full_residual_inf_norm
                ),
            )
        else:
            line_search_result = backtracking_filter_line_search(
                state=state,
                current_nlp=nlp,
                direction=direction,
                evaluate_nlp=evaluate_nlp,
                indices=indices,
                bounds=bounds,
                mu=mu,
                alpha_primal_max=alpha_primal_max,
                alpha_dual_max=alpha_dual_max,
                filter_=filter_,
                iteration=iteration,
                initial_constraint_violation=(
                    initial_constraint_violation
                ),
                options=opts.line_search,
            )

            record = FixedMuIterationRecord(
                iteration=iteration,
                residual_inf_norm=residual_norm,
                direction_inf_norm=_direction_infinity_norm(direction),
                alpha_primal_max=alpha_primal_max,
                alpha_dual_max=alpha_dual_max,
                alpha_primal=line_search_result.alpha_primal,
                alpha_dual=line_search_result.alpha_dual,
                reduced_newton_residual_inf_norm=(
                    newton_result.reduced_residual_inf_norm
                ),
                full_newton_residual_inf_norm=(
                    newton_result.full_residual_inf_norm
                ),
                backtracks=line_search_result.backtracks,
                barrier_objective=(
                    line_search_result.barrier_objective
                ),
                constraint_violation=(
                    line_search_result.constraint_violation
                ),
                acceptance_type=(
                    line_search_result.acceptance_type
                ),
            )

            if not line_search_result.accepted:
                records.append(record)
                return _make_result(
                    state=state,
                    status="line_search_failed",
                    mu=mu,
                    iterations=iteration,
                    final_residual_inf_norm=residual_norm,
                    residual_history=residual_history,
                    records=records,
                    filter_=filter_,
                )

            state = line_search_result.state

        records.append(record)
        state.validate_finite()

        _validate_updated_interior(
            state,
            indices,
            bounds,
            require_positive_multipliers=opts.require_positive_multipliers,
            positivity_tolerance=opts.positivity_tolerance,
        )

    raise RuntimeError("Unreachable fixed-mu iteration state.")
