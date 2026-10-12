"""IPOPT-style two-dimensional filter and line-search metrics.

The filter stores pairs

    (barrier objective, constraint violation)

including the margin already applied to the point that generated the entry.
A trial pair is acceptable to one entry when it improves at least one
coordinate:

    trial_phi <= entry_phi
    or
    trial_theta <= entry_theta.

It is acceptable to the complete filter only when this condition holds for
every stored entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import BoundIndices, BoundSlacks
from .residuals import NLPQuantities
from .state import FloatArray, PrimalDualState


@dataclass(frozen=True)
class FilterEntry:
    """One stored filter corner."""

    barrier_objective: float
    constraint_violation: float
    iteration: int

    def __post_init__(self) -> None:
        phi = float(self.barrier_objective)
        theta = float(self.constraint_violation)

        if not np.isfinite(phi):
            raise ValueError(
                "barrier_objective must be finite; "
                f"received {phi}."
            )
        if not np.isfinite(theta) or theta < 0.0:
            raise ValueError(
                "constraint_violation must be finite and nonnegative; "
                f"received {theta}."
            )
        if self.iteration < 0:
            raise ValueError(
                f"iteration must be nonnegative; received {self.iteration}."
            )

        object.__setattr__(self, "barrier_objective", phi)
        object.__setattr__(self, "constraint_violation", theta)

    def accepts(
        self,
        barrier_objective: float,
        constraint_violation: float,
    ) -> bool:
        """Whether a trial pair improves at least one filter coordinate."""
        return (
            barrier_objective <= self.barrier_objective
            or constraint_violation <= self.constraint_violation
        )

    def is_dominated_by(
        self,
        barrier_objective: float,
        constraint_violation: float,
    ) -> bool:
        """Whether a new entry is no worse in both coordinates."""
        return (
            barrier_objective <= self.barrier_objective
            and constraint_violation <= self.constraint_violation
        )


class Filter:
    """Mutable collection of nondominated filter entries."""

    def __init__(self) -> None:
        self._entries: list[FilterEntry] = []

    @property
    def entries(self) -> tuple[FilterEntry, ...]:
        """Read-only view of the current filter entries."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Remove every filter entry."""
        self._entries.clear()

    def acceptable(
        self,
        barrier_objective: float,
        constraint_violation: float,
    ) -> bool:
        """Check acceptability against every stored entry."""
        phi = float(barrier_objective)
        theta = float(constraint_violation)

        if not np.isfinite(phi):
            return False
        if not np.isfinite(theta) or theta < 0.0:
            return False

        return all(entry.accepts(phi, theta) for entry in self._entries)

    def add(
        self,
        barrier_objective: float,
        constraint_violation: float,
        iteration: int,
    ) -> None:
        """Add an entry and delete entries dominated by it."""
        entry = FilterEntry(
            barrier_objective=barrier_objective,
            constraint_violation=constraint_violation,
            iteration=iteration,
        )

        self._entries = [
            old
            for old in self._entries
            if not old.is_dominated_by(
                entry.barrier_objective,
                entry.constraint_violation,
            )
        ]
        self._entries.append(entry)

    def add_reference_with_margin(
        self,
        *,
        barrier_objective: float,
        constraint_violation: float,
        gamma_phi: float,
        gamma_theta: float,
        iteration: int,
    ) -> None:
        """Add the current point using IPOPT's filter margins.

        The stored corner is

        ``phi - gamma_phi * theta`` and
        ``(1 - gamma_theta) * theta``.
        """
        self.add(
            barrier_objective
            - float(gamma_phi) * float(constraint_violation),
            (1.0 - float(gamma_theta))
            * float(constraint_violation),
            iteration,
        )


def compute_constraint_violation(
    state: PrimalDualState,
    nlp: NLPQuantities,
) -> float:
    """Return IPOPT-style one-norm primal infeasibility.

    The internal equality blocks are

    ``c(x)`` and ``d(x) - s``.
    """
    nlp.validate_against_state(state)
    return float(
        np.sum(np.abs(nlp.c))
        + np.sum(np.abs(nlp.d - state.s))
    )


def compute_barrier_objective(
    nlp: NLPQuantities,
    slacks: BoundSlacks,
    mu: float,
) -> float:
    """Return objective plus logarithmic barrier terms."""
    mu = float(mu)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(
            f"mu must be finite and nonnegative; received {mu}."
        )

    objective = nlp.require_objective_value()
    slacks.validate_strictly_positive()

    if mu == 0.0:
        return objective

    barrier_sum = 0.0
    for _, block in slacks.named_blocks():
        if block.size:
            barrier_sum += float(np.sum(np.log(block)))

    return objective - mu * barrier_sum


def compute_barrier_gradient(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    mu: float,
) -> tuple[FloatArray, FloatArray]:
    """Compute gradients of the barrier objective with respect to ``x`` and ``s``."""
    mu = float(mu)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(
            f"mu must be finite and nonnegative; received {mu}."
        )

    indices.validate_against_state(state)
    nlp.validate_against_state(state)
    slacks.validate_strictly_positive()

    grad_x = nlp.grad_f.copy()
    grad_s = np.zeros_like(state.s)

    if mu == 0.0:
        return grad_x, grad_s

    np.add.at(
        grad_x,
        indices.x_lower,
        -mu / slacks.x_lower,
    )
    np.add.at(
        grad_x,
        indices.x_upper,
        mu / slacks.x_upper,
    )
    np.add.at(
        grad_s,
        indices.s_lower,
        -mu / slacks.s_lower,
    )
    np.add.at(
        grad_s,
        indices.s_upper,
        mu / slacks.s_upper,
    )

    return grad_x, grad_s


def compute_barrier_directional_derivative(
    state: PrimalDualState,
    direction: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    mu: float,
) -> float:
    """Return the directional derivative of the barrier objective."""
    if state.block_sizes != direction.block_sizes:
        raise ValueError(
            "State and direction block sizes must match: "
            f"{state.block_sizes} != {direction.block_sizes}."
        )

    grad_x, grad_s = compute_barrier_gradient(
        state,
        nlp,
        indices,
        slacks,
        mu,
    )
    return float(
        grad_x @ direction.x
        + grad_s @ direction.s
    )
