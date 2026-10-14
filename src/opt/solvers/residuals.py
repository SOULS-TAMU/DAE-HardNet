"""KKT residual assembly for the NumPy IPOPT reference solver.

This module mirrors the residual construction performed by IPOPT's
``PDSearchDirCalculator`` for the primal-dual barrier equations.  It assembles
the eight residual blocks

    (r_x, r_s, r_c, r_d, r_zL, r_zU, r_vL, r_vU)

in the same block order as :class:`src.state.PrimalDualState`.

This first reference implementation intentionally omits IPOPT's optional bound
damping, Mehrotra predictor-corrector modifications, and numerical
regularization.  Those belong to later solver layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .bounds import BoundIndices, BoundSlacks
from .state import FloatArray, PrimalDualState


FloatMatrix = NDArray[np.float64]


def _as_float_vector(value: FloatArray, name: str) -> FloatArray:
    """Return ``value`` as a validated one-dimensional float64 array."""
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


def _as_float_matrix(value: FloatMatrix, name: str) -> FloatMatrix:
    """Return ``value`` as a validated two-dimensional float64 array."""
    array = np.asarray(value, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional; received shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        bad_locations = np.argwhere(~np.isfinite(array))
        raise ValueError(
            f"{name} contains non-finite values at locations "
            f"{bad_locations.tolist()}."
        )

    return array


@dataclass(frozen=True)
class NLPQuantities:
    """NLP values and derivatives evaluated at the current primal point.

    Parameters
    ----------
    grad_f
        Objective gradient ``∇f(x)`` with shape ``(n_x,)``.
    c
        Equality-constraint values ``c(x)`` with shape ``(n_c,)``.
    d
        Ranged-constraint function values ``d(x)`` with shape ``(n_s,)``.
    jac_c
        Equality Jacobian ``J_c(x)`` with shape ``(n_c, n_x)``.
    jac_d
        Ranged-constraint Jacobian ``J_d(x)`` with shape ``(n_s, n_x)``.
    hessian_lagrangian
        Lagrangian Hessian ``W`` with shape ``(n_x, n_x)``.

    Notes
    -----
    The Hessian is not used directly while assembling the residual, but it is
    stored here because the next Newton-system layer uses the same evaluated
    NLP quantities.
    """

    grad_f: FloatArray
    c: FloatArray
    d: FloatArray
    jac_c: FloatMatrix
    jac_d: FloatMatrix
    hessian_lagrangian: FloatMatrix
    objective_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grad_f", _as_float_vector(self.grad_f, "grad_f")
        )
        object.__setattr__(self, "c", _as_float_vector(self.c, "c"))
        object.__setattr__(self, "d", _as_float_vector(self.d, "d"))
        object.__setattr__(
            self, "jac_c", _as_float_matrix(self.jac_c, "jac_c")
        )
        object.__setattr__(
            self, "jac_d", _as_float_matrix(self.jac_d, "jac_d")
        )
        object.__setattr__(
            self,
            "hessian_lagrangian",
            _as_float_matrix(
                self.hessian_lagrangian,
                "hessian_lagrangian",
            ),
        )

        if self.objective_value is not None:
            objective_value = float(self.objective_value)
            if not np.isfinite(objective_value):
                raise ValueError(
                    "objective_value must be finite when supplied; "
                    f"received {objective_value}."
                )
            object.__setattr__(self, "objective_value", objective_value)

    def validate_against_state(self, state: PrimalDualState) -> None:
        """Validate all NLP dimensions against a primal-dual state."""
        n_x = state.x.size
        n_s = state.s.size
        n_c = state.y_c.size

        expected_shapes = {
            "grad_f": (n_x,),
            "c": (n_c,),
            "d": (n_s,),
            "jac_c": (n_c, n_x),
            "jac_d": (n_s, n_x),
            "hessian_lagrangian": (n_x, n_x),
        }

        actual_shapes = {
            "grad_f": self.grad_f.shape,
            "c": self.c.shape,
            "d": self.d.shape,
            "jac_c": self.jac_c.shape,
            "jac_d": self.jac_d.shape,
            "hessian_lagrangian": self.hessian_lagrangian.shape,
        }

        mismatches = [
            f"{name}: expected {expected_shapes[name]}, received {actual_shapes[name]}"
            for name in expected_shapes
            if actual_shapes[name] != expected_shapes[name]
        ]

        if mismatches:
            raise ValueError(
                "NLP quantities are incompatible with the primal-dual state:\n  "
                + "\n  ".join(mismatches)
            )


    def require_objective_value(self) -> float:
        """Return the scalar objective value or raise a clear error.

        The Newton system itself only requires the objective gradient and
        Lagrangian Hessian.  A filter line search additionally requires the
        scalar objective value in order to form the barrier objective.
        """
        if self.objective_value is None:
            raise ValueError(
                "objective_value is required for the filter line search. "
                "Return NLPQuantities(..., objective_value=f_value) from "
                "the NLP evaluator."
            )
        return self.objective_value

    def validate_hessian_symmetry(
        self,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        """Require the supplied Lagrangian Hessian to be numerically symmetric."""
        if not np.allclose(
            self.hessian_lagrangian,
            self.hessian_lagrangian.T,
            rtol=rtol,
            atol=atol,
        ):
            asymmetry = (
                self.hessian_lagrangian
                - self.hessian_lagrangian.T
            )
            raise ValueError(
                "hessian_lagrangian is not symmetric within the requested "
                f"tolerances; maximum asymmetry is "
                f"{np.max(np.abs(asymmetry)):.3e}."
            )


@dataclass(frozen=True)
class KKTResidual:
    """Eight residual blocks of the primal-dual barrier KKT equations.

    The block names match the corresponding primal-dual variables:

    ``x``
        Stationarity residual with respect to ``x``.
    ``s``
        Stationarity residual with respect to ``s``.
    ``y_c``
        Equality feasibility residual ``c(x)``.
    ``y_d``
        Slack-equation residual ``d(x) - s``.
    ``z_L``, ``z_U``
        Complementarity residuals for lower and upper bounds on ``x``.
    ``v_L``, ``v_U``
        Complementarity residuals for lower and upper bounds on ``s``.
    """

    x: FloatArray
    s: FloatArray
    y_c: FloatArray
    y_d: FloatArray
    z_L: FloatArray
    z_U: FloatArray
    v_L: FloatArray
    v_U: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "x",
            "s",
            "y_c",
            "y_d",
            "z_L",
            "z_U",
            "v_L",
            "v_U",
        ):
            object.__setattr__(
                self,
                name,
                _as_float_vector(getattr(self, name), name),
            )

    @property
    def size(self) -> int:
        """Total number of scalar residual entries."""
        return int(sum(block.size for block in self.blocks()))

    @property
    def block_sizes(self) -> tuple[int, int, int, int, int, int, int, int]:
        """Lengths of the eight residual blocks in canonical order."""
        return tuple(int(block.size) for block in self.blocks())  # type: ignore[return-value]

    def blocks(self) -> tuple[FloatArray, ...]:
        """Return residual blocks in canonical IPOPT order."""
        return (
            self.x,
            self.s,
            self.y_c,
            self.y_d,
            self.z_L,
            self.z_U,
            self.v_L,
            self.v_U,
        )

    def named_blocks(self) -> Iterator[tuple[str, FloatArray]]:
        """Iterate over ``(name, block)`` pairs in canonical order."""
        names = ("x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U")
        return iter(zip(names, self.blocks(), strict=True))

    def flatten(self) -> FloatArray:
        """Concatenate all residual blocks into one vector."""
        if self.size == 0:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self.blocks())

    def infinity_norm(self) -> float:
        """Return ``max(abs(residual))`` across all blocks."""
        if self.size == 0:
            return 0.0
        return float(np.max(np.abs(self.flatten())))

    def block_infinity_norms(self) -> dict[str, float]:
        """Return an infinity norm for each residual block."""
        return {
            name: 0.0 if block.size == 0 else float(np.max(np.abs(block)))
            for name, block in self.named_blocks()
        }

    def squared_l2_norm(self) -> float:
        """Return the squared Euclidean norm of the complete residual."""
        return float(
            sum(float(block @ block) for block in self.blocks())
        )

    def validate_against_state(self, state: PrimalDualState) -> None:
        """Require residual blocks to have the same shapes as ``state``."""
        if self.block_sizes != state.block_sizes:
            raise ValueError(
                "Residual and state block sizes must match: "
                f"{self.block_sizes} != {state.block_sizes}."
            )

    def as_state(self) -> PrimalDualState:
        """Return the residual blocks in a ``PrimalDualState`` container."""
        return PrimalDualState(
            *(block.copy() for block in self.blocks())
        )


def compute_kkt_residual(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    mu: float,
) -> KKTResidual:
    r"""Assemble the primal-dual barrier KKT residual.

    The residual equations are

    .. math::

        r_x &=
            \nabla f(x)
            + J_c(x)^T y_c
            + J_d(x)^T y_d
            - P_{xL} z_L
            + P_{xU} z_U, \\

        r_s &=
            -y_d - P_{dL} v_L + P_{dU} v_U, \\

        r_c &= c(x), \\

        r_d &= d(x) - s, \\

        r_{zL} &= a_{xL} \odot z_L - \mu \mathbf{1}, \\

        r_{zU} &= a_{xU} \odot z_U - \mu \mathbf{1}, \\

        r_{vL} &= a_{sL} \odot v_L - \mu \mathbf{1}, \\

        r_{vU} &= a_{sU} \odot v_U - \mu \mathbf{1}.

    Parameters
    ----------
    state
        Current primal-dual iterate.
    nlp
        Objective and constraint values and derivatives evaluated at ``state.x``.
    indices
        Locations of finite lower and upper bounds.
    slacks
        Distances from the current primal variables to those finite bounds.
    mu
        Nonnegative barrier parameter.

    Returns
    -------
    KKTResidual
        The eight residual blocks in IPOPT order.
    """
    mu = float(mu)

    if not np.isfinite(mu):
        raise ValueError(f"mu must be finite; received {mu}.")

    if mu < 0.0:
        raise ValueError(f"mu must be nonnegative; received {mu}.")

    state.validate_finite()
    indices.validate_against_state(state)
    nlp.validate_against_state(state)

    if slacks.sizes != indices.sizes:
        raise ValueError(
            "Bound slack lengths must match the finite-bound index lengths: "
            f"{slacks.sizes} != {indices.sizes}."
        )

    # The barrier Newton equations divide by these slacks in the next layer.
    slacks.validate_strictly_positive()

    r_x = (
        nlp.grad_f
        + nlp.jac_c.T @ state.y_c
        + nlp.jac_d.T @ state.y_d
    ).copy()

    # These indexed scatter-adds are equivalent to applying IPOPT's bound
    # selection matrices.  np.add.at also handles variables having two-sided
    # bounds correctly.
    np.add.at(r_x, indices.x_lower, -state.z_L)
    np.add.at(r_x, indices.x_upper, state.z_U)

    r_s = (-state.y_d).copy()
    np.add.at(r_s, indices.s_lower, -state.v_L)
    np.add.at(r_s, indices.s_upper, state.v_U)

    r_c = nlp.c.copy()
    r_d = nlp.d - state.s

    r_z_L = slacks.x_lower * state.z_L - mu
    r_z_U = slacks.x_upper * state.z_U - mu
    r_v_L = slacks.s_lower * state.v_L - mu
    r_v_U = slacks.s_upper * state.v_U - mu

    residual = KKTResidual(
        x=r_x,
        s=r_s,
        y_c=r_c,
        y_d=r_d,
        z_L=r_z_L,
        z_U=r_z_U,
        v_L=r_v_L,
        v_U=r_v_U,
    )
    residual.validate_against_state(state)
    return residual
