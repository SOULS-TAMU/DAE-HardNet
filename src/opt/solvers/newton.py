"""Dense primal-dual Newton system for the NumPy IPOPT reference solver.

This module mirrors the main mathematical operations performed by IPOPT's

- ``PDSearchDirCalculator``:
  builds the primal-dual KKT residual,
- ``PDFullSpaceSolver``:
  eliminates bound-multiplier directions and recovers them after the solve,
- ``StdAugSystemSolver``:
  solves the reduced symmetric augmented system.

The implementation uses ``numpy.linalg.solve`` intentionally.  It is a dense
correctness reference that can later be ported to JAX and replaced by sparse
CPU or CUDA linear algebra.

The Newton direction satisfies

    F_w(state; mu) @ direction = -F(state; mu).

Optional diagonal regularization follows IPOPT's reduced augmented-system sign
convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import (
    BoundIndices,
    BoundSlacks,
    BoundValues,
    compute_bound_slacks,
)
from .residuals import KKTResidual, NLPQuantities, compute_kkt_residual
from .state import FloatArray, PrimalDualState


@dataclass(frozen=True)
class NewtonRegularization:
    """Diagonal regularization for the reduced augmented KKT matrix.

    The resulting reduced matrix contains

    .. code-block:: text

        W + Sigma_x + delta_x I
        Sigma_s + delta_s I
        -delta_c I
        -delta_d I

    on its four diagonal blocks.

    All values must be finite and nonnegative.
    """

    delta_x: float = 0.0
    delta_s: float = 0.0
    delta_c: float = 0.0
    delta_d: float = 0.0

    def __post_init__(self) -> None:
        for name in ("delta_x", "delta_s", "delta_c", "delta_d"):
            value = float(getattr(self, name))

            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value}.")
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative; received {value}.")

            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ReducedKKTSystem:
    """Dense reduced augmented system after eliminating bound multipliers."""

    matrix: np.ndarray
    rhs: FloatArray
    sigma_x: FloatArray
    sigma_s: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        rhs = np.asarray(self.rhs, dtype=np.float64)
        sigma_x = np.asarray(self.sigma_x, dtype=np.float64)
        sigma_s = np.asarray(self.sigma_s, dtype=np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "matrix must be square; "
                f"received shape {matrix.shape}."
            )
        if rhs.ndim != 1 or rhs.size != matrix.shape[0]:
            raise ValueError(
                "rhs must be one-dimensional and match the matrix dimension; "
                f"received rhs shape {rhs.shape} and matrix shape {matrix.shape}."
            )
        if sigma_x.ndim != 1:
            raise ValueError(
                f"sigma_x must be one-dimensional; received {sigma_x.shape}."
            )
        if sigma_s.ndim != 1:
            raise ValueError(
                f"sigma_s must be one-dimensional; received {sigma_s.shape}."
            )

        for name, array in (
            ("matrix", matrix),
            ("rhs", rhs),
            ("sigma_x", sigma_x),
            ("sigma_s", sigma_s),
        ):
            if not np.all(np.isfinite(array)):
                bad = np.argwhere(~np.isfinite(array))
                raise ValueError(
                    f"{name} contains non-finite values at {bad.tolist()}."
                )

        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "sigma_x", sigma_x)
        object.__setattr__(self, "sigma_s", sigma_s)

    @property
    def dimension(self) -> int:
        """Dimension of the reduced linear system."""
        return int(self.rhs.size)

    def symmetry_error(self) -> float:
        """Return the maximum absolute matrix asymmetry."""
        if self.matrix.size == 0:
            return 0.0
        return float(np.max(np.abs(self.matrix - self.matrix.T)))

    def linear_residual(self, solution: FloatArray) -> FloatArray:
        """Return ``matrix @ solution - rhs``."""
        vector = np.asarray(solution, dtype=np.float64)
        if vector.ndim != 1 or vector.size != self.dimension:
            raise ValueError(
                "solution must be one-dimensional and match the reduced "
                f"dimension {self.dimension}; received shape {vector.shape}."
            )
        return self.matrix @ vector - self.rhs


@dataclass(frozen=True)
class NewtonSolveResult:
    """Result and diagnostics for one primal-dual Newton solve."""

    direction: PrimalDualState
    residual: KKTResidual
    slacks: BoundSlacks
    reduced_system: ReducedKKTSystem
    reduced_solution: FloatArray
    reduced_residual_inf_norm: float
    full_residual_inf_norm: float

    def __post_init__(self) -> None:
        reduced_solution = np.asarray(
            self.reduced_solution,
            dtype=np.float64,
        )

        if reduced_solution.ndim != 1:
            raise ValueError(
                "reduced_solution must be one-dimensional; "
                f"received shape {reduced_solution.shape}."
            )

        if reduced_solution.size != self.reduced_system.dimension:
            raise ValueError(
                "reduced_solution has the wrong length: "
                f"{reduced_solution.size} != "
                f"{self.reduced_system.dimension}."
            )

        object.__setattr__(self, "reduced_solution", reduced_solution)
        object.__setattr__(
            self,
            "reduced_residual_inf_norm",
            float(self.reduced_residual_inf_norm),
        )
        object.__setattr__(
            self,
            "full_residual_inf_norm",
            float(self.full_residual_inf_norm),
        )


def _validate_positive_bound_multipliers(
    state: PrimalDualState,
    *,
    tolerance: float = 0.0,
) -> None:
    """Require all finite-bound multipliers to exceed ``tolerance``."""
    tolerance = float(tolerance)

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "tolerance must be finite and nonnegative; "
            f"received {tolerance}."
        )

    for name in ("z_L", "z_U", "v_L", "v_U"):
        multiplier = getattr(state, name)
        bad = np.flatnonzero(multiplier <= tolerance)

        if bad.size:
            raise ValueError(
                f"{name} must be strictly greater than {tolerance}; "
                f"violations at indices {bad.tolist()} with values "
                f"{multiplier[bad].tolist()}."
            )


def compute_sigma(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
) -> tuple[FloatArray, FloatArray]:
    r"""Compute IPOPT's diagonal ``Sigma_x`` and ``Sigma_s`` vectors.

    The entries are

    .. math::

        \Sigma_x[i]
        =
        \sum_{j: I_{xL}[j]=i} z_L[j] / a_{xL}[j]
        +
        \sum_{j: I_{xU}[j]=i} z_U[j] / a_{xU}[j],

    with an analogous expression for ``Sigma_s``.
    """
    indices.validate_against_state(state)

    if slacks.sizes != indices.sizes:
        raise ValueError(
            "Bound slack lengths must match bound index lengths: "
            f"{slacks.sizes} != {indices.sizes}."
        )

    slacks.validate_strictly_positive()

    sigma_x = np.zeros_like(state.x)
    sigma_s = np.zeros_like(state.s)

    np.add.at(
        sigma_x,
        indices.x_lower,
        state.z_L / slacks.x_lower,
    )
    np.add.at(
        sigma_x,
        indices.x_upper,
        state.z_U / slacks.x_upper,
    )

    np.add.at(
        sigma_s,
        indices.s_lower,
        state.v_L / slacks.s_lower,
    )
    np.add.at(
        sigma_s,
        indices.s_upper,
        state.v_U / slacks.s_upper,
    )

    return sigma_x, sigma_s


def build_reduced_kkt_rhs(
    residual: KKTResidual,
    indices: BoundIndices,
    slacks: BoundSlacks,
) -> FloatArray:
    r"""Build the reduced Newton right-hand side after bound elimination.

    The direction convention is

    .. math::

        F_w \Delta w = -F.

    Therefore,

    .. math::

        b_x &=
        -r_x
        -P_{xL} A_{xL}^{-1} r_{zL}
        +P_{xU} A_{xU}^{-1} r_{zU}, \\

        b_s &=
        -r_s
        -P_{dL} A_{sL}^{-1} r_{vL}
        +P_{dU} A_{sU}^{-1} r_{vU}, \\

        b_c &= -r_c, \qquad b_d = -r_d.
    """
    if slacks.sizes != indices.sizes:
        raise ValueError(
            "Bound slack lengths must match bound index lengths: "
            f"{slacks.sizes} != {indices.sizes}."
        )

    slacks.validate_strictly_positive()

    b_x = -residual.x.copy()
    b_s = -residual.s.copy()

    np.add.at(
        b_x,
        indices.x_lower,
        -residual.z_L / slacks.x_lower,
    )
    np.add.at(
        b_x,
        indices.x_upper,
        residual.z_U / slacks.x_upper,
    )

    np.add.at(
        b_s,
        indices.s_lower,
        -residual.v_L / slacks.s_lower,
    )
    np.add.at(
        b_s,
        indices.s_upper,
        residual.v_U / slacks.s_upper,
    )

    return np.concatenate(
        (
            b_x,
            b_s,
            -residual.y_c,
            -residual.y_d,
        )
    )


def build_reduced_kkt_matrix(
    state: PrimalDualState,
    nlp: NLPQuantities,
    sigma_x: FloatArray,
    sigma_s: FloatArray,
    *,
    regularization: NewtonRegularization | None = None,
) -> np.ndarray:
    """Construct IPOPT's dense reduced symmetric augmented matrix."""
    nlp.validate_against_state(state)
    nlp.validate_hessian_symmetry()

    sigma_x = np.asarray(sigma_x, dtype=np.float64)
    sigma_s = np.asarray(sigma_s, dtype=np.float64)

    if sigma_x.shape != state.x.shape:
        raise ValueError(
            f"sigma_x must have shape {state.x.shape}; "
            f"received {sigma_x.shape}."
        )
    if sigma_s.shape != state.s.shape:
        raise ValueError(
            f"sigma_s must have shape {state.s.shape}; "
            f"received {sigma_s.shape}."
        )
    if not np.all(np.isfinite(sigma_x)):
        raise ValueError("sigma_x contains non-finite values.")
    if not np.all(np.isfinite(sigma_s)):
        raise ValueError("sigma_s contains non-finite values.")

    reg = regularization or NewtonRegularization()

    n_x = state.x.size
    n_s = state.s.size
    n_c = state.y_c.size
    n_d = state.y_d.size
    dimension = n_x + n_s + n_c + n_d

    matrix = np.zeros((dimension, dimension), dtype=np.float64)

    x_slice = slice(0, n_x)
    s_slice = slice(n_x, n_x + n_s)
    c_slice = slice(n_x + n_s, n_x + n_s + n_c)
    d_slice = slice(n_x + n_s + n_c, dimension)

    matrix[x_slice, x_slice] = nlp.hessian_lagrangian
    if n_x:
        matrix[x_slice, x_slice] += np.diag(
            sigma_x + reg.delta_x
        )

    if n_s:
        matrix[s_slice, s_slice] = np.diag(
            sigma_s + reg.delta_s
        )

    matrix[x_slice, c_slice] = nlp.jac_c.T
    matrix[c_slice, x_slice] = nlp.jac_c

    matrix[x_slice, d_slice] = nlp.jac_d.T
    matrix[d_slice, x_slice] = nlp.jac_d

    if n_s:
        identity_s = np.eye(n_s, dtype=np.float64)
        matrix[s_slice, d_slice] = -identity_s
        matrix[d_slice, s_slice] = -identity_s

    if n_c and reg.delta_c:
        matrix[c_slice, c_slice] = -reg.delta_c * np.eye(
            n_c,
            dtype=np.float64,
        )

    if n_d and reg.delta_d:
        matrix[d_slice, d_slice] = -reg.delta_d * np.eye(
            n_d,
            dtype=np.float64,
        )

    return matrix


def assemble_reduced_kkt_system(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    *,
    regularization: NewtonRegularization | None = None,
) -> ReducedKKTSystem:
    """Compute sigma terms and assemble the reduced matrix and right-hand side."""
    indices.validate_against_state(state)
    nlp.validate_against_state(state)
    residual.validate_against_state(state)

    sigma_x, sigma_s = compute_sigma(
        state,
        indices,
        slacks,
    )

    matrix = build_reduced_kkt_matrix(
        state,
        nlp,
        sigma_x,
        sigma_s,
        regularization=regularization,
    )

    rhs = build_reduced_kkt_rhs(
        residual,
        indices,
        slacks,
    )

    system = ReducedKKTSystem(
        matrix=matrix,
        rhs=rhs,
        sigma_x=sigma_x,
        sigma_s=sigma_s,
    )

    asymmetry = system.symmetry_error()
    if asymmetry > 1e-10:
        raise ValueError(
            "Reduced KKT matrix is not numerically symmetric; "
            f"maximum asymmetry is {asymmetry:.3e}."
        )

    return system


def split_reduced_solution(
    reduced_solution: FloatArray,
    state: PrimalDualState,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split a reduced solution into ``dx, ds, dy_c, dy_d``."""
    solution = np.asarray(reduced_solution, dtype=np.float64)

    if solution.ndim != 1:
        raise ValueError(
            "reduced_solution must be one-dimensional; "
            f"received shape {solution.shape}."
        )

    expected = (
        state.x.size
        + state.s.size
        + state.y_c.size
        + state.y_d.size
    )

    if solution.size != expected:
        raise ValueError(
            "reduced_solution has the wrong length: "
            f"{solution.size} != {expected}."
        )

    n_x = state.x.size
    n_s = state.s.size
    n_c = state.y_c.size

    x_end = n_x
    s_end = x_end + n_s
    c_end = s_end + n_c

    return (
        solution[:x_end].copy(),
        solution[x_end:s_end].copy(),
        solution[s_end:c_end].copy(),
        solution[c_end:].copy(),
    )


def recover_bound_multiplier_directions(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    delta_x: FloatArray,
    delta_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Recover ``dz_L, dz_U, dv_L, dv_U`` from complementarity rows."""
    delta_x = np.asarray(delta_x, dtype=np.float64)
    delta_s = np.asarray(delta_s, dtype=np.float64)

    if delta_x.shape != state.x.shape:
        raise ValueError(
            f"delta_x must have shape {state.x.shape}; "
            f"received {delta_x.shape}."
        )
    if delta_s.shape != state.s.shape:
        raise ValueError(
            f"delta_s must have shape {state.s.shape}; "
            f"received {delta_s.shape}."
        )

    slacks.validate_strictly_positive()

    delta_z_L = (
        -residual.z_L
        - state.z_L * delta_x[indices.x_lower]
    ) / slacks.x_lower

    delta_z_U = (
        -residual.z_U
        + state.z_U * delta_x[indices.x_upper]
    ) / slacks.x_upper

    delta_v_L = (
        -residual.v_L
        - state.v_L * delta_s[indices.s_lower]
    ) / slacks.s_lower

    delta_v_U = (
        -residual.v_U
        + state.v_U * delta_s[indices.s_upper]
    ) / slacks.s_upper

    return delta_z_L, delta_z_U, delta_v_L, delta_v_U


def build_full_kkt_matrix(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    *,
    regularization: NewtonRegularization | None = None,
) -> np.ndarray:
    """Build the full dense eight-block primal-dual KKT Jacobian.

    This function is intended primarily for tests and residual verification.
    The actual Newton solve uses the reduced system.
    """
    indices.validate_against_state(state)
    nlp.validate_against_state(state)
    nlp.validate_hessian_symmetry()

    if slacks.sizes != indices.sizes:
        raise ValueError(
            "Bound slack lengths must match bound index lengths: "
            f"{slacks.sizes} != {indices.sizes}."
        )

    slacks.validate_strictly_positive()
    reg = regularization or NewtonRegularization()

    sizes = state.block_sizes
    offsets = np.cumsum((0, *sizes))
    slices = tuple(
        slice(int(offsets[i]), int(offsets[i + 1]))
        for i in range(8)
    )
    (
        x_slice,
        s_slice,
        y_c_slice,
        y_d_slice,
        z_L_slice,
        z_U_slice,
        v_L_slice,
        v_U_slice,
    ) = slices

    matrix = np.zeros((state.size, state.size), dtype=np.float64)

    # Stationarity with respect to x.
    matrix[x_slice, x_slice] = nlp.hessian_lagrangian
    if state.x.size and reg.delta_x:
        matrix[x_slice, x_slice] += reg.delta_x * np.eye(
            state.x.size,
            dtype=np.float64,
        )
    matrix[x_slice, y_c_slice] = nlp.jac_c.T
    matrix[x_slice, y_d_slice] = nlp.jac_d.T

    for column, variable_index in enumerate(indices.x_lower):
        matrix[x_slice.start + variable_index, z_L_slice.start + column] = -1.0

    for column, variable_index in enumerate(indices.x_upper):
        matrix[x_slice.start + variable_index, z_U_slice.start + column] = 1.0

    # Stationarity with respect to s.
    if state.s.size and reg.delta_s:
        matrix[s_slice, s_slice] = reg.delta_s * np.eye(
            state.s.size,
            dtype=np.float64,
        )
    matrix[s_slice, y_d_slice] = -np.eye(
        state.s.size,
        dtype=np.float64,
    )

    for column, variable_index in enumerate(indices.s_lower):
        matrix[s_slice.start + variable_index, v_L_slice.start + column] = -1.0

    for column, variable_index in enumerate(indices.s_upper):
        matrix[s_slice.start + variable_index, v_U_slice.start + column] = 1.0

    # Equality and slack-equation feasibility.
    matrix[y_c_slice, x_slice] = nlp.jac_c
    matrix[y_d_slice, x_slice] = nlp.jac_d
    matrix[y_d_slice, s_slice] = -np.eye(
        state.s.size,
        dtype=np.float64,
    )

    if state.y_c.size and reg.delta_c:
        matrix[y_c_slice, y_c_slice] = -reg.delta_c * np.eye(
            state.y_c.size,
            dtype=np.float64,
        )
    if state.y_d.size and reg.delta_d:
        matrix[y_d_slice, y_d_slice] = -reg.delta_d * np.eye(
            state.y_d.size,
            dtype=np.float64,
        )

    # Complementarity rows for x lower bounds.
    for row, variable_index in enumerate(indices.x_lower):
        matrix[z_L_slice.start + row, x_slice.start + variable_index] = (
            state.z_L[row]
        )
        matrix[z_L_slice.start + row, z_L_slice.start + row] = (
            slacks.x_lower[row]
        )

    # Complementarity rows for x upper bounds.
    for row, variable_index in enumerate(indices.x_upper):
        matrix[z_U_slice.start + row, x_slice.start + variable_index] = (
            -state.z_U[row]
        )
        matrix[z_U_slice.start + row, z_U_slice.start + row] = (
            slacks.x_upper[row]
        )

    # Complementarity rows for s lower bounds.
    for row, variable_index in enumerate(indices.s_lower):
        matrix[v_L_slice.start + row, s_slice.start + variable_index] = (
            state.v_L[row]
        )
        matrix[v_L_slice.start + row, v_L_slice.start + row] = (
            slacks.s_lower[row]
        )

    # Complementarity rows for s upper bounds.
    for row, variable_index in enumerate(indices.s_upper):
        matrix[v_U_slice.start + row, s_slice.start + variable_index] = (
            -state.v_U[row]
        )
        matrix[v_U_slice.start + row, v_U_slice.start + row] = (
            slacks.s_upper[row]
        )

    return matrix


def compute_full_newton_residual(
    state: PrimalDualState,
    direction: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    *,
    regularization: NewtonRegularization | None = None,
) -> FloatArray:
    """Return ``K @ direction + residual`` for the full KKT system."""
    if direction.block_sizes != state.block_sizes:
        raise ValueError(
            "Direction and state block sizes must match: "
            f"{direction.block_sizes} != {state.block_sizes}."
        )

    matrix = build_full_kkt_matrix(
        state,
        nlp,
        indices,
        slacks,
        regularization=regularization,
    )

    return matrix @ direction.flatten() + residual.flatten()


def solve_primal_dual_newton_system(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: float,
    *,
    regularization: NewtonRegularization | None = None,
    require_positive_multipliers: bool = True,
    positivity_tolerance: float = 0.0,
    check_full_residual: bool = True,
) -> NewtonSolveResult:
    """Compute one dense IPOPT-style primal-dual Newton direction.

    Parameters
    ----------
    state
        Current primal-dual iterate.
    nlp
        NLP values and derivatives evaluated at ``state.x``.
    indices
        Locations of finite bounds.
    bounds
        Finite bound values aligned with ``indices``.
    mu
        Fixed nonnegative barrier parameter.
    regularization
        Optional IPOPT-style diagonal regularization.
    require_positive_multipliers
        Require all finite-bound multipliers to be strictly positive.
    positivity_tolerance
        Minimum accepted bound slack and multiplier value.
    check_full_residual
        Explicitly build the full KKT matrix and evaluate
        ``K @ direction + residual``.

    Returns
    -------
    NewtonSolveResult
        The complete eight-block direction and solve diagnostics.
    """
    state.validate_finite()
    indices.validate_against_state(state)
    bounds.validate_against_indices(indices)
    nlp.validate_against_state(state)
    nlp.validate_hessian_symmetry()

    if require_positive_multipliers:
        _validate_positive_bound_multipliers(
            state,
            tolerance=positivity_tolerance,
        )

    slacks = compute_bound_slacks(
        state,
        indices,
        bounds,
        require_strict_interior=True,
        positivity_tolerance=positivity_tolerance,
    )

    residual = compute_kkt_residual(
        state,
        nlp,
        indices,
        slacks,
        mu,
    )

    reduced_system = assemble_reduced_kkt_system(
        state,
        nlp,
        indices,
        slacks,
        residual,
        regularization=regularization,
    )

    try:
        reduced_solution = np.linalg.solve(
            reduced_system.matrix,
            reduced_system.rhs,
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The reduced primal-dual KKT matrix could not be solved. "
            "The system may be singular or numerically ill-conditioned. "
            "Consider checking the model derivatives or supplying nonzero "
            "NewtonRegularization values."
        ) from error

    delta_x, delta_s, delta_y_c, delta_y_d = split_reduced_solution(
        reduced_solution,
        state,
    )

    (
        delta_z_L,
        delta_z_U,
        delta_v_L,
        delta_v_U,
    ) = recover_bound_multiplier_directions(
        state,
        indices,
        slacks,
        residual,
        delta_x,
        delta_s,
    )

    direction = PrimalDualState(
        x=delta_x,
        s=delta_s,
        y_c=delta_y_c,
        y_d=delta_y_d,
        z_L=delta_z_L,
        z_U=delta_z_U,
        v_L=delta_v_L,
        v_U=delta_v_U,
    )
    direction.validate_finite()

    reduced_linear_residual = reduced_system.linear_residual(
        reduced_solution
    )
    reduced_residual_inf_norm = (
        0.0
        if reduced_linear_residual.size == 0
        else float(np.max(np.abs(reduced_linear_residual)))
    )

    if check_full_residual:
        full_linear_residual = compute_full_newton_residual(
            state,
            direction,
            nlp,
            indices,
            slacks,
            residual,
            regularization=regularization,
        )
        full_residual_inf_norm = (
            0.0
            if full_linear_residual.size == 0
            else float(np.max(np.abs(full_linear_residual)))
        )
    else:
        full_residual_inf_norm = float("nan")

    return NewtonSolveResult(
        direction=direction,
        residual=residual,
        slacks=slacks,
        reduced_system=reduced_system,
        reduced_solution=reduced_solution,
        reduced_residual_inf_norm=reduced_residual_inf_norm,
        full_residual_inf_norm=full_residual_inf_norm,
    )


def compute_primal_dual_newton_direction(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: float,
    *,
    regularization: NewtonRegularization | None = None,
    require_positive_multipliers: bool = True,
    positivity_tolerance: float = 0.0,
) -> PrimalDualState:
    """Return only the complete eight-block Newton direction.

    This is the compact public entry point.  Use
    :func:`solve_primal_dual_newton_system` when diagnostics are also needed.
    """
    result = solve_primal_dual_newton_system(
        state,
        nlp,
        indices,
        bounds,
        mu,
        regularization=regularization,
        require_positive_multipliers=require_positive_multipliers,
        positivity_tolerance=positivity_tolerance,
        check_full_residual=False,
    )
    return result.direction
