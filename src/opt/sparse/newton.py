"""Dense JAX primal-dual Newton kernels.

These functions mirror ``src.newton`` while remaining array-pure and suitable
for ``jax.jit``.  Input validation and Python exception translation stay in
the NumPy reference/orchestration layer.
"""

from __future__ import annotations

import jax.numpy as jnp

from .bounds import compute_bound_slacks
from .residuals import compute_kkt_residual
from .types import (
    BoundIndices,
    BoundSlacks,
    BoundValues,
    KKTResidual,
    NLPQuantities,
    NewtonRegularization,
    NewtonSolveResult,
    PrimalDualState,
    ReducedKKTSystem,
)


def compute_sigma(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
):
    sigma_x = jnp.zeros_like(state.x)
    sigma_s = jnp.zeros_like(state.s)

    sigma_x = sigma_x.at[indices.x_lower].add(
        state.z_L / slacks.x_lower
    )
    sigma_x = sigma_x.at[indices.x_upper].add(
        state.z_U / slacks.x_upper
    )
    sigma_s = sigma_s.at[indices.s_lower].add(
        state.v_L / slacks.s_lower
    )
    sigma_s = sigma_s.at[indices.s_upper].add(
        state.v_U / slacks.s_upper
    )
    return sigma_x, sigma_s


def build_reduced_kkt_rhs(
    residual: KKTResidual,
    indices: BoundIndices,
    slacks: BoundSlacks,
):
    b_x = -residual.x
    b_s = -residual.s

    b_x = b_x.at[indices.x_lower].add(
        -residual.z_L / slacks.x_lower
    )
    b_x = b_x.at[indices.x_upper].add(
        residual.z_U / slacks.x_upper
    )
    b_s = b_s.at[indices.s_lower].add(
        -residual.v_L / slacks.s_lower
    )
    b_s = b_s.at[indices.s_upper].add(
        residual.v_U / slacks.s_upper
    )

    return jnp.concatenate((b_x, b_s, -residual.y_c, -residual.y_d))


def build_reduced_kkt_matrix(
    state: PrimalDualState,
    nlp: NLPQuantities,
    sigma_x,
    sigma_s,
    *,
    regularization: NewtonRegularization | None = None,
):
    reg = regularization or NewtonRegularization()

    n_x = state.x.shape[0]
    n_s = state.s.shape[0]
    n_c = state.y_c.shape[0]
    n_d = state.y_d.shape[0]
    dtype = state.x.dtype

    xx = nlp.hessian_lagrangian + jnp.diag(
        sigma_x + reg.delta_x
    )
    xs = jnp.zeros((n_x, n_s), dtype=dtype)
    sc = jnp.zeros((n_s, n_c), dtype=dtype)
    cs = jnp.zeros((n_c, n_s), dtype=dtype)
    cd = jnp.zeros((n_c, n_d), dtype=dtype)
    dc = jnp.zeros((n_d, n_c), dtype=dtype)

    ss = jnp.diag(sigma_s + reg.delta_s)
    cc = -reg.delta_c * jnp.eye(n_c, dtype=dtype)
    dd = -reg.delta_d * jnp.eye(n_d, dtype=dtype)
    minus_identity = -jnp.eye(n_s, dtype=dtype)

    row_x = jnp.concatenate(
        (xx, xs, nlp.jac_c.T, nlp.jac_d.T),
        axis=1,
    )
    row_s = jnp.concatenate(
        (xs.T, ss, sc, minus_identity),
        axis=1,
    )
    row_c = jnp.concatenate(
        (nlp.jac_c, cs, cc, cd),
        axis=1,
    )
    row_d = jnp.concatenate(
        (nlp.jac_d, minus_identity, dc, dd),
        axis=1,
    )
    return jnp.concatenate((row_x, row_s, row_c, row_d), axis=0)


def assemble_reduced_kkt_system(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    *,
    regularization: NewtonRegularization | None = None,
) -> ReducedKKTSystem:
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    matrix = build_reduced_kkt_matrix(
        state,
        nlp,
        sigma_x,
        sigma_s,
        regularization=regularization,
    )
    rhs = build_reduced_kkt_rhs(residual, indices, slacks)
    return ReducedKKTSystem(matrix, rhs, sigma_x, sigma_s)


def split_reduced_solution(
    reduced_solution,
    state: PrimalDualState,
):
    n_x = state.x.shape[0]
    n_s = state.s.shape[0]
    n_c = state.y_c.shape[0]

    x_end = n_x
    s_end = x_end + n_s
    c_end = s_end + n_c
    return (
        reduced_solution[:x_end],
        reduced_solution[x_end:s_end],
        reduced_solution[s_end:c_end],
        reduced_solution[c_end:],
    )


def recover_bound_multiplier_directions(
    state: PrimalDualState,
    indices: BoundIndices,
    slacks: BoundSlacks,
    residual: KKTResidual,
    delta_x,
    delta_s,
):
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
):
    reg = regularization or NewtonRegularization()
    sizes = tuple(block.shape[0] for block in state)
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)

    (
        x0, s0, yc0, yd0, zl0, zu0, vl0, vu0, end
    ) = offsets
    n_x, n_s, n_c, n_d, n_zl, n_zu, n_vl, n_vu = sizes
    dtype = state.x.dtype

    matrix = jnp.zeros((end, end), dtype=dtype)

    matrix = matrix.at[x0:s0, x0:s0].set(
        nlp.hessian_lagrangian
        + reg.delta_x * jnp.eye(n_x, dtype=dtype)
    )
    matrix = matrix.at[x0:s0, yc0:yd0].set(nlp.jac_c.T)
    matrix = matrix.at[x0:s0, yd0:zl0].set(nlp.jac_d.T)

    if n_zl:
        cols = zl0 + jnp.arange(n_zl)
        rows = x0 + indices.x_lower
        matrix = matrix.at[rows, cols].set(-1.0)
    if n_zu:
        cols = zu0 + jnp.arange(n_zu)
        rows = x0 + indices.x_upper
        matrix = matrix.at[rows, cols].set(1.0)

    matrix = matrix.at[s0:yc0, s0:yc0].set(
        reg.delta_s * jnp.eye(n_s, dtype=dtype)
    )
    matrix = matrix.at[s0:yc0, yd0:zl0].set(
        -jnp.eye(n_s, dtype=dtype)
    )
    if n_vl:
        cols = vl0 + jnp.arange(n_vl)
        rows = s0 + indices.s_lower
        matrix = matrix.at[rows, cols].set(-1.0)
    if n_vu:
        cols = vu0 + jnp.arange(n_vu)
        rows = s0 + indices.s_upper
        matrix = matrix.at[rows, cols].set(1.0)

    matrix = matrix.at[yc0:yd0, x0:s0].set(nlp.jac_c)
    matrix = matrix.at[yd0:zl0, x0:s0].set(nlp.jac_d)
    matrix = matrix.at[yd0:zl0, s0:yc0].set(
        -jnp.eye(n_s, dtype=dtype)
    )
    matrix = matrix.at[yc0:yd0, yc0:yd0].set(
        -reg.delta_c * jnp.eye(n_c, dtype=dtype)
    )
    matrix = matrix.at[yd0:zl0, yd0:zl0].set(
        -reg.delta_d * jnp.eye(n_d, dtype=dtype)
    )

    if n_zl:
        rows = zl0 + jnp.arange(n_zl)
        matrix = matrix.at[rows, x0 + indices.x_lower].set(state.z_L)
        matrix = matrix.at[rows, rows].set(slacks.x_lower)
    if n_zu:
        rows = zu0 + jnp.arange(n_zu)
        matrix = matrix.at[rows, x0 + indices.x_upper].set(-state.z_U)
        matrix = matrix.at[rows, rows].set(slacks.x_upper)
    if n_vl:
        rows = vl0 + jnp.arange(n_vl)
        matrix = matrix.at[rows, s0 + indices.s_lower].set(state.v_L)
        matrix = matrix.at[rows, rows].set(slacks.s_lower)
    if n_vu:
        rows = vu0 + jnp.arange(n_vu)
        matrix = matrix.at[rows, s0 + indices.s_upper].set(-state.v_U)
        matrix = matrix.at[rows, rows].set(slacks.s_upper)

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
):
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
    check_full_residual: bool = True,
) -> NewtonSolveResult:
    slacks = compute_bound_slacks(state, indices, bounds)
    residual = compute_kkt_residual(state, nlp, indices, slacks, mu)
    system = assemble_reduced_kkt_system(
        state,
        nlp,
        indices,
        slacks,
        residual,
        regularization=regularization,
    )
    solution = jnp.linalg.solve(system.matrix, system.rhs)
    delta_x, delta_s, delta_y_c, delta_y_d = split_reduced_solution(
        solution,
        state,
    )
    delta_z_L, delta_z_U, delta_v_L, delta_v_U = (
        recover_bound_multiplier_directions(
            state,
            indices,
            slacks,
            residual,
            delta_x,
            delta_s,
        )
    )
    direction = PrimalDualState(
        delta_x,
        delta_s,
        delta_y_c,
        delta_y_d,
        delta_z_L,
        delta_z_U,
        delta_v_L,
        delta_v_U,
    )
    reduced_linear_residual = system.matrix @ solution - system.rhs
    reduced_norm = jnp.max(
        jnp.abs(reduced_linear_residual),
        initial=0.0,
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
        full_norm = jnp.max(jnp.abs(full_linear_residual), initial=0.0)
    else:
        full_norm = jnp.asarray(jnp.nan, dtype=state.x.dtype)

    return NewtonSolveResult(
        direction,
        residual,
        slacks,
        system,
        solution,
        reduced_norm,
        full_norm,
    )


def compute_primal_dual_newton_direction(
    state: PrimalDualState,
    nlp: NLPQuantities,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: float,
    *,
    regularization: NewtonRegularization | None = None,
):
    return solve_primal_dual_newton_system(
        state,
        nlp,
        indices,
        bounds,
        mu,
        regularization=regularization,
        check_full_residual=False,
    ).direction
