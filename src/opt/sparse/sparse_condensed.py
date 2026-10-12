"""Pure-Python/JAX condensed sparse projection solver.

This module keeps the primal--dual equations unchanged while eliminating the
inequality-slack Newton block algebraically.  Static row supports are prepared
before the solve; each Newton iteration evaluates only local Jacobian and
Hessian blocks and applies the condensed KKT operator matrix-free.

All authored code is Python.  Staged :func:`jax.jit` and :func:`jax.vmap`
kernels provide compiled CPU/CUDA execution without a custom C++ extension or
external sparse direct solver.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .batch_iteration import (
    BATCH_STATUS_ACTIVE,
    BATCH_STATUS_CONVERGED,
    BATCH_STATUS_MAXIMUM_ITERATIONS,
    BATCH_STATUS_NUMERICAL_FAILURE,
    BATCH_STATUS_STALLED,
    BatchedFixedMuResult,
    _state_rows_are_finite,
    _tree_batch_where,
)
from .bounds import compute_bound_slacks
from .newton import compute_sigma, recover_bound_multiplier_directions
from .step import apply_primal_dual_step, compute_fraction_to_boundary_steps
from .types import (
    BoundIndices,
    BoundSlacks,
    BoundValues,
    KKTResidual,
    NewtonRegularization,
    PrimalDualState,
)
from .gpu_kernels import (
    build_affine_constant_kkt_pallas_matvec,
    pallas_available,
)

VectorFunction = Callable[[jax.Array, jax.Array], jax.Array]
LocalDerivativeFunction = Callable[
    [jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


class SparseNumericData(NamedTuple):
    """Numerical sparse data evaluated once per Newton iteration."""

    residual: KKTResidual
    slacks: BoundSlacks
    sigma_x: jax.Array
    sigma_s: jax.Array
    equality_jacobian: jax.Array
    inequality_jacobian: jax.Array
    equality_hessian: jax.Array
    inequality_hessian: jax.Array


class SparseProjectionLinearization(NamedTuple):
    """Final condensed-KKT data reused by implicit adjoint solves."""

    combined_jacobian: jax.Array
    equality_hessian: jax.Array
    inequality_hessian: jax.Array
    inverse_slack: jax.Array
    primal_kkt_diagonal: jax.Array
    dual_kkt_diagonal: jax.Array
    preconditioner_diagonal: jax.Array


class SparseProjectionSolveResult(NamedTuple):
    """Forward result with the final sparse KKT linearization attached."""

    state: PrimalDualState
    status_codes: jax.Array
    iterations: jax.Array
    final_residual_inf_norm: jax.Array
    residual_history: jax.Array
    final_linearization: SparseProjectionLinearization

    @property
    def converged(self):
        return self.status_codes == BATCH_STATUS_CONVERGED


class SparseProjectionAdjointResult(NamedTuple):
    """Condensed final-KKT adjoint result for projection-input gradients."""

    x: jax.Array
    y_c: jax.Array
    y_d: jax.Array
    linear_status: jax.Array
    linear_residual_inf_norm: jax.Array
    linear_iterations: jax.Array


class SparseProjectionLossGradientResult(NamedTuple):
    """Fused projection, quadratic loss, and ``dL/dy_hat`` result."""

    losses: jax.Array
    y_tilde: jax.Array
    grad_y_hat: jax.Array
    forward_status_codes: jax.Array
    forward_iterations: jax.Array
    forward_residual_inf_norm: jax.Array
    adjoint_status_codes: jax.Array
    adjoint_iterations: jax.Array
    adjoint_residual_inf_norm: jax.Array


def _safe_support(support: jax.Array) -> jax.Array:
    """Return integer support indices (padding entries are expected to be zero)."""

    return jnp.asarray(support, dtype=jnp.int32)


def _is_jax_array(value) -> bool:
    return isinstance(value, jax.Array)


def local_jacobian_matvec(
    values: jax.Array,
    support: jax.Array,
    mask: jax.Array,
    vector: jax.Array,
    *,
    full_support: bool = False,
) -> jax.Array:
    """Apply a fixed-width row-sparse Jacobian to one vector."""

    if values.shape[0] == 0:
        return jnp.empty((0,), dtype=vector.dtype)
    gathered = vector[_safe_support(support)]
    if full_support:
        return jnp.sum(values * gathered, axis=-1)
    return jnp.sum(jnp.where(mask, values * gathered, 0.0), axis=-1)


def local_jacobian_transpose_matvec(
    values: jax.Array,
    support: jax.Array,
    mask: jax.Array,
    vector: jax.Array,
    n_variables: int,
    *,
    full_support: bool = False,
) -> jax.Array:
    """Apply the transpose of a fixed-width row-sparse Jacobian."""

    if values.shape[0] == 0:
        return jnp.zeros((n_variables,), dtype=vector.dtype)
    contributions = values * vector[:, None]
    if not full_support:
        contributions = jnp.where(mask, contributions, 0.0)
    result = jnp.zeros((n_variables,), dtype=values.dtype)
    return result.at[_safe_support(support)].add(contributions)


def local_jacobian_transpose_gather_matvec(
    values: jax.Array,
    column_rows: jax.Array,
    column_slots: jax.Array,
    column_mask: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Apply ``J.T`` using a precomputed column-oriented gather layout."""

    n_variables = int(column_rows.shape[0])
    if values.shape[0] == 0 or column_rows.shape[1] == 0:
        return jnp.zeros((n_variables,), dtype=vector.dtype)
    gathered_values = values[column_rows, column_slots]
    gathered_vector = vector[column_rows]
    return jnp.sum(
        jnp.where(column_mask, gathered_values * gathered_vector, 0.0),
        axis=-1,
    )


def _build_column_gather_structure(
    support: np.ndarray,
    mask: np.ndarray,
    n_variables: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded column adjacency for gather/reduce transpose products."""

    entries: list[list[tuple[int, int]]] = [[] for _ in range(n_variables)]
    for row in range(support.shape[0]):
        for slot in range(support.shape[1]):
            if mask[row, slot]:
                entries[int(support[row, slot])].append((row, slot))
    width = max((len(column_entries) for column_entries in entries), default=0)
    rows = np.zeros((n_variables, width), dtype=np.int32)
    slots = np.zeros((n_variables, width), dtype=np.int32)
    valid = np.zeros((n_variables, width), dtype=bool)
    for variable, column_entries in enumerate(entries):
        for slot_index, (row, local_slot) in enumerate(column_entries):
            rows[variable, slot_index] = row
            slots[variable, slot_index] = local_slot
            valid[variable, slot_index] = True
    return rows, slots, valid


def _build_column_gather_buckets(
    support: np.ndarray,
    mask: np.ndarray,
    n_variables: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """Build power-of-two degree buckets for transpose gather/reduce.

    A single padded column layout can waste substantial bandwidth when one
    variable has a much larger degree than the rest.  Bucketing variables by
    their next power-of-two degree keeps GPU reads regular while sharply
    reducing padding.  Each tuple contains ``(variables, rows, slots, mask)``.
    """

    entries: list[list[tuple[int, int]]] = [[] for _ in range(n_variables)]
    for row in range(support.shape[0]):
        for slot in range(support.shape[1]):
            if mask[row, slot]:
                entries[int(support[row, slot])].append((row, slot))

    grouped: dict[int, list[int]] = {}
    for variable, column_entries in enumerate(entries):
        degree = len(column_entries)
        if degree == 0:
            continue
        width = 1 << (degree - 1).bit_length()
        grouped.setdefault(width, []).append(variable)

    buckets = []
    for width in sorted(grouped):
        variables = np.asarray(grouped[width], dtype=np.int32)
        rows = np.zeros((variables.size, width), dtype=np.int32)
        slots = np.zeros((variables.size, width), dtype=np.int32)
        valid = np.zeros((variables.size, width), dtype=bool)
        for local_variable, variable in enumerate(variables):
            for position, (row, slot) in enumerate(entries[int(variable)]):
                rows[local_variable, position] = row
                slots[local_variable, position] = slot
                valid[local_variable, position] = True
        buckets.append((variables, rows, slots, valid))
    return tuple(buckets)


def _column_bucket_padding_fraction(
    buckets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]
) -> float:
    """Return the fraction of padded entries in degree buckets."""

    stored = sum(int(mask.size) for _, _, _, mask in buckets)
    valid = sum(int(np.count_nonzero(mask)) for _, _, _, mask in buckets)
    if stored == 0:
        return 0.0
    return float((stored - valid) / stored)


def local_constraint_hessian_matvec(
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    equality_multipliers: jax.Array,
    inequality_multipliers: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Apply the weighted sum of local constraint Hessians."""

    result = jnp.zeros_like(vector)
    if equality_hessian.shape[0]:
        local_vector = vector[_safe_support(equality_support)]
        local_result = jnp.einsum("mij,mj->mi", equality_hessian, local_vector)
        local_result = jnp.where(
            equality_mask,
            local_result * equality_multipliers[:, None],
            0.0,
        )
        result = result.at[_safe_support(equality_support)].add(local_result)
    if inequality_hessian.shape[0]:
        local_vector = vector[_safe_support(inequality_support)]
        local_result = jnp.einsum("mij,mj->mi", inequality_hessian, local_vector)
        local_result = jnp.where(
            inequality_mask,
            local_result * inequality_multipliers[:, None],
            0.0,
        )
        result = result.at[_safe_support(inequality_support)].add(local_result)
    return result


def local_constraint_hessian_diagonal(
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    equality_multipliers: jax.Array,
    inequality_multipliers: jax.Array,
    n_variables: int,
) -> jax.Array:
    """Scatter the diagonal of the weighted constraint Hessian."""

    dtype = (
        equality_hessian.dtype
        if equality_hessian.shape[0]
        else inequality_hessian.dtype
    )
    result = jnp.zeros((n_variables,), dtype=dtype)
    if equality_hessian.shape[0]:
        diagonal = jnp.diagonal(equality_hessian, axis1=-2, axis2=-1)
        contributions = jnp.where(
            equality_mask,
            diagonal * equality_multipliers[:, None],
            0.0,
        )
        result = result.at[_safe_support(equality_support)].add(contributions)
    if inequality_hessian.shape[0]:
        diagonal = jnp.diagonal(inequality_hessian, axis1=-2, axis2=-1)
        contributions = jnp.where(
            inequality_mask,
            diagonal * inequality_multipliers[:, None],
            0.0,
        )
        result = result.at[_safe_support(inequality_support)].add(contributions)
    return result


def _residual_inf_norm(residual: KKTResidual) -> jax.Array:
    """Infinity norm without concatenating all residual blocks.

    This is algebraically identical to ``max(abs(residual.flatten()))`` but
    avoids materializing a large temporary vector in every Newton iteration.
    """

    block_norms = tuple(
        jnp.max(jnp.abs(block), initial=0.0) for block in residual.blocks()
    )
    return jnp.max(jnp.stack(block_norms), initial=0.0)


def _assemble_numeric_data(
    state: PrimalDualState,
    parameters: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_jacobian: jax.Array,
    inequality_jacobian: jax.Array,
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> SparseNumericData:
    """Assemble residual and condensed-system data from derivative blocks."""

    equalities = equality_function(state.x, parameters)
    inequalities = inequality_function(state.x, parameters)
    return _assemble_numeric_data_from_constraint_values(
        state,
        parameters,
        equalities,
        inequalities,
        equality_jacobian,
        inequality_jacobian,
        equality_hessian,
        inequality_hessian,
        equality_support,
        equality_mask,
        inequality_support,
        inequality_mask,
        target_parameter_indices,
        indices,
        bounds,
        mu,
    )


def _assemble_numeric_data_from_constraint_values(
    state: PrimalDualState,
    parameters: jax.Array,
    equalities: jax.Array,
    inequalities: jax.Array,
    equality_jacobian: jax.Array,
    inequality_jacobian: jax.Array,
    equality_hessian: jax.Array,
    inequality_hessian: jax.Array,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> SparseNumericData:
    """Assemble sparse residual data from already evaluated constraints."""

    slacks = compute_bound_slacks(state, indices, bounds)

    target = parameters[target_parameter_indices]
    residual_x = state.x - target
    residual_x = residual_x + local_jacobian_transpose_matvec(
        equality_jacobian,
        equality_support,
        equality_mask,
        state.y_c,
        state.x.shape[0],
    )
    residual_x = residual_x + local_jacobian_transpose_matvec(
        inequality_jacobian,
        inequality_support,
        inequality_mask,
        state.y_d,
        state.x.shape[0],
    )
    residual_x = residual_x.at[indices.x_lower].add(-state.z_L)
    residual_x = residual_x.at[indices.x_upper].add(state.z_U)

    residual_s = -state.y_d
    residual_s = residual_s.at[indices.s_lower].add(-state.v_L)
    residual_s = residual_s.at[indices.s_upper].add(state.v_U)

    residual = KKTResidual(
        x=residual_x,
        s=residual_s,
        y_c=equalities,
        y_d=inequalities - state.s,
        z_L=slacks.x_lower * state.z_L - mu,
        z_U=slacks.x_upper * state.z_U - mu,
        v_L=slacks.s_lower * state.v_L - mu,
        v_U=slacks.s_upper * state.v_U - mu,
    )
    sigma_x, sigma_s = compute_sigma(state, indices, slacks)
    return SparseNumericData(
        residual=residual,
        slacks=slacks,
        sigma_x=sigma_x,
        sigma_s=sigma_s,
        equality_jacobian=equality_jacobian,
        inequality_jacobian=inequality_jacobian,
        equality_hessian=equality_hessian,
        inequality_hessian=inequality_hessian,
    )


def _evaluate_numeric_data(
    state: PrimalDualState,
    parameters: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_local_derivatives: LocalDerivativeFunction,
    inequality_local_derivatives: LocalDerivativeFunction,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu: jax.Array,
) -> SparseNumericData:
    equality_jacobian, equality_hessian = equality_local_derivatives(
        state.x, parameters
    )
    inequality_jacobian, inequality_hessian = inequality_local_derivatives(
        state.x, parameters
    )
    return _assemble_numeric_data(
        state,
        parameters,
        equality_function,
        inequality_function,
        equality_jacobian,
        inequality_jacobian,
        equality_hessian,
        inequality_hessian,
        equality_support,
        equality_mask,
        inequality_support,
        inequality_mask,
        target_parameter_indices,
        indices,
        bounds,
        mu,
    )


class PreparedSparseCondensedBatchSolver:
    """Persistent fixed-shape sparse solver with reusable JIT kernels.

    The symbolic functions, sparse supports, solver options, and batch shape are
    captured once.  The first :meth:`solve` call compiles the staged JAX kernels;
    subsequent calls with the same shapes execute the cached executables.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        equality_function: VectorFunction,
        inequality_function: VectorFunction,
        equality_local_derivatives: LocalDerivativeFunction,
        inequality_local_derivatives: LocalDerivativeFunction,
        equality_affine_offsets: VectorFunction | None = None,
        inequality_affine_offsets: VectorFunction | None = None,
        equality_constant_local_jacobian: jax.Array | None = None,
        inequality_constant_local_jacobian: jax.Array | None = None,
        equality_support: jax.Array,
        equality_mask: jax.Array,
        inequality_support: jax.Array,
        inequality_mask: jax.Array,
        target_parameter_indices: jax.Array,
        indices: BoundIndices,
        bounds: BoundValues,
        max_iterations: int = 50,
        residual_tolerance: float = 1e-9,
        tau: float = 0.995,
        step_scale: float = 1.0,
        minimum_step: float = 1e-14,
        regularization: NewtonRegularization | None = None,
        gmres_tolerance: float = 1e-9,
        gmres_atol: float = 0.0,
        gmres_restart: int = 30,
        gmres_maxiter: int = 50,
        use_preconditioner: bool = True,
        backtracking: bool = False,
        max_backtracking_steps: int = 8,
        backtracking_factor: float = 0.5,
        constraints_affine: bool = False,
        use_column_gather_transpose: bool | None = None,
        use_column_bucketed_transpose: bool = False,
        use_shared_constant_jacobian: bool = False,
        gmres_unroll: int = 1,
        arnoldi_prefix_block: int = 0,
        gpu_kkt_mode: str = "xla",
        pallas_kkt_block_size: int = 64,
        pallas_interpret: bool = False,
        store_residual_history: bool = True,
        adjoint_gmres_tolerance: float | None = None,
        adjoint_gmres_atol: float | None = None,
        adjoint_gmres_restart: int | None = None,
        adjoint_gmres_maxiter: int | None = None,
        adjoint_gmres_unroll: int | None = None,
        adjoint_arnoldi_prefix_block: int | None = None,
        adjoint_solver_mode: str = "gmres",
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer.")
        if not isinstance(gmres_maxiter, int) or gmres_maxiter < 1:
            raise ValueError("gmres_maxiter must be a positive integer.")
        if not isinstance(gmres_restart, int) or gmres_restart < 1:
            raise ValueError("gmres_restart must be a positive integer.")
        if residual_tolerance <= 0.0 or gmres_tolerance <= 0.0:
            raise ValueError("residual and GMRES tolerances must be positive.")
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must satisfy 0 < tau < 1.")
        if not 0.0 < step_scale <= 1.0:
            raise ValueError("step_scale must satisfy 0 < step_scale <= 1.")
        if not isinstance(max_backtracking_steps, int) or max_backtracking_steps < 1:
            raise ValueError("max_backtracking_steps must be a positive integer.")
        if not 0.0 < backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must satisfy 0 < factor < 1.")
        if not isinstance(gmres_unroll, int) or gmres_unroll < 1:
            raise ValueError("gmres_unroll must be a positive integer.")
        if not isinstance(arnoldi_prefix_block, int) or arnoldi_prefix_block < 0:
            raise ValueError("arnoldi_prefix_block must be a nonnegative integer.")
        if gpu_kkt_mode not in {"xla", "pallas"}:
            raise ValueError("gpu_kkt_mode must be 'xla' or 'pallas'.")
        if not isinstance(pallas_kkt_block_size, int) or pallas_kkt_block_size < 1:
            raise ValueError("pallas_kkt_block_size must be a positive integer.")
        if adjoint_solver_mode not in {"gmres", "dense"}:
            raise ValueError("adjoint_solver_mode must be 'gmres' or 'dense'.")
        adjoint_gmres_tolerance = (
            gmres_tolerance
            if adjoint_gmres_tolerance is None
            else float(adjoint_gmres_tolerance)
        )
        adjoint_gmres_atol = (
            gmres_atol if adjoint_gmres_atol is None else float(adjoint_gmres_atol)
        )
        adjoint_gmres_restart = (
            gmres_restart
            if adjoint_gmres_restart is None
            else int(adjoint_gmres_restart)
        )
        adjoint_gmres_maxiter = (
            gmres_maxiter
            if adjoint_gmres_maxiter is None
            else int(adjoint_gmres_maxiter)
        )
        adjoint_gmres_unroll = (
            gmres_unroll
            if adjoint_gmres_unroll is None
            else int(adjoint_gmres_unroll)
        )
        adjoint_arnoldi_prefix_block = (
            arnoldi_prefix_block
            if adjoint_arnoldi_prefix_block is None
            else int(adjoint_arnoldi_prefix_block)
        )
        if adjoint_gmres_tolerance <= 0.0:
            raise ValueError("adjoint_gmres_tolerance must be positive.")
        if adjoint_gmres_restart < 1:
            raise ValueError("adjoint_gmres_restart must be positive.")
        if adjoint_gmres_maxiter < 1:
            raise ValueError("adjoint_gmres_maxiter must be positive.")
        if adjoint_gmres_unroll < 1:
            raise ValueError("adjoint_gmres_unroll must be positive.")
        if adjoint_arnoldi_prefix_block < 0:
            raise ValueError("adjoint_arnoldi_prefix_block must be nonnegative.")

        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.residual_tolerance = float(residual_tolerance)
        self.gmres_unroll = int(gmres_unroll)
        self.arnoldi_prefix_block = int(arnoldi_prefix_block)
        self.gpu_kkt_mode = str(gpu_kkt_mode)
        self.pallas_kkt_block_size = int(pallas_kkt_block_size)
        self.store_residual_history = bool(store_residual_history)
        self.adjoint_gmres_tolerance = float(adjoint_gmres_tolerance)
        self.adjoint_gmres_atol = float(adjoint_gmres_atol)
        self.adjoint_gmres_restart = int(adjoint_gmres_restart)
        self.adjoint_gmres_maxiter = int(adjoint_gmres_maxiter)
        self.adjoint_gmres_unroll = int(adjoint_gmres_unroll)
        self.adjoint_arnoldi_prefix_block = int(adjoint_arnoldi_prefix_block)
        self.adjoint_solver_mode = str(adjoint_solver_mode)
        self.solve_calls = 0

        reg = regularization or NewtonRegularization()
        equality_support_np = np.asarray(equality_support, dtype=np.int32)
        inequality_support_np = np.asarray(inequality_support, dtype=np.int32)
        equality_mask_np = np.asarray(equality_mask, dtype=bool)
        inequality_mask_np = np.asarray(inequality_mask, dtype=bool)
        support_width = next(
            (
                int(array.shape[1])
                for array in (equality_support_np, inequality_support_np)
                if array.ndim == 2
            ),
            0,
        )
        if equality_support_np.ndim == 1 and equality_support_np.size == 0:
            equality_support_np = equality_support_np.reshape(0, support_width)
            equality_mask_np = equality_mask_np.reshape(0, support_width)
        if inequality_support_np.ndim == 1 and inequality_support_np.size == 0:
            inequality_support_np = inequality_support_np.reshape(0, support_width)
            inequality_mask_np = inequality_mask_np.reshape(0, support_width)
        equality_full_support = bool(np.all(equality_mask_np))
        inequality_full_support = bool(np.all(inequality_mask_np))
        combined_full_support = equality_full_support and inequality_full_support
        if equality_support_np.shape[0] == 0:
            combined_support_np = inequality_support_np
            combined_mask_np = inequality_mask_np
        elif inequality_support_np.shape[0] == 0:
            combined_support_np = equality_support_np
            combined_mask_np = equality_mask_np
        else:
            combined_support_np = np.concatenate(
                (equality_support_np, inequality_support_np), axis=0
            )
            combined_mask_np = np.concatenate(
                (equality_mask_np, inequality_mask_np), axis=0
            )
        n_variables = int(target_parameter_indices.shape[0])
        column_rows_np, column_slots_np, column_mask_np = (
            _build_column_gather_structure(
                combined_support_np,
                combined_mask_np,
                n_variables,
            )
        )
        column_buckets_np = _build_column_gather_buckets(
            combined_support_np,
            combined_mask_np,
            n_variables,
        )
        equality_support = jnp.asarray(equality_support_np, dtype=jnp.int32)
        inequality_support = jnp.asarray(inequality_support_np, dtype=jnp.int32)
        equality_mask = jnp.asarray(equality_mask_np, dtype=bool)
        inequality_mask = jnp.asarray(inequality_mask_np, dtype=bool)
        combined_support = jnp.asarray(combined_support_np, dtype=jnp.int32)
        combined_mask = jnp.asarray(combined_mask_np, dtype=bool)
        column_rows = jnp.asarray(column_rows_np, dtype=jnp.int32)
        column_slots = jnp.asarray(column_slots_np, dtype=jnp.int32)
        column_mask = jnp.asarray(column_mask_np, dtype=bool)
        column_buckets = tuple(
            (
                jnp.asarray(variables, dtype=jnp.int32),
                jnp.asarray(rows, dtype=jnp.int32),
                jnp.asarray(slots, dtype=jnp.int32),
                jnp.asarray(valid, dtype=bool),
            )
            for variables, rows, slots, valid in column_buckets_np
        )
        # CPU XLA currently lowers indexed scatter-add more efficiently for
        # these small fixed-width rows.  The column gather layout remains
        # useful on accelerator backends where irregular scatters may require
        # atomics and poorly coalesced writes.
        if use_column_gather_transpose is None:
            use_column_gather_transpose = jax.default_backend() != "cpu"
        use_column_gather_transpose = bool(use_column_gather_transpose)
        use_column_bucketed_transpose = bool(use_column_bucketed_transpose)
        use_shared_constant_jacobian = bool(use_shared_constant_jacobian)
        if use_column_gather_transpose and use_column_bucketed_transpose:
            raise ValueError(
                "Global gather and bucketed gather transpose modes are mutually exclusive."
            )
        target_parameter_indices = jnp.asarray(
            target_parameter_indices, dtype=jnp.int32
        )

        constraints_affine = bool(constraints_affine)

        equality_constant_local_jacobian = (
            None
            if equality_constant_local_jacobian is None
            else jnp.asarray(equality_constant_local_jacobian, dtype=jnp.float64)
        )
        inequality_constant_local_jacobian = (
            None
            if inequality_constant_local_jacobian is None
            else jnp.asarray(inequality_constant_local_jacobian, dtype=jnp.float64)
        )
        combined_constant_local_jacobian = None
        combined_constant_local_jacobian_squared = None
        if (
            equality_constant_local_jacobian is not None
            and inequality_constant_local_jacobian is not None
        ):
            if equality_constant_local_jacobian.shape[0] == 0:
                combined_constant_local_jacobian = inequality_constant_local_jacobian
            elif inequality_constant_local_jacobian.shape[0] == 0:
                combined_constant_local_jacobian = equality_constant_local_jacobian
            else:
                combined_constant_local_jacobian = jnp.concatenate(
                    (
                        equality_constant_local_jacobian,
                        inequality_constant_local_jacobian,
                    ),
                    axis=0,
                )
            combined_constant_local_jacobian_squared = (
                combined_constant_local_jacobian
                * combined_constant_local_jacobian
            )

        pallas_kkt_matvec = None
        if gpu_kkt_mode == "pallas":
            if not use_shared_constant_jacobian:
                raise ValueError(
                    "The Pallas KKT kernel requires shared constant Jacobians."
                )
            if not constraints_affine:
                raise ValueError(
                    "The Pallas KKT kernel currently supports affine constraints only."
                )
            if combined_constant_local_jacobian is None:
                raise ValueError(
                    "The Pallas KKT kernel requires a parameter-independent Jacobian."
                )
            if not pallas_available():
                raise RuntimeError("Pallas is unavailable in this JAX installation.")
            pallas_kkt_matvec = build_affine_constant_kkt_pallas_matvec(
                jacobian=combined_constant_local_jacobian,
                row_support=combined_support,
                row_mask=combined_mask,
                column_rows=column_rows,
                column_slots=column_slots,
                column_mask=column_mask,
                n_variables=n_variables,
                block_size=pallas_kkt_block_size,
                interpret=bool(pallas_interpret),
            )

        derivative_batch = jax.vmap(
            lambda state, params: (
                *equality_local_derivatives(state.x, params),
                *inequality_local_derivatives(state.x, params),
            ),
            in_axes=(0, 0),
        )
        constraint_values_batch = jax.vmap(
            lambda state, params: (
                equality_function(state.x, params),
                inequality_function(state.x, params),
            ),
            in_axes=(0, 0),
        )
        affine_offsets_batch = None
        if equality_affine_offsets is not None and inequality_affine_offsets is not None:
            affine_offsets_batch = jax.vmap(
                lambda state, params: (
                    equality_affine_offsets(state.x, params),
                    inequality_affine_offsets(state.x, params),
                ),
                in_axes=(0, 0),
            )

        evaluate_dynamic_one = lambda state, params, mu_i: _evaluate_numeric_data(
            state,
            params,
            equality_function,
            inequality_function,
            equality_local_derivatives,
            inequality_local_derivatives,
            equality_support,
            equality_mask,
            inequality_support,
            inequality_mask,
            target_parameter_indices,
            indices,
            bounds,
            mu_i,
        )
        evaluate_dynamic_batch = jax.vmap(
            evaluate_dynamic_one, in_axes=(0, 0, 0)
        )

        equality_jacobian_batch_axis = (
            None
            if use_shared_constant_jacobian
            and equality_constant_local_jacobian is not None
            else 0
        )
        inequality_jacobian_batch_axis = (
            None
            if use_shared_constant_jacobian
            and inequality_constant_local_jacobian is not None
            else 0
        )

        def evaluate_fixed_one(
            state,
            params,
            mu_i,
            eq_j,
            ineq_j,
            eq_offset,
            ineq_offset,
        ):
            equalities = local_jacobian_matvec(
                eq_j,
                equality_support,
                equality_mask,
                state.x,
                full_support=equality_full_support,
            ) + eq_offset
            inequalities = local_jacobian_matvec(
                ineq_j,
                inequality_support,
                inequality_mask,
                state.x,
                full_support=inequality_full_support,
            ) + ineq_offset
            empty_hessian = jnp.empty((0, 0, 0), dtype=state.x.dtype)
            numeric = _assemble_numeric_data_from_constraint_values(
                state,
                params,
                equalities,
                inequalities,
                eq_j,
                ineq_j,
                empty_hessian,
                empty_hessian,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                target_parameter_indices,
                indices,
                bounds,
                mu_i,
            )
            # Keep constant Jacobians outside vmap output so they remain truly
            # unbatched in the Newton/GMRES loop.
            return (
                numeric.residual,
                numeric.slacks,
                numeric.sigma_x,
                numeric.sigma_s,
            )

        evaluate_fixed_batch = jax.vmap(
            evaluate_fixed_one,
            in_axes=(
                0,
                0,
                0,
                equality_jacobian_batch_axis,
                inequality_jacobian_batch_axis,
                0,
                0,
            ),
        )
        residual_norm_batch = jax.vmap(_residual_inf_norm)

        def evaluate_numeric(state, params, mus, fixed_derivatives):
            if constraints_affine:
                eq_j, ineq_j, eq_offset, ineq_offset = fixed_derivatives
                residual, slacks, sigma_x, sigma_s = evaluate_fixed_batch(
                    state,
                    params,
                    mus,
                    eq_j,
                    ineq_j,
                    eq_offset,
                    ineq_offset,
                )
                empty_hessian = jnp.empty((0, 0, 0), dtype=state.x.dtype)
                return SparseNumericData(
                    residual=residual,
                    slacks=slacks,
                    sigma_x=sigma_x,
                    sigma_s=sigma_s,
                    equality_jacobian=eq_j,
                    inequality_jacobian=ineq_j,
                    equality_hessian=empty_hessian,
                    inequality_hessian=empty_hessian,
                )
            return evaluate_dynamic_batch(state, params, mus)

        def build_fixed_derivatives(reference_state, parameters):
            """Evaluate affine derivative data once for a complete batch."""

            if not constraints_affine:
                return None
            if (
                equality_constant_local_jacobian is None
                or inequality_constant_local_jacobian is None
            ):
                eq_j_dynamic, _eq_h, ineq_j_dynamic, _ineq_h = derivative_batch(
                    reference_state, parameters
                )
                eq_j = (
                    eq_j_dynamic
                    if equality_constant_local_jacobian is None
                    else equality_constant_local_jacobian
                )
                ineq_j = (
                    ineq_j_dynamic
                    if inequality_constant_local_jacobian is None
                    else inequality_constant_local_jacobian
                )
            else:
                if use_shared_constant_jacobian:
                    eq_j = equality_constant_local_jacobian
                    ineq_j = inequality_constant_local_jacobian
                else:
                    eq_j = jnp.broadcast_to(
                        equality_constant_local_jacobian,
                        (batch_size,) + equality_constant_local_jacobian.shape,
                    )
                    ineq_j = jnp.broadcast_to(
                        inequality_constant_local_jacobian,
                        (batch_size,) + inequality_constant_local_jacobian.shape,
                    )

            if affine_offsets_batch is not None:
                equality_offset, inequality_offset = affine_offsets_batch(
                    reference_state, parameters
                )
            else:
                equality_values, inequality_values = constraint_values_batch(
                    reference_state, parameters
                )
                equality_offset = equality_values - _batched_local_jacobian_matvec(
                    eq_j,
                    equality_support,
                    equality_mask,
                    reference_state.x,
                    full_support=equality_full_support,
                )
                inequality_offset = (
                    inequality_values
                    - _batched_local_jacobian_matvec(
                        ineq_j,
                        inequality_support,
                        inequality_mask,
                        reference_state.x,
                        full_support=inequality_full_support,
                    )
                )
            return eq_j, ineq_j, equality_offset, inequality_offset

        solve_batch = lambda state, numeric: _solve_condensed_direction_batch(
            state,
            numeric,
            equality_support,
            equality_mask,
            inequality_support,
            inequality_mask,
            indices,
            reg,
            tolerance=gmres_tolerance,
            absolute_tolerance=gmres_atol,
            max_iterations=gmres_maxiter,
            restart=gmres_restart,
            use_preconditioner=use_preconditioner,
            constraints_affine=constraints_affine,
            equality_full_support=equality_full_support,
            inequality_full_support=inequality_full_support,
            combined_full_support=combined_full_support,
            combined_support=combined_support,
            combined_mask=combined_mask,
            column_rows=column_rows if use_column_gather_transpose else None,
            column_slots=column_slots if use_column_gather_transpose else None,
            column_mask=column_mask if use_column_gather_transpose else None,
            column_buckets=(
                column_buckets if use_column_bucketed_transpose else None
            ),
            constant_combined_jacobian=(
                combined_constant_local_jacobian
                if use_shared_constant_jacobian
                else None
            ),
            constant_combined_jacobian_squared=(
                combined_constant_local_jacobian_squared
                if use_shared_constant_jacobian
                else None
            ),
            affine_kkt_matvec=pallas_kkt_matvec,
            gmres_unroll=gmres_unroll,
            arnoldi_prefix_block=arnoldi_prefix_block,
        )
        def build_final_linearization(state, numeric):
            return _build_sparse_projection_linearization(
                state,
                numeric,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                reg,
                use_preconditioner=use_preconditioner,
                constraints_affine=constraints_affine,
                equality_full_support=equality_full_support,
                inequality_full_support=inequality_full_support,
                combined_full_support=combined_full_support,
                combined_support=combined_support,
                combined_mask=combined_mask,
                constant_combined_jacobian=(
                    combined_constant_local_jacobian
                    if use_shared_constant_jacobian
                    else None
                ),
                constant_combined_jacobian_squared=(
                    combined_constant_local_jacobian_squared
                    if use_shared_constant_jacobian
                    else None
                ),
            )

        solve_yhat_adjoint_batch = (
            lambda state, linearization, grad_x, initial_solution: _solve_condensed_yhat_adjoint_batch(
                state,
                linearization,
                grad_x,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                reg,
                tolerance=adjoint_gmres_tolerance,
                absolute_tolerance=adjoint_gmres_atol,
                max_iterations=adjoint_gmres_maxiter,
                restart=adjoint_gmres_restart,
                use_preconditioner=use_preconditioner,
                constraints_affine=constraints_affine,
                equality_full_support=equality_full_support,
                inequality_full_support=inequality_full_support,
                combined_full_support=combined_full_support,
                combined_support=combined_support,
                combined_mask=combined_mask,
                column_rows=(column_rows if use_column_gather_transpose else None),
                column_slots=(column_slots if use_column_gather_transpose else None),
                column_mask=(column_mask if use_column_gather_transpose else None),
                column_buckets=(
                    column_buckets if use_column_bucketed_transpose else None
                ),
                affine_kkt_matvec=pallas_kkt_matvec,
                gmres_unroll=adjoint_gmres_unroll,
                arnoldi_prefix_block=adjoint_arnoldi_prefix_block,
                solver_mode=adjoint_solver_mode,
                initial_solution=initial_solution,
            )
        )
        step_limit_batch = jax.vmap(
            lambda state, direction: compute_fraction_to_boundary_steps(
                state, direction, indices, bounds, tau
            ),
            in_axes=(0, 0),
        )
        update_batch = jax.vmap(
            lambda state, direction, alpha_primal, alpha_dual: apply_primal_dual_step(
                state,
                direction,
                alpha_primal=alpha_primal,
                alpha_dual=alpha_dual,
            ),
            in_axes=(0, 0, 0, 0),
        )

        def select_backtracked_candidate(
            state: PrimalDualState,
            direction: PrimalDualState,
            alpha_primal: jax.Array,
            alpha_dual: jax.Array,
            params: jax.Array,
            mus: jax.Array,
            current_norm: jax.Array,
            fixed_derivatives,
        ) -> tuple[PrimalDualState, jax.Array]:
            if not backtracking:
                candidate = update_batch(
                    state, direction, alpha_primal, alpha_dual
                )
                return candidate, jnp.ones((batch_size,), dtype=bool)

            accepted0 = jnp.zeros((batch_size,), dtype=bool)
            selected0 = state
            scale0 = jnp.ones((batch_size,), dtype=state.x.dtype)

            def body(_, carry):
                scale, accepted, selected = carry
                candidate = update_batch(
                    state,
                    direction,
                    alpha_primal * scale,
                    alpha_dual * scale,
                )
                trial_numeric = evaluate_numeric(
                    candidate, params, mus, fixed_derivatives
                )
                trial_norm = residual_norm_batch(trial_numeric.residual)
                accept_now = (
                    ~accepted
                    & _state_rows_are_finite(candidate)
                    & jnp.isfinite(trial_norm)
                    & (trial_norm <= current_norm)
                )
                selected = _tree_batch_where(accept_now, candidate, selected)
                accepted = accepted | accept_now
                scale = jnp.where(accepted, scale, scale * backtracking_factor)
                return scale, accepted, selected

            _, accepted, selected = jax.lax.fori_loop(
                0,
                max_backtracking_steps,
                body,
                (scale0, accepted0, selected0),
            )
            return selected, accepted

        def precheck_status(
            state,
            status,
            stopping_iterations,
            residual_norm,
            iteration,
        ):
            active = status == BATCH_STATUS_ACTIVE
            finite = jnp.isfinite(residual_norm) & _state_rows_are_finite(state)
            failed = active & ~finite
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(
                failed, iteration, stopping_iterations
            )

            active = status == BATCH_STATUS_ACTIVE
            converged = active & (residual_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            stopping_iterations = jnp.where(
                converged, iteration, stopping_iterations
            )
            return status, stopping_iterations

        def advance_after_direction(
            state,
            status,
            stopping_iterations,
            direction,
            linear_residual,
            residual_norm,
            iteration,
            params,
            mus,
            fixed_derivatives,
        ):
            forcing = jnp.minimum(
                0.1, jnp.sqrt(jnp.maximum(residual_norm, 0.0))
            )
            allowed_linear_residual = jnp.maximum(
                1e-10, forcing * jnp.maximum(residual_norm, 1e-16)
            )
            active = status == BATCH_STATUS_ACTIVE
            direction_ok = (
                _state_rows_are_finite(direction)
                & jnp.isfinite(linear_residual)
                & (linear_residual <= allowed_linear_residual)
            )
            failed = active & ~direction_ok
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(
                failed, iteration, stopping_iterations
            )

            active = status == BATCH_STATUS_ACTIVE
            limits = step_limit_batch(state, direction)
            alpha_primal = step_scale * limits.alpha_primal
            alpha_dual = step_scale * limits.alpha_dual
            stalled = active & (
                jnp.minimum(alpha_primal, alpha_dual) <= minimum_step
            )
            status = jnp.where(stalled, BATCH_STATUS_STALLED, status)
            stopping_iterations = jnp.where(
                stalled, iteration, stopping_iterations
            )

            candidate, accepted = select_backtracked_candidate(
                state,
                direction,
                alpha_primal,
                alpha_dual,
                params,
                mus,
                residual_norm,
                fixed_derivatives,
            )
            should_update = status == BATCH_STATUS_ACTIVE
            failed = should_update & (
                ~accepted | ~_state_rows_are_finite(candidate)
            )
            status = jnp.where(failed, BATCH_STATUS_NUMERICAL_FAILURE, status)
            stopping_iterations = jnp.where(
                failed, iteration, stopping_iterations
            )
            should_update = status == BATCH_STATUS_ACTIVE
            state = _tree_batch_where(should_update, candidate, state)
            return state, status, stopping_iterations

        def solve_all_core(initial_state, parameters, mu_values):
            fixed_derivatives = build_fixed_derivatives(
                initial_state, parameters
            )

            status0 = jnp.full(
                (batch_size,), BATCH_STATUS_ACTIVE, dtype=jnp.int32
            )
            stopping0 = jnp.full(
                (batch_size,), max_iterations, dtype=jnp.int32
            )
            history0 = jnp.zeros(
                (max_iterations + 1, batch_size)
                if store_residual_history
                else (0, batch_size),
                dtype=initial_state.x.dtype,
            )
            iteration0 = jnp.asarray(0, dtype=jnp.int32)

            zero_direction = jax.tree_util.tree_map(jnp.zeros_like, initial_state)
            zero_info = jnp.zeros((batch_size,), dtype=jnp.int32)
            zero_linear_residual = jnp.zeros(
                (batch_size,), dtype=initial_state.x.dtype
            )

            def loop_condition(carry):
                iteration, _state, status, _stopping, _history = carry
                return (iteration < max_iterations) & jnp.any(
                    status == BATCH_STATUS_ACTIVE
                )

            def loop_body(carry):
                iteration, state, status, stopping_iterations, history = carry
                numeric = evaluate_numeric(
                    state, parameters, mu_values, fixed_derivatives
                )
                residual_norm = residual_norm_batch(numeric.residual)
                if store_residual_history:
                    history = history.at[iteration].set(residual_norm)
                status, stopping_iterations = precheck_status(
                    state,
                    status,
                    stopping_iterations,
                    residual_norm,
                    iteration,
                )

                def perform_solve(_):
                    return solve_batch(state, numeric)

                def skip_solve(_):
                    return zero_direction, zero_info, zero_linear_residual

                direction, _info, linear_residual = jax.lax.cond(
                    jnp.any(status == BATCH_STATUS_ACTIVE),
                    perform_solve,
                    skip_solve,
                    operand=None,
                )
                state, status, stopping_iterations = advance_after_direction(
                    state,
                    status,
                    stopping_iterations,
                    direction,
                    linear_residual,
                    residual_norm,
                    iteration,
                    parameters,
                    mu_values,
                    fixed_derivatives,
                )
                return (
                    iteration + 1,
                    state,
                    status,
                    stopping_iterations,
                    history,
                )

            iteration, state, status, stopping_iterations, history = (
                jax.lax.while_loop(
                    loop_condition,
                    loop_body,
                    (iteration0, initial_state, status0, stopping0, history0),
                )
            )

            final_numeric = evaluate_numeric(
                state, parameters, mu_values, fixed_derivatives
            )
            final_norm = residual_norm_batch(final_numeric.residual)
            active = status == BATCH_STATUS_ACTIVE
            finite = jnp.isfinite(final_norm) & _state_rows_are_finite(state)
            status = jnp.where(
                active & ~finite, BATCH_STATUS_NUMERICAL_FAILURE, status
            )
            active = status == BATCH_STATUS_ACTIVE
            converged = active & (final_norm <= residual_tolerance)
            status = jnp.where(converged, BATCH_STATUS_CONVERGED, status)
            status = jnp.where(
                status == BATCH_STATUS_ACTIVE,
                BATCH_STATUS_MAXIMUM_ITERATIONS,
                status,
            )
            stopping_iterations = jnp.where(
                converged, max_iterations, stopping_iterations
            )

            # The previous fixed-length implementation evaluated the frozen
            # converged state on every remaining iteration.  Filling the tail
            # with the final norm preserves the public history shape and values
            # while avoiding those expensive evaluations and linear solves.
            if store_residual_history:
                tail = jnp.arange(max_iterations + 1, dtype=jnp.int32)[:, None]
                history = jnp.where(tail >= iteration, final_norm[None, :], history)
            return (
                state,
                status,
                stopping_iterations,
                final_norm,
                history,
                final_numeric,
            )

        def solve_all(initial_state, parameters, mu_values):
            state, status, iterations, final_norm, history, _numeric = solve_all_core(
                initial_state, parameters, mu_values
            )
            return BatchedFixedMuResult(
                state=state,
                status_codes=status,
                iterations=iterations,
                final_residual_inf_norm=final_norm,
                residual_history=history,
            )

        def solve_all_with_linearization(initial_state, parameters, mu_values):
            state, status, iterations, final_norm, history, final_numeric = solve_all_core(
                initial_state, parameters, mu_values
            )
            return SparseProjectionSolveResult(
                state=state,
                status_codes=status,
                iterations=iterations,
                final_residual_inf_norm=final_norm,
                residual_history=history,
                final_linearization=build_final_linearization(state, final_numeric),
            )

        def solve_yhat_adjoint(
            final_state, final_linearization, grad_x, initial_solution
        ):
            return solve_yhat_adjoint_batch(
                final_state, final_linearization, grad_x, initial_solution
            )

        def solve_yhat_adjoint_recompute(
            final_state, parameters, mu_values, grad_x, initial_solution
        ):
            fixed_derivatives = build_fixed_derivatives(final_state, parameters)
            final_numeric = evaluate_numeric(
                final_state, parameters, mu_values, fixed_derivatives
            )
            final_linearization = build_final_linearization(
                final_state, final_numeric
            )
            return solve_yhat_adjoint_batch(
                final_state, final_linearization, grad_x, initial_solution
            )

        def make_loss_gradient_function(q_mode):
            def solve_loss_gradient(
                initial_state,
                parameters,
                mu_values,
                q_data,
                valid_mask,
                initial_adjoint,
            ):
                forward = solve_all_with_linearization(
                    initial_state, parameters, mu_values
                )
                y_tilde = forward.state.x
                if q_mode == "dense":
                    q_y = y_tilde @ q_data.T
                else:
                    q_y = q_data[None, :] * y_tilde
                losses = jnp.sum(y_tilde * q_y, axis=1)
                grad_x = jnp.where(valid_mask[:, None], 2.0 * q_y, 0.0)
                adjoint = solve_yhat_adjoint_batch(
                    forward.state,
                    forward.final_linearization,
                    grad_x,
                    initial_adjoint,
                )
                return SparseProjectionLossGradientResult(
                    losses=losses,
                    y_tilde=y_tilde,
                    grad_y_hat=adjoint.x,
                    forward_status_codes=forward.status_codes,
                    forward_iterations=forward.iterations,
                    forward_residual_inf_norm=forward.final_residual_inf_norm,
                    adjoint_status_codes=adjoint.linear_status,
                    adjoint_iterations=adjoint.linear_iterations,
                    adjoint_residual_inf_norm=adjoint.linear_residual_inf_norm,
                )
            return solve_loss_gradient

        # One compiled executable owns the complete Newton loop.  The first
        # fixed-shape solve traces/compiles it; later batches perform one JAX
        # dispatch and stop as soon as no row remains active.
        self._solve_all_function = jax.jit(solve_all)
        self._solve_all_with_linearization_function = jax.jit(
            solve_all_with_linearization
        )
        self._solve_yhat_adjoint_function = jax.jit(solve_yhat_adjoint)
        self._solve_yhat_adjoint_recompute_function = jax.jit(
            solve_yhat_adjoint_recompute
        )
        self._solve_loss_gradient_diagonal_function = jax.jit(
            make_loss_gradient_function("diagonal")
        )
        self._solve_loss_gradient_dense_function = jax.jit(
            make_loss_gradient_function("dense")
        )


    def solve(
        self,
        *,
        initial_state: PrimalDualState,
        parameter_values: jax.Array,
        mu,
    ) -> BatchedFixedMuResult:
        """Solve one batch with one reusable, dynamically terminating graph."""

        target_device = next(iter(initial_state.x.devices()))
        parameters = jax.device_put(
            jnp.asarray(parameter_values, dtype=initial_state.x.dtype),
            target_device,
        )
        if parameters.ndim != 2:
            raise ValueError("parameter_values must be a rank-two array.")
        if parameters.shape[0] != self.batch_size:
            raise ValueError(
                "Prepared solver batch-size mismatch: expected "
                f"{self.batch_size}, received {parameters.shape[0]}."
            )

        if isinstance(mu, jax.Array):
            mu_values = jax.device_put(
                jnp.asarray(mu, dtype=initial_state.x.dtype), target_device
            )
        else:
            mu_host = np.asarray(mu, dtype=np.float64)
            if not np.all(np.isfinite(mu_host)) or np.any(mu_host < 0.0):
                raise ValueError("mu must contain finite nonnegative values.")
            mu_values = jax.device_put(
                jnp.asarray(mu_host, dtype=initial_state.x.dtype), target_device
            )
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        if mu_values.shape != (self.batch_size,):
            raise ValueError(
                f"mu must be scalar or have shape ({self.batch_size},)."
            )

        result = self._solve_all_function(initial_state, parameters, mu_values)
        self.solve_calls += 1
        return result

    def solve_with_linearization(
        self,
        *,
        initial_state: PrimalDualState,
        parameter_values: jax.Array,
        mu,
    ) -> SparseProjectionSolveResult:
        """Solve and retain the final KKT linearization for an adjoint call."""

        target_device = next(iter(initial_state.x.devices()))
        parameters = jax.device_put(
            jnp.asarray(parameter_values, dtype=initial_state.x.dtype), target_device
        )
        mu_values = jax.device_put(
            jnp.asarray(mu, dtype=initial_state.x.dtype), target_device
        )
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        result = self._solve_all_with_linearization_function(
            initial_state, parameters, mu_values
        )
        self.solve_calls += 1
        return result

    def solve_yhat_adjoint(
        self,
        *,
        final_state: PrimalDualState,
        grad_x: jax.Array,
        final_linearization: SparseProjectionLinearization | None = None,
        parameter_values: jax.Array | None = None,
        mu=None,
        initial_solution: jax.Array | None = None,
    ) -> SparseProjectionAdjointResult:
        """Solve the sparse final-KKT adjoint for a projection loss.

        ``grad_x`` is the per-sample loss cotangent with respect to the final
        projected primal output.  The returned ``x`` block is the gradient with
        respect to the distance-minimization target ``y_hat``.
        """

        target_device = next(iter(final_state.x.devices()))
        gradient = jax.device_put(
            jnp.asarray(grad_x, dtype=final_state.x.dtype), target_device
        )
        expected_gradient_shape = (self.batch_size, final_state.x.shape[1])
        if tuple(gradient.shape) != expected_gradient_shape:
            raise ValueError(
                f"grad_x must have shape {expected_gradient_shape}; "
                f"received {tuple(gradient.shape)}."
            )
        width = final_state.x.shape[1] + final_state.y_c.shape[1] + final_state.y_d.shape[1]
        if initial_solution is None:
            initial = jnp.zeros((self.batch_size, width), dtype=final_state.x.dtype)
        else:
            initial = jax.device_put(
                jnp.asarray(initial_solution, dtype=final_state.x.dtype), target_device
            )
            if tuple(initial.shape) != (self.batch_size, width):
                raise ValueError(
                    f"initial_solution must have shape {(self.batch_size, width)}; "
                    f"received {tuple(initial.shape)}."
                )
        if final_linearization is not None:
            return self._solve_yhat_adjoint_function(
                final_state, final_linearization, gradient, initial
            )
        if parameter_values is None:
            raise ValueError(
                "parameter_values are required when final_linearization is omitted."
            )
        parameters = jax.device_put(
            jnp.asarray(parameter_values, dtype=final_state.x.dtype), target_device
        )
        if parameters.ndim != 2 or parameters.shape[0] != self.batch_size:
            raise ValueError(
                "parameter_values must have shape "
                f"({self.batch_size}, parameter_count)."
            )
        mu_values = jax.device_put(
            jnp.asarray(mu, dtype=final_state.x.dtype), target_device
        )
        if mu_values.ndim == 0:
            mu_values = jnp.broadcast_to(mu_values, (self.batch_size,))
        return self._solve_yhat_adjoint_recompute_function(
            final_state, parameters, mu_values, gradient, initial
        )

    def solve_loss_yhat_gradient(
        self,
        *,
        initial_state: PrimalDualState,
        parameter_values: jax.Array,
        mu_values: jax.Array,
        q_data: jax.Array,
        valid_mask: jax.Array,
        q_mode: str,
        initial_adjoint: jax.Array | None = None,
    ) -> SparseProjectionLossGradientResult:
        """Run forward projection, quadratic loss, and sparse adjoint together."""

        if q_mode not in {"identity", "diagonal", "dense"}:
            raise ValueError("q_mode must be identity, diagonal, or dense.")
        width = initial_state.x.shape[1] + initial_state.y_c.shape[1] + initial_state.y_d.shape[1]
        if initial_adjoint is None:
            initial_adjoint = jnp.zeros(
                (self.batch_size, width), dtype=initial_state.x.dtype
            )
        function = (
            self._solve_loss_gradient_dense_function
            if q_mode == "dense"
            else self._solve_loss_gradient_diagonal_function
        )
        return function(
            initial_state,
            parameter_values,
            mu_values,
            q_data,
            valid_mask,
            initial_adjoint,
        )



def prepare_batched_projection_condensed_sparse_solver(
    **kwargs,
) -> PreparedSparseCondensedBatchSolver:
    """Create a persistent fixed-shape sparse solver."""

    return PreparedSparseCondensedBatchSolver(**kwargs)


def run_batched_projection_condensed_sparse_iterations(
    initial_state: PrimalDualState,
    parameter_values: jax.Array,
    equality_function: VectorFunction,
    inequality_function: VectorFunction,
    equality_local_derivatives: LocalDerivativeFunction,
    inequality_local_derivatives: LocalDerivativeFunction,
    equality_support: jax.Array,
    equality_mask: jax.Array,
    inequality_support: jax.Array,
    inequality_mask: jax.Array,
    target_parameter_indices: jax.Array,
    indices: BoundIndices,
    bounds: BoundValues,
    mu,
    *,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-9,
    tau: float = 0.995,
    step_scale: float = 1.0,
    minimum_step: float = 1e-14,
    regularization: NewtonRegularization | None = None,
    gmres_tolerance: float = 1e-9,
    gmres_atol: float = 0.0,
    gmres_restart: int = 30,
    gmres_maxiter: int = 50,
    use_preconditioner: bool = True,
    backtracking: bool = False,
    max_backtracking_steps: int = 8,
    backtracking_factor: float = 0.5,
    constraints_affine: bool = False,
    gmres_unroll: int = 1,
    store_residual_history: bool = True,
    jit: bool = True,
) -> BatchedFixedMuResult:
    """Backward-compatible one-shot sparse solve.

    Repeated workloads should reuse
    :class:`PreparedSparseCondensedBatchSolver`; the high-level
    :class:`src.problem.Problem` API does this automatically.
    """

    del jit  # The sparse kernels are always individually JIT compiled.
    if _is_jax_array(parameter_values):
        parameters = jnp.asarray(parameter_values, dtype=initial_state.x.dtype)
    else:
        parameter_values_host = np.asarray(parameter_values, dtype=np.float64)
        if not np.all(np.isfinite(parameter_values_host)):
            raise ValueError("parameter_values must be finite.")
        parameters = jnp.asarray(parameter_values_host, dtype=initial_state.x.dtype)
    if parameters.ndim != 2:
        raise ValueError("parameter_values must be a rank-two array.")
    batch_size = int(parameters.shape[0])
    solver = PreparedSparseCondensedBatchSolver(
        batch_size=batch_size,
        equality_function=equality_function,
        inequality_function=inequality_function,
        equality_local_derivatives=equality_local_derivatives,
        inequality_local_derivatives=inequality_local_derivatives,
        equality_support=equality_support,
        equality_mask=equality_mask,
        inequality_support=inequality_support,
        inequality_mask=inequality_mask,
        target_parameter_indices=target_parameter_indices,
        indices=indices,
        bounds=bounds,
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance,
        tau=tau,
        step_scale=step_scale,
        minimum_step=minimum_step,
        regularization=regularization,
        gmres_tolerance=gmres_tolerance,
        gmres_atol=gmres_atol,
        gmres_restart=gmres_restart,
        gmres_maxiter=gmres_maxiter,
        use_preconditioner=use_preconditioner,
        backtracking=backtracking,
        max_backtracking_steps=max_backtracking_steps,
        backtracking_factor=backtracking_factor,
        constraints_affine=constraints_affine,
        gmres_unroll=gmres_unroll,
        store_residual_history=store_residual_history,
    )
    return solver.solve(
        initial_state=initial_state,
        parameter_values=parameters,
        mu=mu,
    )

# ---------------------------------------------------------------------------
# Compact batched Krylov kernels
# ---------------------------------------------------------------------------


def _jacobian_row_count(values: jax.Array) -> int:
    return int(values.shape[-2])


def _batched_local_jacobian_matvec(
    values,
    support,
    mask,
    vector,
    *,
    full_support=False,
):
    """Apply row-sparse Jacobians with batched or shared coefficients."""

    if _jacobian_row_count(values) == 0:
        return jnp.empty((vector.shape[0], 0), dtype=vector.dtype)
    gathered = vector[:, _safe_support(support)]
    coefficients = values if values.ndim == 3 else values[None, :, :]
    if full_support:
        return jnp.sum(coefficients * gathered, axis=-1)
    return jnp.sum(
        jnp.where(mask[None, :, :], coefficients * gathered, 0.0), axis=-1
    )


def _batched_local_jacobian_transpose_matvec(
    values,
    support,
    mask,
    vector,
    n_variables,
    *,
    full_support=False,
):
    """Scatter transpose product with batched or shared coefficients."""

    if _jacobian_row_count(values) == 0:
        return jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    coefficients = values if values.ndim == 3 else values[None, :, :]
    contributions = coefficients * vector[:, :, None]
    if not full_support:
        contributions = jnp.where(mask[None, :, :], contributions, 0.0)
    result = jnp.zeros((vector.shape[0], n_variables), dtype=values.dtype)
    return result.at[:, _safe_support(support)].add(contributions)


def _batched_local_jacobian_transpose_gather_matvec(
    values,
    column_rows,
    column_slots,
    column_mask,
    vector,
):
    """Global padded gather/reduce transpose product."""

    n_variables = int(column_rows.shape[0])
    if _jacobian_row_count(values) == 0 or column_rows.shape[1] == 0:
        return jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    if values.ndim == 3:
        gathered_values = values[:, column_rows, column_slots]
    else:
        gathered_values = values[column_rows, column_slots][None, :, :]
    gathered_vector = vector[:, column_rows]
    return jnp.sum(
        jnp.where(
            column_mask[None, :, :],
            gathered_values * gathered_vector,
            0.0,
        ),
        axis=-1,
    )


def _batched_local_jacobian_transpose_bucketed_matvec(
    values,
    column_buckets,
    vector,
    n_variables,
):
    """Degree-bucketed gather/reduce transpose product for GPUs."""

    if _jacobian_row_count(values) == 0:
        return jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    result = jnp.zeros((vector.shape[0], n_variables), dtype=vector.dtype)
    for variable_indices, rows, slots, valid in column_buckets:
        if values.ndim == 3:
            gathered_values = values[:, rows, slots]
        else:
            gathered_values = values[rows, slots][None, :, :]
        gathered_vector = vector[:, rows]
        contributions = jnp.sum(
            jnp.where(
                valid[None, :, :],
                gathered_values * gathered_vector,
                0.0,
            ),
            axis=-1,
        )
        result = result.at[:, variable_indices].set(contributions)
    return result


def _batched_local_constraint_hessian_matvec(
    equality_hessian,
    inequality_hessian,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    equality_multipliers,
    inequality_multipliers,
    vector,
):
    result = jnp.zeros_like(vector)
    if equality_hessian.shape[1]:
        local_vector = vector[:, _safe_support(equality_support)]
        local_result = jnp.einsum("bmij,bmj->bmi", equality_hessian, local_vector)
        local_result = jnp.where(
            equality_mask[None, :, :],
            local_result * equality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(equality_support)].add(local_result)
    if inequality_hessian.shape[1]:
        local_vector = vector[:, _safe_support(inequality_support)]
        local_result = jnp.einsum("bmij,bmj->bmi", inequality_hessian, local_vector)
        local_result = jnp.where(
            inequality_mask[None, :, :],
            local_result * inequality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(inequality_support)].add(local_result)
    return result


def _batched_local_constraint_hessian_diagonal(
    equality_hessian,
    inequality_hessian,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    equality_multipliers,
    inequality_multipliers,
    n_variables,
):
    result = jnp.zeros(
        (equality_multipliers.shape[0], n_variables),
        dtype=equality_hessian.dtype,
    )
    if equality_hessian.shape[1]:
        diagonal = jnp.diagonal(equality_hessian, axis1=-2, axis2=-1)
        contribution = jnp.where(
            equality_mask[None, :, :],
            diagonal * equality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(equality_support)].add(contribution)
    if inequality_hessian.shape[1]:
        diagonal = jnp.diagonal(inequality_hessian, axis1=-2, axis2=-1)
        contribution = jnp.where(
            inequality_mask[None, :, :],
            diagonal * inequality_multipliers[:, :, None],
            0.0,
        )
        result = result.at[:, _safe_support(inequality_support)].add(contribution)
    return result


def _batched_schur_diagonal(
    jacobian,
    support,
    mask,
    inverse_primal,
    base,
    *,
    full_support=False,
    jacobian_squared=None,
):
    if _jacobian_row_count(jacobian) == 0:
        return jnp.empty((inverse_primal.shape[0], 0), dtype=inverse_primal.dtype)
    local_inverse = inverse_primal[:, _safe_support(support)]
    if jacobian_squared is None:
        squared = jacobian * jacobian
    else:
        squared = jacobian_squared
    squared = squared if squared.ndim == 3 else squared[None, :, :]
    contributions = squared * local_inverse
    if not full_support:
        contributions = jnp.where(mask[None, :, :], contributions, 0.0)
    correction = jnp.sum(contributions, axis=-1)
    return base + correction


def _solve_condensed_direction_batch(
    state: PrimalDualState,
    numeric: SparseNumericData,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    indices,
    regularization,
    *,
    tolerance,
    absolute_tolerance,
    max_iterations,
    restart,
    use_preconditioner,
    constraints_affine=False,
    equality_full_support=False,
    inequality_full_support=False,
    combined_full_support=False,
    combined_support=None,
    combined_mask=None,
    column_rows=None,
    column_slots=None,
    column_mask=None,
    column_buckets=None,
    constant_combined_jacobian=None,
    constant_combined_jacobian_squared=None,
    affine_kkt_matvec=None,
    gmres_unroll=1,
    arnoldi_prefix_block=0,
):
    batch_size, n_variables = state.x.shape
    n_equalities = state.y_c.shape[1]

    # Build the reduced blocks directly instead of materializing the generic
    # full reduced-KKT RHS and slicing it again.  The scatter corrections are
    # exactly the same algebra as ``build_reduced_kkt_rhs``.
    rhs_x = -numeric.residual.x
    rhs_s = -numeric.residual.s
    rhs_x = rhs_x.at[:, indices.x_lower].add(
        -numeric.residual.z_L / numeric.slacks.x_lower
    )
    rhs_x = rhs_x.at[:, indices.x_upper].add(
        numeric.residual.z_U / numeric.slacks.x_upper
    )
    rhs_s = rhs_s.at[:, indices.s_lower].add(
        -numeric.residual.v_L / numeric.slacks.s_lower
    )
    rhs_s = rhs_s.at[:, indices.s_upper].add(
        numeric.residual.v_U / numeric.slacks.s_upper
    )
    rhs_c = -numeric.residual.y_c
    rhs_d = -numeric.residual.y_d

    slack_diagonal = numeric.sigma_s + regularization.delta_s
    inverse_slack = 1.0 / slack_diagonal
    rhs = jnp.concatenate(
        (rhs_x, rhs_c, rhs_d + inverse_slack * rhs_s), axis=1
    )

    def split(vector):
        c_end_local = n_variables + n_equalities
        return (
            vector[:, :n_variables],
            vector[:, n_variables:c_end_local],
            vector[:, c_end_local:],
        )

    if constant_combined_jacobian is not None:
        combined_jacobian = constant_combined_jacobian
    elif n_equalities == 0:
        combined_jacobian = numeric.inequality_jacobian
        combined_full_support = inequality_full_support
    elif state.y_d.shape[1] == 0:
        combined_jacobian = numeric.equality_jacobian
        combined_full_support = equality_full_support
    else:
        equality_jacobian = numeric.equality_jacobian
        inequality_jacobian = numeric.inequality_jacobian
        if equality_jacobian.ndim != inequality_jacobian.ndim:
            if equality_jacobian.ndim == 2:
                equality_jacobian = jnp.broadcast_to(
                    equality_jacobian,
                    (batch_size,) + equality_jacobian.shape,
                )
            else:
                inequality_jacobian = jnp.broadcast_to(
                    inequality_jacobian,
                    (batch_size,) + inequality_jacobian.shape,
                )
        axis = 1 if equality_jacobian.ndim == 3 else 0
        combined_jacobian = jnp.concatenate(
            (equality_jacobian, inequality_jacobian), axis=axis
        )

    use_column_gather = (
        column_rows is not None
        and column_slots is not None
        and column_mask is not None
    )
    use_column_buckets = column_buckets is not None

    primal_kkt_diagonal = (
        1.0 + numeric.sigma_x + regularization.delta_x
    )
    equality_dual_diagonal = jnp.full(
        (batch_size, n_equalities),
        regularization.delta_c,
        dtype=state.x.dtype,
    )
    inequality_dual_diagonal = inverse_slack + regularization.delta_d
    dual_kkt_diagonal = jnp.concatenate(
        (equality_dual_diagonal, inequality_dual_diagonal), axis=1
    )

    def matvec(vector):
        with jax.named_scope("pyopt_kkt_matvec"):
            if affine_kkt_matvec is not None:
                return affine_kkt_matvec(
                    vector, primal_kkt_diagonal, dual_kkt_diagonal
                )

            vector_x, vector_c, vector_d = split(vector)
            vector_dual = vector[:, n_variables:]
            if constraints_affine:
                constraint_hvp = jnp.zeros_like(vector_x)
            else:
                constraint_hvp = _batched_local_constraint_hessian_matvec(
                    numeric.equality_hessian,
                    numeric.inequality_hessian,
                    equality_support,
                    equality_mask,
                    inequality_support,
                    inequality_mask,
                    state.y_c,
                    state.y_d,
                    vector_x,
                )
            if use_column_buckets:
                transpose_product = (
                    _batched_local_jacobian_transpose_bucketed_matvec(
                        combined_jacobian,
                        column_buckets,
                        vector_dual,
                        n_variables,
                    )
                )
            elif use_column_gather:
                transpose_product = (
                    _batched_local_jacobian_transpose_gather_matvec(
                        combined_jacobian,
                        column_rows,
                        column_slots,
                        column_mask,
                        vector_dual,
                    )
                )
            else:
                transpose_product = _batched_local_jacobian_transpose_matvec(
                    combined_jacobian,
                    combined_support,
                    combined_mask,
                    vector_dual,
                    n_variables,
                    full_support=combined_full_support,
                )
            row_x = (
                primal_kkt_diagonal * vector_x
                + constraint_hvp
                + transpose_product
            )
            constraint_rows = _batched_local_jacobian_matvec(
                combined_jacobian,
                combined_support,
                combined_mask,
                vector_x,
                full_support=combined_full_support,
            )
            row_c = (
                constraint_rows[:, :n_equalities]
                - regularization.delta_c * vector_c
            )
            row_d = constraint_rows[:, n_equalities:] - (
                inverse_slack + regularization.delta_d
            ) * vector_d
            return jnp.concatenate((row_x, row_c, row_d), axis=1)

    if use_preconditioner:
        if constraints_affine:
            hessian_diagonal = jnp.zeros_like(state.x)
        else:
            hessian_diagonal = _batched_local_constraint_hessian_diagonal(
                numeric.equality_hessian,
                numeric.inequality_hessian,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                state.y_c,
                state.y_d,
                n_variables,
            )
        primal_diagonal = jnp.maximum(
            jnp.abs(
                1.0
                + numeric.sigma_x
                + regularization.delta_x
                + hessian_diagonal
            ),
            1e-8,
        )
        inverse_primal = 1.0 / primal_diagonal
        equality_base = jnp.full(
            (batch_size, n_equalities),
            max(abs(float(regularization.delta_c)), 1e-8),
            dtype=state.x.dtype,
        )
        inequality_base = jnp.abs(inverse_slack + regularization.delta_d)
        combined_schur = jnp.maximum(
            _batched_schur_diagonal(
                combined_jacobian,
                combined_support,
                combined_mask,
                inverse_primal,
                jnp.concatenate((equality_base, inequality_base), axis=1),
                full_support=combined_full_support,
                jacobian_squared=constant_combined_jacobian_squared,
            ),
            1e-8,
        )
        equality_schur = combined_schur[:, :n_equalities]
        inequality_schur = combined_schur[:, n_equalities:]
        diagonal = jnp.concatenate(
            (primal_diagonal, equality_schur, inequality_schur), axis=1
        )
        preconditioner = lambda value: value / diagonal
    else:
        preconditioner = lambda value: value

    nonlinear_residual_inf = jax.vmap(_residual_inf_norm)(numeric.residual)
    forcing = jnp.minimum(
        0.1, jnp.sqrt(jnp.maximum(nonlinear_residual_inf, 0.0))
    )
    accepted_linear_residual = jnp.maximum(
        1e-10,
        forcing * jnp.maximum(nonlinear_residual_inf, 1e-16),
    )

    solution, info, linear_residual = _batched_restarted_gmres(
        matvec,
        rhs,
        preconditioner,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        accepted_residual_threshold=accepted_linear_residual,
        restart=min(restart, max_iterations),
        max_iterations=max_iterations,
        unroll=gmres_unroll,
        arnoldi_prefix_block=arnoldi_prefix_block,
    )
    delta_x, delta_y_c, delta_y_d = split(solution)
    delta_s = inverse_slack * (rhs_s + delta_y_d)
    recovered = jax.vmap(
        recover_bound_multiplier_directions,
        in_axes=(0, None, 0, 0, 0, 0),
    )(
        state,
        indices,
        numeric.slacks,
        numeric.residual,
        delta_x,
        delta_s,
    )
    delta_z_L, delta_z_U, delta_v_L, delta_v_U = recovered
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
    regular = jnp.all(jnp.isfinite(inverse_slack), axis=1)
    return direction, info, jnp.where(regular, linear_residual, jnp.inf)


def _combine_projection_jacobians(
    state: PrimalDualState,
    numeric: SparseNumericData,
    *,
    constant_combined_jacobian=None,
):
    """Return one equality/inequality Jacobian block with a stable batch layout."""

    batch_size = state.x.shape[0]
    n_equalities = state.y_c.shape[1]
    n_inequalities = state.y_d.shape[1]
    if constant_combined_jacobian is not None:
        return constant_combined_jacobian
    if n_equalities == 0:
        return numeric.inequality_jacobian
    if n_inequalities == 0:
        return numeric.equality_jacobian
    equality_jacobian = numeric.equality_jacobian
    inequality_jacobian = numeric.inequality_jacobian
    if equality_jacobian.ndim != inequality_jacobian.ndim:
        if equality_jacobian.ndim == 2:
            equality_jacobian = jnp.broadcast_to(
                equality_jacobian, (batch_size,) + equality_jacobian.shape
            )
        else:
            inequality_jacobian = jnp.broadcast_to(
                inequality_jacobian, (batch_size,) + inequality_jacobian.shape
            )
    axis = 1 if equality_jacobian.ndim == 3 else 0
    return jnp.concatenate((equality_jacobian, inequality_jacobian), axis=axis)


def _build_sparse_projection_linearization(
    state: PrimalDualState,
    numeric: SparseNumericData,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    regularization,
    *,
    use_preconditioner,
    constraints_affine,
    equality_full_support,
    inequality_full_support,
    combined_full_support,
    combined_support,
    combined_mask,
    constant_combined_jacobian=None,
    constant_combined_jacobian_squared=None,
) -> SparseProjectionLinearization:
    """Build and cache all final-KKT data needed by repeated adjoint solves."""

    batch_size, n_variables = state.x.shape
    n_equalities = state.y_c.shape[1]
    combined_jacobian = _combine_projection_jacobians(
        state,
        numeric,
        constant_combined_jacobian=constant_combined_jacobian,
    )
    inverse_slack = 1.0 / (numeric.sigma_s + regularization.delta_s)
    primal_kkt_diagonal = 1.0 + numeric.sigma_x + regularization.delta_x
    equality_dual_diagonal = jnp.full(
        (batch_size, n_equalities),
        regularization.delta_c,
        dtype=state.x.dtype,
    )
    dual_kkt_diagonal = jnp.concatenate(
        (
            equality_dual_diagonal,
            inverse_slack + regularization.delta_d,
        ),
        axis=1,
    )

    if use_preconditioner:
        if constraints_affine:
            hessian_diagonal = jnp.zeros_like(state.x)
        else:
            hessian_diagonal = _batched_local_constraint_hessian_diagonal(
                numeric.equality_hessian,
                numeric.inequality_hessian,
                equality_support,
                equality_mask,
                inequality_support,
                inequality_mask,
                state.y_c,
                state.y_d,
                n_variables,
            )
        primal_diagonal = jnp.maximum(
            jnp.abs(primal_kkt_diagonal + hessian_diagonal), 1e-8
        )
        inverse_primal = 1.0 / primal_diagonal
        equality_base = jnp.full(
            (batch_size, n_equalities),
            max(abs(float(regularization.delta_c)), 1e-8),
            dtype=state.x.dtype,
        )
        inequality_base = jnp.abs(inverse_slack + regularization.delta_d)
        combined_schur = jnp.maximum(
            _batched_schur_diagonal(
                combined_jacobian,
                combined_support,
                combined_mask,
                inverse_primal,
                jnp.concatenate((equality_base, inequality_base), axis=1),
                full_support=combined_full_support,
                jacobian_squared=constant_combined_jacobian_squared,
            ),
            1e-8,
        )
        preconditioner_diagonal = jnp.concatenate(
            (primal_diagonal, combined_schur), axis=1
        )
    else:
        preconditioner_diagonal = jnp.empty(
            (batch_size, 0), dtype=state.x.dtype
        )

    return SparseProjectionLinearization(
        combined_jacobian=combined_jacobian,
        equality_hessian=numeric.equality_hessian,
        inequality_hessian=numeric.inequality_hessian,
        inverse_slack=inverse_slack,
        primal_kkt_diagonal=primal_kkt_diagonal,
        dual_kkt_diagonal=dual_kkt_diagonal,
        preconditioner_diagonal=preconditioner_diagonal,
    )


def _solve_condensed_yhat_adjoint_batch(
    state: PrimalDualState,
    linearization: SparseProjectionLinearization,
    grad_x: jax.Array,
    equality_support,
    equality_mask,
    inequality_support,
    inequality_mask,
    regularization,
    *,
    tolerance,
    absolute_tolerance,
    max_iterations,
    restart,
    use_preconditioner,
    constraints_affine=False,
    equality_full_support=False,
    inequality_full_support=False,
    combined_full_support=False,
    combined_support=None,
    combined_mask=None,
    column_rows=None,
    column_slots=None,
    column_mask=None,
    column_buckets=None,
    affine_kkt_matvec=None,
    gmres_unroll=1,
    arnoldi_prefix_block=0,
    solver_mode="gmres",
    initial_solution=None,
) -> SparseProjectionAdjointResult:
    """Solve the final condensed adjoint using saved forward linearization data."""

    batch_size, n_variables = state.x.shape
    n_equalities = state.y_c.shape[1]
    n_inequalities = state.y_d.shape[1]
    rhs = jnp.pad(grad_x, ((0, 0), (0, n_equalities + n_inequalities)))

    def split(vector):
        c_end_local = n_variables + n_equalities
        return (
            vector[:, :n_variables],
            vector[:, n_variables:c_end_local],
            vector[:, c_end_local:],
        )

    combined_jacobian = linearization.combined_jacobian
    inverse_slack = linearization.inverse_slack
    primal_kkt_diagonal = linearization.primal_kkt_diagonal
    dual_kkt_diagonal = linearization.dual_kkt_diagonal
    use_column_gather = (
        column_rows is not None
        and column_slots is not None
        and column_mask is not None
    )
    use_column_buckets = column_buckets is not None

    def matvec(vector):
        with jax.named_scope("pyopt_adjoint_kkt_matvec"):
            if affine_kkt_matvec is not None:
                return affine_kkt_matvec(
                    vector, primal_kkt_diagonal, dual_kkt_diagonal
                )
            vector_x, vector_c, vector_d = split(vector)
            vector_dual = vector[:, n_variables:]
            if constraints_affine:
                constraint_hvp = jnp.zeros_like(vector_x)
            else:
                constraint_hvp = _batched_local_constraint_hessian_matvec(
                    linearization.equality_hessian,
                    linearization.inequality_hessian,
                    equality_support,
                    equality_mask,
                    inequality_support,
                    inequality_mask,
                    state.y_c,
                    state.y_d,
                    vector_x,
                )
            if use_column_buckets:
                transpose_product = _batched_local_jacobian_transpose_bucketed_matvec(
                    combined_jacobian,
                    column_buckets,
                    vector_dual,
                    n_variables,
                )
            elif use_column_gather:
                transpose_product = _batched_local_jacobian_transpose_gather_matvec(
                    combined_jacobian,
                    column_rows,
                    column_slots,
                    column_mask,
                    vector_dual,
                )
            else:
                transpose_product = _batched_local_jacobian_transpose_matvec(
                    combined_jacobian,
                    combined_support,
                    combined_mask,
                    vector_dual,
                    n_variables,
                    full_support=combined_full_support,
                )
            row_x = primal_kkt_diagonal * vector_x + constraint_hvp + transpose_product
            constraint_rows = _batched_local_jacobian_matvec(
                combined_jacobian,
                combined_support,
                combined_mask,
                vector_x,
                full_support=combined_full_support,
            )
            row_c = constraint_rows[:, :n_equalities] - regularization.delta_c * vector_c
            row_d = constraint_rows[:, n_equalities:] - (
                inverse_slack + regularization.delta_d
            ) * vector_d
            return jnp.concatenate((row_x, row_c, row_d), axis=1)

    if use_preconditioner:
        diagonal = linearization.preconditioner_diagonal
        preconditioner = lambda value: value / diagonal
    else:
        preconditioner = lambda value: value

    if initial_solution is None:
        initial_solution = jnp.zeros_like(rhs)

    if solver_mode == "dense":
        width = rhs.shape[1]
        identity = jnp.eye(width, dtype=rhs.dtype)
        columns = jax.vmap(
            lambda column: matvec(jnp.broadcast_to(column, rhs.shape)),
            out_axes=2,
        )(identity)
        solution = jax.vmap(jnp.linalg.solve)(columns, rhs)
        residual = matvec(solution) - rhs
        linear_residual = jnp.max(jnp.abs(residual), axis=1, initial=0.0)
        rhs_scale = jnp.max(jnp.abs(rhs), axis=1, initial=0.0)
        threshold = jnp.maximum(absolute_tolerance, tolerance * rhs_scale)
        finite = jnp.all(jnp.isfinite(solution), axis=1) & jnp.isfinite(linear_residual)
        info = jnp.where(~finite, 2, jnp.where(linear_residual <= threshold, 0, 1)).astype(jnp.int32)
        linear_iterations = jnp.ones((batch_size,), dtype=jnp.int32)
    else:
        solution, info, linear_residual, linear_iterations = _batched_restarted_gmres(
            matvec,
            rhs,
            preconditioner,
            tolerance=tolerance,
            absolute_tolerance=absolute_tolerance,
            accepted_residual_threshold=None,
            restart=min(restart, max_iterations),
            max_iterations=max_iterations,
            unroll=gmres_unroll,
            arnoldi_prefix_block=arnoldi_prefix_block,
            initial_solution=initial_solution,
            return_iterations=True,
        )

    lambda_x, lambda_c, lambda_d = split(solution)
    regular = jnp.all(jnp.isfinite(inverse_slack), axis=1)
    linear_residual = jnp.where(regular, linear_residual, jnp.inf)
    info = jnp.where(regular, info, jnp.asarray(2, dtype=info.dtype))
    return SparseProjectionAdjointResult(
        x=lambda_x,
        y_c=lambda_c,
        y_d=lambda_d,
        linear_status=info,
        linear_residual_inf_norm=linear_residual,
        linear_iterations=linear_iterations,
    )


def _batched_restarted_gmres(
    matvec,
    rhs,
    right_preconditioner,
    *,
    tolerance: float,
    absolute_tolerance: float,
    accepted_residual_threshold: jax.Array | None = None,
    restart: int,
    max_iterations: int,
    unroll: int = 1,
    arnoldi_prefix_block: int = 0,
    initial_solution: jax.Array | None = None,
    return_iterations: bool = False,
):
    """Restarted batched GMRES with incremental Givens rotations.

    The Krylov basis remains fixed-width for JAX compilation, but Arnoldi work
    stops as soon as every active row satisfies the linear residual estimate.
    The final restart block is also capped by the exact remaining iteration
    budget, so ``max_iterations`` is never exceeded.
    """

    batch_size, width = rhs.shape
    restart = int(min(restart, max_iterations, width))
    dtype = rhs.dtype
    epsilon = jnp.asarray(1e-30, dtype=dtype)
    breakdown_epsilon = jnp.asarray(1e-14, dtype=dtype)
    rhs_scale = jnp.max(jnp.abs(rhs), axis=1, initial=0.0)
    threshold = jnp.maximum(absolute_tolerance, tolerance * rhs_scale)
    if accepted_residual_threshold is not None:
        inexact_cap = jnp.maximum(1e-10, 1e-8 * rhs_scale)
        threshold = jnp.maximum(
            threshold,
            jnp.minimum(accepted_residual_threshold, inexact_cap),
        )

    prefix_block = int(arnoldi_prefix_block)
    if prefix_block <= 0 or prefix_block >= restart:
        prefix_sizes = (restart,)
    else:
        prefix_sizes = tuple(range(prefix_block, restart, prefix_block)) + (restart,)

    def orthogonalize_prefix(basis_, image, iteration):
        """Two-pass MGS over a small static prefix bucket.

        Future Krylov vectors are zero, so rounding the active prefix upward to
        a bucket boundary is algebraically identical while avoiding the full
        restart-width dot products used by the generic GPU graph.
        """

        def make_branch(prefix_size):
            def branch(operand):
                basis_value, image_value = operand
                active_basis = basis_value[:, :prefix_size, :]
                coefficients_prefix = jnp.einsum(
                    "bid,bd->bi", active_basis, image_value
                )
                image_value = image_value - jnp.einsum(
                    "bi,bid->bd", coefficients_prefix, active_basis
                )
                correction_prefix = jnp.einsum(
                    "bid,bd->bi", active_basis, image_value
                )
                coefficients_prefix = coefficients_prefix + correction_prefix
                image_value = image_value - jnp.einsum(
                    "bi,bid->bd", correction_prefix, active_basis
                )
                coefficients = jnp.pad(
                    coefficients_prefix,
                    ((0, 0), (0, restart - prefix_size)),
                )
                return coefficients, image_value

            return branch

        if len(prefix_sizes) == 1:
            return make_branch(prefix_sizes[0])((basis_, image))
        branch_index = jnp.minimum(
            iteration // prefix_block, len(prefix_sizes) - 1
        )
        return jax.lax.switch(
            branch_index,
            tuple(make_branch(size) for size in prefix_sizes),
            (basis_, image),
        )

    def apply_rotation_prefix(
        column, cosines, sines, iteration
    ):
        def apply_previous_rotation(rotation_index, rotated_column):
            first = rotated_column[:, rotation_index]
            second = rotated_column[:, rotation_index + 1]
            cosine = cosines[:, rotation_index]
            sine = sines[:, rotation_index]
            rotated_first = cosine * first + sine * second
            rotated_second = -sine * first + cosine * second
            apply = rotation_index < iteration
            rotated_column = rotated_column.at[:, rotation_index].set(
                jnp.where(apply, rotated_first, first)
            )
            rotated_column = rotated_column.at[:, rotation_index + 1].set(
                jnp.where(apply, rotated_second, second)
            )
            return rotated_column

        def make_branch(prefix_size):
            return lambda value: jax.lax.fori_loop(
                0,
                prefix_size,
                apply_previous_rotation,
                value,
                unroll=unroll,
            )

        if len(prefix_sizes) == 1:
            return make_branch(prefix_sizes[0])(column)
        branch_index = jnp.minimum(
            iteration // prefix_block, len(prefix_sizes) - 1
        )
        return jax.lax.switch(
            branch_index,
            tuple(make_branch(size) for size in prefix_sizes),
            column,
        )

    def back_substitute(r_factor, projected_rhs, row_steps):
        coefficients = jnp.zeros((batch_size, restart), dtype=dtype)

        def body(reverse_index, values):
            row = restart - 1 - reverse_index
            trailing = jnp.sum(r_factor[:, row, :] * values, axis=1)
            numerator = projected_rhs[:, row] - trailing
            diagonal = r_factor[:, row, row]
            safe_diagonal = jnp.where(
                jnp.abs(diagonal) > epsilon,
                diagonal,
                jnp.ones_like(diagonal),
            )
            value = numerator / safe_diagonal
            value = jnp.where(row < row_steps, value, 0.0)
            return values.at[:, row].set(value)

        return jax.lax.fori_loop(
            0, restart, body, coefficients, unroll=unroll
        )

    def cycle_body(carry):
        completed_iterations, x, residual_inf, total_row_iterations = carry
        remaining_iterations = max_iterations - completed_iterations
        cycle_limit = jnp.minimum(
            jnp.asarray(restart, dtype=jnp.int32), remaining_iterations
        )

        residual = rhs - matvec(x)
        beta = jnp.linalg.norm(residual, axis=1)
        safe_beta = jnp.maximum(beta, epsilon)
        cycle_active = residual_inf > threshold

        basis = jnp.zeros((batch_size, restart + 1, width), dtype=dtype)
        basis = basis.at[:, 0, :].set(residual / safe_beta[:, None])
        preconditioned_basis = jnp.zeros(
            (batch_size, restart, width), dtype=dtype
        )
        hessenberg = jnp.zeros(
            (batch_size, restart + 1, restart), dtype=dtype
        )
        cosines = jnp.zeros((batch_size, restart), dtype=dtype)
        sines = jnp.zeros((batch_size, restart), dtype=dtype)
        projected_rhs = jnp.zeros(
            (batch_size, restart + 1), dtype=dtype
        ).at[:, 0].set(beta)
        row_steps = jnp.zeros((batch_size,), dtype=jnp.int32)
        arnoldi_iteration = jnp.asarray(0, dtype=jnp.int32)

        def arnoldi_condition(inner_carry):
            (
                iteration,
                _basis,
                _preconditioned,
                _hessenberg,
                _cosines,
                _sines,
                _projected_rhs,
                active,
                _row_steps,
            ) = inner_carry
            return (iteration < cycle_limit) & jnp.any(active)

        def arnoldi_body(inner_carry):
            (
                iteration,
                basis_,
                preconditioned_,
                hessenberg_,
                cosines_,
                sines_,
                projected_rhs_,
                active,
                row_steps_,
            ) = inner_carry

            vector = basis_[:, iteration, :]
            preconditioned_vector = right_preconditioner(vector)
            old_preconditioned = preconditioned_[:, iteration, :]
            preconditioned_ = preconditioned_.at[:, iteration, :].set(
                jnp.where(
                    active[:, None], preconditioned_vector, old_preconditioned
                )
            )

            image = matvec(preconditioned_vector)
            with jax.named_scope("pyopt_gmres_orthogonalize"):
                coefficients, image = orthogonalize_prefix(
                    basis_, image, iteration
                )

            next_norm = jnp.linalg.norm(image, axis=1)
            safe_norm = jnp.maximum(next_norm, epsilon)
            next_vector = image / safe_norm[:, None]
            old_next_vector = basis_[:, iteration + 1, :]
            basis_ = basis_.at[:, iteration + 1, :].set(
                jnp.where(active[:, None], next_vector, old_next_vector)
            )

            column = jnp.zeros((batch_size, restart + 1), dtype=dtype)
            column = column.at[:, :restart].set(coefficients)
            column = column.at[:, iteration + 1].set(next_norm)

            with jax.named_scope("pyopt_gmres_givens"):
                column = apply_rotation_prefix(
                    column, cosines_, sines_, iteration
                )

            diagonal = column[:, iteration]
            subdiagonal = column[:, iteration + 1]
            rotation_norm = jnp.hypot(diagonal, subdiagonal)
            safe_rotation_norm = jnp.maximum(rotation_norm, epsilon)
            new_cosine = diagonal / safe_rotation_norm
            new_sine = subdiagonal / safe_rotation_norm
            new_cosine = jnp.where(
                rotation_norm > epsilon, new_cosine, jnp.ones_like(new_cosine)
            )
            new_sine = jnp.where(
                rotation_norm > epsilon, new_sine, jnp.zeros_like(new_sine)
            )

            old_cosine = cosines_[:, iteration]
            old_sine = sines_[:, iteration]
            cosines_ = cosines_.at[:, iteration].set(
                jnp.where(active, new_cosine, old_cosine)
            )
            sines_ = sines_.at[:, iteration].set(
                jnp.where(active, new_sine, old_sine)
            )

            column = column.at[:, iteration].set(rotation_norm)
            column = column.at[:, iteration + 1].set(0.0)
            old_column = hessenberg_[:, :, iteration]
            hessenberg_ = hessenberg_.at[:, :, iteration].set(
                jnp.where(active[:, None], column, old_column)
            )

            projected_current = projected_rhs_[:, iteration]
            projected_next = projected_rhs_[:, iteration + 1]
            rotated_current = (
                new_cosine * projected_current + new_sine * projected_next
            )
            rotated_next = (
                -new_sine * projected_current + new_cosine * projected_next
            )
            projected_rhs_ = projected_rhs_.at[:, iteration].set(
                jnp.where(active, rotated_current, projected_current)
            )
            projected_rhs_ = projected_rhs_.at[:, iteration + 1].set(
                jnp.where(active, rotated_next, projected_next)
            )

            row_steps_ = jnp.where(active, iteration + 1, row_steps_)
            residual_estimate = jnp.abs(rotated_next)
            finite = (
                jnp.isfinite(residual_estimate)
                & jnp.isfinite(rotation_norm)
                & jnp.all(jnp.isfinite(next_vector), axis=1)
            )
            happy_breakdown = next_norm <= breakdown_epsilon
            active = (
                active
                & finite
                & ~happy_breakdown
                & (residual_estimate > threshold)
            )

            return (
                iteration + 1,
                basis_,
                preconditioned_,
                hessenberg_,
                cosines_,
                sines_,
                projected_rhs_,
                active,
                row_steps_,
            )

        (
            arnoldi_iterations,
            _basis,
            preconditioned_basis,
            hessenberg,
            _cosines,
            _sines,
            projected_rhs,
            _active,
            row_steps,
        ) = jax.lax.while_loop(
            arnoldi_condition,
            arnoldi_body,
            (
                arnoldi_iteration,
                basis,
                preconditioned_basis,
                hessenberg,
                cosines,
                sines,
                projected_rhs,
                cycle_active,
                row_steps,
            ),
        )

        r_factor = hessenberg[:, :restart, :]
        coefficients = back_substitute(
            r_factor, projected_rhs[:, :restart], row_steps
        )
        correction = jnp.einsum(
            "bi,bid->bd", coefficients, preconditioned_basis
        )
        candidate = x + correction
        candidate_residual = rhs - matvec(candidate)
        candidate_norm = jnp.max(
            jnp.abs(candidate_residual), axis=1, initial=0.0
        )
        x = jnp.where(cycle_active[:, None], candidate, x)
        residual_inf = jnp.where(cycle_active, candidate_norm, residual_inf)
        return (
            completed_iterations + arnoldi_iterations,
            x,
            residual_inf,
            total_row_iterations + row_steps,
        )

    def cycle_condition(carry):
        completed_iterations, _x, residual_inf, _row_iterations = carry
        return (completed_iterations < max_iterations) & jnp.any(
            residual_inf > threshold
        )

    if initial_solution is None:
        initial_solution = jnp.zeros_like(rhs)
    initial_residual = rhs - matvec(initial_solution)
    initial_residual_inf = jnp.max(
        jnp.abs(initial_residual), axis=1, initial=0.0
    )
    _, solution, residual_inf, row_iterations = jax.lax.while_loop(
        cycle_condition,
        cycle_body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            initial_solution,
            initial_residual_inf,
            jnp.zeros((batch_size,), dtype=jnp.int32),
        ),
    )
    finite = jnp.all(jnp.isfinite(solution), axis=1) & jnp.isfinite(residual_inf)
    info = jnp.where(~finite, 2, jnp.where(residual_inf <= threshold, 0, 1)).astype(
        jnp.int32
    )
    if return_iterations:
        return solution, info, residual_inf, row_iterations
    return solution, info, residual_inf
