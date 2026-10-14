"""Optional GPU-specific kernels for the sparse condensed projection solver.

The public solver never requires these kernels.  CPU execution and unsupported
GPU configurations retain the standard JAX/XLA implementations.  The Pallas
kernel below specializes the affine, constant-Jacobian condensed KKT product;
it is intentionally isolated so GPU tuning cannot regress the CPU path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import jax
import jax.numpy as jnp

try:  # Pallas is optional across JAX versions/installations.
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - depends on the installed JAX build.
    pl = None


def pallas_available() -> bool:
    """Return whether the installed JAX exposes Pallas."""

    return pl is not None



def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""

    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _pad_second_axis(array, width: int, *, fill_value=0):
    """Pad a rank-two static sparse-layout array to ``width`` columns."""

    host = np.asarray(array)
    if host.ndim != 2:
        raise ValueError("Expected a rank-two sparse-layout array.")
    if host.shape[1] > width:
        raise ValueError("Requested padded width is smaller than the input width.")
    if host.shape[1] == width:
        return host
    pad = np.full(
        (host.shape[0], width - host.shape[1]),
        fill_value,
        dtype=host.dtype,
    )
    return np.concatenate((host, pad), axis=1)


def build_affine_constant_kkt_pallas_matvec(
    *,
    jacobian: jax.Array,
    row_support: jax.Array,
    row_mask: jax.Array,
    column_rows: jax.Array,
    column_slots: jax.Array,
    column_mask: jax.Array,
    n_variables: int,
    block_size: int = 64,
    interpret: bool = False,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Build a fused affine condensed-KKT product using Pallas.

    Parameters
    ----------
    jacobian:
        Constant combined equality/inequality local Jacobian with shape
        ``(n_constraints, row_width)``.
    row_support, row_mask:
        Row-oriented sparse layout used for ``J @ x``.
    column_rows, column_slots, column_mask:
        Column-oriented adjacency used for ``J.T @ y``.
    n_variables:
        Width of the primal block.
    block_size:
        Number of output entries produced by one Pallas program.
    interpret:
        Pallas interpreter mode, used by CPU unit tests only.  Production CUDA
        execution uses the Triton/Mosaic GPU lowering selected by JAX.

    Notes
    -----
    The mathematical product is exactly the affine condensed KKT operator::

        row_x = primal_diagonal * vector_x + J.T @ vector_dual
        row_d = J @ vector_x - dual_diagonal * vector_dual

    Nonlinear curvature and parameter-dependent Jacobians deliberately fall
    back to the standard JAX implementation.
    """

    if pl is None:
        raise RuntimeError("Pallas is not available in this JAX installation.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if jacobian.ndim != 2:
        raise ValueError("The Pallas affine KKT kernel requires a rank-two Jacobian.")

    n_constraints = int(jacobian.shape[0])
    row_width = int(jacobian.shape[1])
    max_column_degree = int(column_rows.shape[1])

    # Pallas' Triton lowering requires every vector/tile operation to have a
    # power-of-two element count.  The natural sparse widths are commonly not
    # powers of two (for example nnz=3 and maximum column degree=11).  Pad only
    # the static sparse-layout axes and mask the added entries so the numerical
    # operator is unchanged.
    pallas_block_size = _next_power_of_two(block_size)
    padded_row_width = _next_power_of_two(row_width)
    padded_column_degree = _next_power_of_two(max_column_degree)

    total_width = int(n_variables + n_constraints)
    padded_width = (
        (total_width + pallas_block_size - 1) // pallas_block_size
    ) * pallas_block_size

    jacobian_host = _pad_second_axis(
        jacobian, padded_row_width, fill_value=0.0
    )
    row_support_host = _pad_second_axis(
        row_support, padded_row_width, fill_value=0
    )
    row_mask_host = _pad_second_axis(
        row_mask, padded_row_width, fill_value=False
    )
    column_rows_host = _pad_second_axis(
        column_rows, padded_column_degree, fill_value=0
    )
    column_slots_host = _pad_second_axis(
        column_slots, padded_column_degree, fill_value=0
    )
    column_mask_host = _pad_second_axis(
        column_mask, padded_column_degree, fill_value=False
    )

    jacobian = jnp.asarray(jacobian_host)
    row_support = jnp.asarray(row_support_host, dtype=jnp.int32)
    row_mask = jnp.asarray(row_mask_host, dtype=bool)
    column_rows = jnp.asarray(column_rows_host, dtype=jnp.int32)
    column_slots = jnp.asarray(column_slots_host, dtype=jnp.int32)
    column_mask = jnp.asarray(column_mask_host, dtype=bool)

    def kernel(
        vector_ref,
        jacobian_ref,
        row_support_ref,
        row_mask_ref,
        column_rows_ref,
        column_slots_ref,
        column_mask_ref,
        primal_diagonal_ref,
        dual_diagonal_ref,
        output_ref,
    ):
        batch_index = pl.program_id(0)
        output_block = pl.program_id(1)
        lanes = jnp.arange(pallas_block_size, dtype=jnp.int32)
        columns = output_block * pallas_block_size + lanes
        valid = columns < total_width
        is_primal = columns < n_variables

        # Primal block: diagonal contribution plus J.T @ vector_dual.
        primal_columns = jnp.clip(columns, 0, max(n_variables - 1, 0))
        degree_slots = jnp.arange(padded_column_degree, dtype=jnp.int32)
        gathered_rows = column_rows_ref[
            primal_columns[:, None], degree_slots[None, :]
        ]
        gathered_slots = column_slots_ref[
            primal_columns[:, None], degree_slots[None, :]
        ]
        gathered_mask = column_mask_ref[
            primal_columns[:, None], degree_slots[None, :]
        ]
        primal_valid = valid[:, None] & is_primal[:, None] & gathered_mask
        transpose_values = (
            jacobian_ref[gathered_rows, gathered_slots] * primal_valid
        )
        transpose_duals = (
            vector_ref[batch_index, n_variables + gathered_rows] * primal_valid
        )
        transpose_product = jnp.sum(
            transpose_values * transpose_duals, axis=1
        )
        primal_vector = vector_ref[batch_index, primal_columns] * (valid & is_primal)
        primal_diagonal = primal_diagonal_ref[
            batch_index, primal_columns
        ] * (valid & is_primal)
        primal_output = primal_diagonal * primal_vector + transpose_product

        # Constraint block: J @ vector_x minus the equality/inequality diagonal.
        constraint_rows = jnp.clip(
            columns - n_variables, 0, max(n_constraints - 1, 0)
        )
        row_slots = jnp.arange(padded_row_width, dtype=jnp.int32)
        supports = row_support_ref[
            constraint_rows[:, None], row_slots[None, :]
        ]
        support_mask = row_mask_ref[
            constraint_rows[:, None], row_slots[None, :]
        ]
        dual_valid = valid[:, None] & (~is_primal[:, None]) & support_mask
        row_values = jacobian_ref[
            constraint_rows[:, None], row_slots[None, :]
        ] * dual_valid
        primal_values = vector_ref[batch_index, supports] * dual_valid
        forward_product = jnp.sum(row_values * primal_values, axis=1)
        dual_vector = vector_ref[
            batch_index, n_variables + constraint_rows
        ] * (valid & ~is_primal)
        dual_diagonal = dual_diagonal_ref[
            batch_index, constraint_rows
        ] * (valid & ~is_primal)
        dual_output = forward_product - dual_diagonal * dual_vector

        output_ref[batch_index, columns] = jnp.where(
            valid,
            jnp.where(is_primal, primal_output, dual_output),
            jnp.asarray(0.0, dtype=vector_ref.dtype),
        )

    def matvec(
        vector: jax.Array,
        primal_diagonal: jax.Array,
        dual_diagonal: jax.Array,
    ) -> jax.Array:
        if vector.ndim != 2:
            raise ValueError("vector must be rank two.")
        if int(vector.shape[1]) != total_width:
            raise ValueError(
                f"Expected vector width {total_width}, received {vector.shape[1]}."
            )
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (int(vector.shape[0]), padded_width), vector.dtype
            ),
            grid=(int(vector.shape[0]), padded_width // pallas_block_size),
            interpret=interpret,
            name="pyopt_affine_constant_kkt",
        )
        result = call(
            vector,
            jacobian,
            row_support,
            row_mask,
            column_rows,
            column_slots,
            column_mask,
            primal_diagonal,
            dual_diagonal,
        )
        return result[:, :total_width]

    return matvec
