"""Backend-aware linear-solve helpers for JAX.

The dense path uses ``jax.numpy.linalg.solve`` inside
``jax.lax.custom_linear_solve``.  JAX/XLA dispatches the compiled operation to
the active CPU or accelerator backend.  The custom linear solve gives the
operation an implicit transpose-solve rule instead of differentiating through
factorization internals.

An optional CSR sparse-QR entry point is exposed for callers that already have
a static CSR KKT representation.  JAX currently provides a CUDA implementation
and a SciPy CPU fallback for that experimental routine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def dense_linear_solve(
    matrix,
    rhs,
    *,
    symmetric: bool = False,
):
    """Solve a dense linear system with an implicit differentiation rule."""
    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)

    def matvec(vector):
        return matrix @ vector

    def solve(_matvec, right_hand_side):
        return jnp.linalg.solve(matrix, right_hand_side)

    if symmetric:
        return lax.custom_linear_solve(
            matvec,
            rhs,
            solve=solve,
            symmetric=True,
        )

    def transpose_solve(_vecmat, right_hand_side):
        return jnp.linalg.solve(matrix.T, right_hand_side)

    return lax.custom_linear_solve(
        matvec,
        rhs,
        solve=solve,
        transpose_solve=transpose_solve,
        symmetric=False,
    )


def sparse_csr_qr_solve(
    data,
    indices,
    indptr,
    rhs,
    *,
    tolerance: float = 1e-6,
    reorder: int = 1,
):
    """Solve a CSR system with JAX's experimental sparse QR implementation.

    This path is useful only when the caller already owns a CSR representation
    with a fixed sparsity structure.  A matrix right-hand side is handled by
    vectorizing independent sparse solves.
    """
    from jax.experimental.sparse.linalg import spsolve

    rhs = jnp.asarray(rhs)

    def solve_one(vector):
        return spsolve(
            data,
            indices,
            indptr,
            vector,
            tol=tolerance,
            reorder=reorder,
        )

    if rhs.ndim == 1:
        return solve_one(rhs)
    if rhs.ndim == 2:
        return jax.vmap(solve_one, in_axes=1, out_axes=1)(rhs)
    raise ValueError(
        "rhs must be one- or two-dimensional; "
        f"received shape {rhs.shape}."
    )


def device_summary() -> tuple[tuple[str, str, int], ...]:
    """Return ``(platform, device_kind, id)`` for currently visible devices."""
    return tuple(
        (device.platform, device.device_kind, device.id)
        for device in jax.devices()
    )
