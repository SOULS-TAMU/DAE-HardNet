"""Primal-dual state containers for the NumPy IPOPT reference solver.

This module defines the eight vector blocks used by IPOPT's primal-dual
interior-point method:

    (x, s, y_c, y_d, z_L, z_U, v_L, v_U)

The implementation is intentionally NumPy-only.  A later JAX version can mirror
the same data layout and public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_float_vector(value: FloatArray, name: str) -> FloatArray:
    """Return *value* as a one-dimensional float64 NumPy array.

    A copy is not forced when the input already has the expected representation.
    """
    array = np.asarray(value, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array; received shape {array.shape}."
        )

    return array


@dataclass
class PrimalDualState:
    """Complete primal-dual state used by the IPOPT Newton system.

    Attributes
    ----------
    x
        Original primal variables.
    s
        Slack variables for ranged constraints.
    y_c
        Multipliers for equality constraints ``c(x) = 0``.
    y_d
        Multipliers for slack equations ``d(x) - s = 0``.
    z_L
        Multipliers for finite lower bounds on ``x``.
    z_U
        Multipliers for finite upper bounds on ``x``.
    v_L
        Multipliers for finite lower bounds on ``s``.
    v_U
        Multipliers for finite upper bounds on ``s``.
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
        """Normalize every state block to a one-dimensional float64 array."""
        self.x = _as_float_vector(self.x, "x")
        self.s = _as_float_vector(self.s, "s")
        self.y_c = _as_float_vector(self.y_c, "y_c")
        self.y_d = _as_float_vector(self.y_d, "y_d")
        self.z_L = _as_float_vector(self.z_L, "z_L")
        self.z_U = _as_float_vector(self.z_U, "z_U")
        self.v_L = _as_float_vector(self.v_L, "v_L")
        self.v_U = _as_float_vector(self.v_U, "v_U")

        if self.y_d.size != self.s.size:
            raise ValueError(
                "y_d and s must have the same length because y_d is the "
                "multiplier for d(x) - s = 0; "
                f"received len(y_d)={self.y_d.size} and len(s)={self.s.size}."
            )

    @property
    def primal_size(self) -> int:
        """Total number of primal variables, ``len(x) + len(s)``."""
        return int(self.x.size + self.s.size)

    @property
    def dual_size(self) -> int:
        """Total number of equality and bound multiplier variables."""
        return int(
            self.y_c.size
            + self.y_d.size
            + self.z_L.size
            + self.z_U.size
            + self.v_L.size
            + self.v_U.size
        )

    @property
    def size(self) -> int:
        """Total number of scalar entries in the state."""
        return self.primal_size + self.dual_size

    @property
    def block_sizes(self) -> tuple[int, int, int, int, int, int, int, int]:
        """Lengths of the eight state blocks in flattening order."""
        return (
            int(self.x.size),
            int(self.s.size),
            int(self.y_c.size),
            int(self.y_d.size),
            int(self.z_L.size),
            int(self.z_U.size),
            int(self.v_L.size),
            int(self.v_U.size),
        )

    def blocks(self) -> tuple[FloatArray, ...]:
        """Return the eight state arrays in canonical IPOPT order."""
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
        """Iterate over ``(block_name, array)`` pairs in canonical order."""
        names = ("x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U")
        return iter(zip(names, self.blocks(), strict=True))

    def validate_finite(self) -> None:
        """Raise ``ValueError`` if any block contains NaN or infinity."""
        for name, block in self.named_blocks():
            if not np.all(np.isfinite(block)):
                bad_indices = np.flatnonzero(~np.isfinite(block))
                raise ValueError(
                    f"{name} contains non-finite values at indices "
                    f"{bad_indices.tolist()}."
                )

    def copy(self) -> "PrimalDualState":
        """Return a deep copy of all eight state blocks."""
        return PrimalDualState(*(block.copy() for block in self.blocks()))

    def flatten(self) -> FloatArray:
        """Concatenate all blocks into one vector.

        Flattening order:

        ``x, s, y_c, y_d, z_L, z_U, v_L, v_U``.
        """
        if self.size == 0:
            return np.empty(0, dtype=np.float64)

        return np.concatenate(self.blocks())

    @classmethod
    def zeros(
        cls,
        *,
        n_x: int,
        n_s: int,
        n_c: int,
        n_x_lower: int,
        n_x_upper: int,
        n_s_lower: int,
        n_s_upper: int,
    ) -> "PrimalDualState":
        """Create a zero-filled state from block dimensions."""
        dimensions = {
            "n_x": n_x,
            "n_s": n_s,
            "n_c": n_c,
            "n_x_lower": n_x_lower,
            "n_x_upper": n_x_upper,
            "n_s_lower": n_s_lower,
            "n_s_upper": n_s_upper,
        }

        for name, value in dimensions.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative; received {value}.")

        zeros = lambda length: np.zeros(length, dtype=np.float64)

        return cls(
            x=zeros(n_x),
            s=zeros(n_s),
            y_c=zeros(n_c),
            y_d=zeros(n_s),
            z_L=zeros(n_x_lower),
            z_U=zeros(n_x_upper),
            v_L=zeros(n_s_lower),
            v_U=zeros(n_s_upper),
        )

    @classmethod
    def zeros_like(cls, other: "PrimalDualState") -> "PrimalDualState":
        """Create a zero-filled state with the same block shapes as ``other``."""
        return cls(*(np.zeros_like(block) for block in other.blocks()))

    @classmethod
    def from_flat(
        cls,
        flat: FloatArray,
        *,
        template: "PrimalDualState",
    ) -> "PrimalDualState":
        """Split a flat vector using the block sizes of ``template``."""
        vector = _as_float_vector(flat, "flat")

        if vector.size != template.size:
            raise ValueError(
                "Flat vector size does not match the template state: "
                f"received {vector.size}, expected {template.size}."
            )

        blocks: list[FloatArray] = []
        start = 0

        for block_size in template.block_sizes:
            stop = start + block_size
            blocks.append(vector[start:stop].copy())
            start = stop

        return cls(*blocks)

    def axpy(self, alpha: float, direction: "PrimalDualState") -> "PrimalDualState":
        """Return ``self + alpha * direction`` block by block."""
        if self.block_sizes != direction.block_sizes:
            raise ValueError(
                "State and direction block sizes must match: "
                f"{self.block_sizes} != {direction.block_sizes}."
            )

        return PrimalDualState(
            *(
                block + float(alpha) * direction_block
                for block, direction_block in zip(
                    self.blocks(), direction.blocks(), strict=True
                )
            )
        )
