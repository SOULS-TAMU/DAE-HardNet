"""Neural-network composition and full-model derivatives for compiled DAE systems."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def extract_network_derivatives(
    apply_fn,
    parameters,
    coordinates,
    derivatives,
    differential_states,
):
    """Evaluate selected derivatives of an unprojected network with autodiff."""
    outputs = []
    if isinstance(differential_states, int):
        unique = []
        for spec in derivatives:
            if spec.state.name not in unique:
                unique.append(spec.state.name)
        state_map = {name: i for i, name in enumerate(unique)}
    else:
        state_map = {state.name: i for i, state in enumerate(differential_states)}
    for spec in derivatives:
        repeated = []
        for coordinate, power in enumerate(spec.multi_index):
            repeated.extend([coordinate] * power)

        def scalar(x, state_index=state_map[spec.state.name]):
            return apply_fn(parameters, x)[state_index]

        function = scalar
        for coordinate in repeated:
            previous = function
            function = (
                lambda x, previous=previous, coordinate=coordinate: (
                    jax.grad(previous)(x)[coordinate]
                )
            )
        outputs.append(jax.vmap(function)(coordinates))
    if not outputs:
        return jnp.empty(coordinates.shape[:-1] + (0,), coordinates.dtype)
    return jnp.stack(outputs, axis=-1)


def _central_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric unit-grid nodes and finite-difference weights.

    The stencil is exact for polynomials through the number of stencil nodes
    minus one.  Orders one and two use the standard three-point family.
    """
    order = int(order)
    if order < 0:
        raise ValueError("derivative order must be nonnegative")
    if order == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    radius = max(1, (order + 1) // 2)
    while 2 * radius + 1 <= order:
        radius += 1
    nodes = np.arange(-radius, radius + 1, dtype=np.float64)
    powers = np.arange(nodes.size, dtype=np.int32)
    vandermonde = nodes[None, :] ** powers[:, None]
    rhs = np.zeros((nodes.size,), dtype=np.float64)
    rhs[order] = float(factorial(order))
    weights = np.linalg.solve(vandermonde, rhs)
    return nodes, weights


@dataclass(frozen=True)
class FullModelDerivativePlan:
    """Static differentiable stencil for derivatives of final model outputs.

    The plan applies finite-difference stencils to the *complete* projected
    model mapping.  Every shifted state is therefore produced by
    NN -> projection -> reconstruction, rather than read from the projection's
    independent derivative variables.
    """

    shifts: Any
    coefficients: Any
    state_indices: Any
    steps: tuple[float, ...]
    zero_shift_index: int

    @classmethod
    def build(cls, derivatives, differential_states, steps):
        steps = tuple(float(value) for value in steps)
        if any(value <= 0.0 for value in steps):
            raise ValueError("full-model derivative steps must be positive")
        coordinate_size = len(steps)
        state_map = {state.name: i for i, state in enumerate(differential_states)}
        shift_to_index: dict[tuple[float, ...], int] = {}
        rows: list[dict[int, float]] = []
        state_indices = []

        for spec in derivatives:
            if len(spec.multi_index) != coordinate_size:
                raise ValueError("derivative multi-index and step dimension differ")
            per_coordinate = []
            for coordinate, order in enumerate(spec.multi_index):
                nodes, weights = _central_weights(int(order))
                scale = steps[coordinate] ** int(order)
                per_coordinate.append((nodes, weights / scale))

            row: dict[int, float] = {}
            choices = [range(len(nodes)) for nodes, _weights in per_coordinate]
            for selected in product(*choices):
                shift = []
                coefficient = 1.0
                for coordinate, node_index in enumerate(selected):
                    nodes, weights = per_coordinate[coordinate]
                    shift.append(float(nodes[node_index] * steps[coordinate]))
                    coefficient *= float(weights[node_index])
                key = tuple(shift)
                index = shift_to_index.setdefault(key, len(shift_to_index))
                row[index] = row.get(index, 0.0) + coefficient
            rows.append(row)
            state_indices.append(state_map[spec.state.name])

        shifts = np.zeros((len(shift_to_index), coordinate_size), dtype=np.float64)
        for shift, index in shift_to_index.items():
            shifts[index] = shift
        coefficients = np.zeros((len(rows), len(shift_to_index)), dtype=np.float64)
        for derivative_index, row in enumerate(rows):
            for shift_index, value in row.items():
                coefficients[derivative_index, shift_index] = value
        zero_shift_index = shift_to_index.get(tuple(0.0 for _ in steps), -1)
        return cls(
            shifts=jnp.asarray(shifts),
            coefficients=jnp.asarray(coefficients),
            state_indices=jnp.asarray(state_indices, dtype=jnp.int32),
            steps=steps,
            zero_shift_index=int(zero_shift_index),
        )

    def evaluate(
        self,
        projected_model,
        parameters,
        coordinates,
        physical_parameters,
        *,
        center_states=None,
    ):
        """Differentiate complete projected outputs with one batched stencil.

        All shifted coordinate sets are flattened into one large projection
        batch.  This keeps the number of projection-layer invocations constant
        even when several first- and second-order derivatives are requested.
        """
        del center_states
        coordinates = jnp.asarray(coordinates)
        physical_parameters = jnp.asarray(physical_parameters)
        shifts = jnp.asarray(self.shifts, dtype=coordinates.dtype)
        batch_size = coordinates.shape[0]
        shift_count = shifts.shape[0]
        shifted = coordinates[:, None, :] + shifts[None, :, :]
        flat_coordinates = shifted.reshape(
            (batch_size * shift_count, coordinates.shape[-1])
        )
        parameter_size = physical_parameters.shape[-1]
        flat_parameters = jnp.broadcast_to(
            physical_parameters[:, None, :],
            (batch_size, shift_count, parameter_size),
        ).reshape((batch_size * shift_count, parameter_size))
        decoded, _info, _context, _target = projected_model.forward(
            parameters,
            flat_coordinates,
            flat_parameters,
        )
        stacked = decoded["states"].reshape(
            (batch_size, shift_count, decoded["states"].shape[-1])
        )
        selected = jnp.take(stacked, self.state_indices, axis=-1)
        return jnp.einsum(
            "ds,bsd->bd",
            jnp.asarray(self.coefficients, dtype=coordinates.dtype),
            selected,
        )



@dataclass(frozen=True)
class ProjectedDAEModel:
    compiled: Any
    backbone_apply: Any
    prepared_projection: Any

    def forward(self, parameters, coordinates, physical_parameters):
        coordinates = jnp.asarray(coordinates)
        offsets = jnp.asarray(
            self.compiled.discretization.offsets, dtype=coordinates.dtype
        )
        all_offsets = jnp.concatenate(
            (jnp.zeros((1, coordinates.shape[-1]), coordinates.dtype), offsets),
            axis=0,
        )
        points = coordinates[:, None, :] + all_offsets[None, :, :]
        flat = points.reshape((-1, coordinates.shape[-1]))
        predicted = self.backbone_apply(parameters, flat).reshape(
            (coordinates.shape[0], all_offsets.shape[0], -1)
        )
        center_all = predicted[:, 0, :]
        ndiff = len(self.compiled.system.differential_states)
        center_diff = center_all[:, :ndiff]
        neighbors = predicted[:, 1:, :ndiff]
        d_anchor_normalized = self.compiled.discretization.estimate_derivatives(
            center_diff, neighbors
        )
        context = self.compiled.pack_context(
            coordinates, physical_parameters, center_diff, neighbors
        )
        target = self.compiled.build_target(center_all, d_anchor_normalized)
        projected, info = self.prepared_projection.with_info(target, context)
        decoded = self.compiled.decode(context, projected)
        return decoded, info, context, target
