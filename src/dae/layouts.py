"""Static array layouts used by compiled DAE problems."""
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class ProjectionLayout:
    derivative_slice: slice
    algebraic_slice: slice
    size: int

    def unpack(self, values):
        values = jnp.asarray(values)
        return values[..., self.derivative_slice], values[..., self.algebraic_slice]


@dataclass(frozen=True)
class ContextLayout:
    coordinate_slice: slice
    parameter_slice: slice
    center_state_slice: slice
    neighbor_state_slice: slice
    coordinate_count: int
    parameter_count: int
    differential_state_count: int
    neighborhood_count: int
    size: int

    def pack(self, coordinates, parameters, center_states, neighbor_states):
        coordinates = jnp.asarray(coordinates)
        center_states = jnp.asarray(center_states)
        parameters = jnp.asarray(parameters)
        neighbor_states = jnp.asarray(neighbor_states)
        batch_shape = coordinates.shape[:-1]
        flat_neighbors = neighbor_states.reshape(batch_shape + (-1,))
        return jnp.concatenate((coordinates, parameters, center_states, flat_neighbors), axis=-1)

    def unpack(self, context):
        context = jnp.asarray(context)
        coords = context[..., self.coordinate_slice]
        parameters = context[..., self.parameter_slice]
        center = context[..., self.center_state_slice]
        neighbors = context[..., self.neighbor_state_slice].reshape(
            context.shape[:-1] + (self.neighborhood_count, self.differential_state_count)
        )
        return coords, parameters, center, neighbors
