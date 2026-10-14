"""Paper-compatible multiple-neighborhood Taylor reconstruction."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from math import prod
import json
import jax.numpy as jnp
import numpy as np
from .base import CompiledDiscretization, DiscretizationStrategy
from ..scaling import derivative_physical_scale


@dataclass(frozen=True)
class TaylorNeighborhood(DiscretizationStrategy):
    deltas: tuple[float, ...]
    order: int = 2
    radii: tuple[float, ...] = (1.0,)
    aggregation: str = "equal_average"

    def __post_init__(self):
        if self.order < 1:
            raise ValueError("Taylor order must be positive")
        if any(float(v) <= 0 for v in self.deltas):
            raise ValueError("Taylor deltas must be positive")
        if not self.radii or any(float(v) <= 0 for v in self.radii):
            raise ValueError("Taylor radii must be positive")
        if self.aggregation not in {"equal_average", "weighted_average"}:
            raise ValueError("unsupported Taylor aggregation")

    def compile(self, system, scaling):
        if len(self.deltas) != len(system.coordinates):
            raise ValueError("one Taylor delta is required per coordinate")
        derivatives = system.derivative_closure
        state_index = {state.name: i for i, state in enumerate(system.differential_states)}
        offsets = []
        neighborhood_coordinates = []
        for coord, delta in enumerate(self.deltas):
            for radius in self.radii:
                value = np.zeros((len(self.deltas),), dtype=np.float64)
                value[coord] = float(delta) * float(radius)
                offsets.append(value)
                neighborhood_coordinates.append((coord, float(radius)))
        offsets = np.asarray(offsets, dtype=np.float64)
        weights = np.full((len(offsets),), 1.0 / max(1, len(offsets)), dtype=np.float64)
        physical_scales = np.asarray([
            derivative_physical_scale(spec, self.deltas, spec.state.scale)
            for spec in derivatives
        ], dtype=np.float64)
        # coeff[n, d] multiplies normalized derivative d in reconstruction for its state.
        coeff = np.zeros((len(offsets), len(derivatives)), dtype=np.float64)
        for n, offset in enumerate(offsets):
            for d, spec in enumerate(derivatives):
                if spec.order > self.order:
                    continue
                raw = prod(float(offset[i]) ** power for i, power in enumerate(spec.multi_index))
                raw /= float(spec.multi_factorial)
                coeff[n, d] = raw * physical_scales[d]
        derivative_state = np.asarray([state_index[s.state.name] for s in derivatives], dtype=np.int32)
        ndiff_states = len(system.differential_states)

        def reconstruct(context_parts, normalized_derivatives):
            _coords, _params, _center, neighbors = context_parts
            # Average each neighborhood's center reconstruction.
            recs = []
            for n in range(offsets.shape[0]):
                current = neighbors[..., n, :]
                correction = jnp.zeros(current.shape, dtype=current.dtype)
                for d in range(len(derivatives)):
                    correction = correction.at[..., derivative_state[d]].add(
                        coeff[n, d] * normalized_derivatives[..., d]
                    )
                recs.append(current - correction)
            return jnp.sum(jnp.stack(recs, axis=-2) * jnp.asarray(weights)[..., None], axis=-2)

        # Estimate normalized derivatives from all neighborhoods without network
        # autodiff.  For each differential state, solve the static Taylor design
        # system C dbar ~= y(neighbor) - y(center) by a precomputed pseudoinverse.
        state_columns = []
        state_pseudoinverses = []
        for state in range(ndiff_states):
            columns = np.flatnonzero(derivative_state == state).astype(np.int32)
            state_columns.append(columns)
            if columns.size:
                state_pseudoinverses.append(
                    np.linalg.pinv(coeff[:, columns], rcond=1e-12).astype(np.float64)
                )
            else:
                state_pseudoinverses.append(np.empty((0, len(offsets)), dtype=np.float64))

        def estimate(center, neighbors):
            output = jnp.zeros(center.shape[:-1] + (len(derivatives),), dtype=center.dtype)
            for state, columns in enumerate(state_columns):
                if not len(columns):
                    continue
                rhs = neighbors[..., :, state] - center[..., state, None]
                pinv = jnp.asarray(state_pseudoinverses[state], dtype=center.dtype)
                values = jnp.einsum("dn,...n->...d", pinv, rhs)
                output = output.at[..., jnp.asarray(columns)].set(values)
            return output

        signature_data = {
            "name": "taylor_neighborhood", "deltas": self.deltas,
            "order": self.order, "radii": self.radii,
            "derivatives": [(d.state.name, d.multi_index) for d in derivatives],
        }
        signature = sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        return CompiledDiscretization(
            name="taylor_neighborhood", offsets=jnp.asarray(offsets),
            derivatives=derivatives,
            derivative_physical_scales=jnp.asarray(physical_scales),
            reconstruct_differential_states=reconstruct,
            estimate_derivatives=estimate,
            signature=signature,
        )
