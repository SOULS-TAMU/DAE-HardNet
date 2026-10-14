"""One-sided finite-difference derivative anchors."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import jax.numpy as jnp
import numpy as np
from .base import CompiledDiscretization, DiscretizationStrategy
from ..scaling import derivative_physical_scale


@dataclass(frozen=True)
class ForwardDifference(DiscretizationStrategy):
    deltas: tuple[float, ...]

    def compile(self, system, scaling):
        del scaling
        if len(self.deltas) != len(system.coordinates):
            raise ValueError("one forward delta is required per coordinate")
        derivatives = system.derivative_closure
        offsets = []
        offset_lookup = {}
        for coord, delta in enumerate(self.deltas):
            for multiple in (1, 2):
                value = np.zeros((len(self.deltas),), dtype=np.float64)
                value[coord] = multiple * float(delta)
                offset_lookup[(coord, multiple)] = len(offsets)
                offsets.append(value)
        offsets = np.asarray(offsets)
        scales = np.asarray([derivative_physical_scale(d, self.deltas, d.state.scale) for d in derivatives])
        state_index = {s.name: i for i, s in enumerate(system.differential_states)}

        def reconstruct(parts, normalized_derivatives):
            del normalized_derivatives
            return parts[2]  # center neural state is the state anchor in difference modes

        def estimate(center, neighbors):
            values = []
            for spec in derivatives:
                coord = spec.pure_coordinate
                state = state_index[spec.state.name]
                if coord is None:
                    values.append(jnp.asarray(0.0, center.dtype)); continue
                h = float(self.deltas[coord])
                y0 = center[..., state]
                y1 = neighbors[..., offset_lookup[(coord, 1)], state]
                if spec.order == 1:
                    physical = (y1 - y0) / h
                elif spec.order == 2:
                    y2 = neighbors[..., offset_lookup[(coord, 2)], state]
                    physical = (y2 - 2.0 * y1 + y0) / (h * h)
                else:
                    physical = jnp.asarray(0.0, center.dtype)
                values.append(physical / scales[len(values)])
            return jnp.stack(values, axis=-1)

        signature = sha256(json.dumps({"name":"forward_difference","deltas":self.deltas},sort_keys=True).encode()).hexdigest()
        return CompiledDiscretization("forward_difference", jnp.asarray(offsets), derivatives,
            jnp.asarray(scales), reconstruct, estimate, signature)
