"""Symmetric finite-difference derivative anchors."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import jax.numpy as jnp
import numpy as np
from .base import CompiledDiscretization, DiscretizationStrategy
from ..scaling import derivative_physical_scale


@dataclass(frozen=True)
class CentralDifference(DiscretizationStrategy):
    deltas: tuple[float, ...]

    def compile(self, system, scaling):
        del scaling
        if len(self.deltas) != len(system.coordinates):
            raise ValueError("one central delta is required per coordinate")
        derivatives = system.derivative_closure
        offsets=[]; lookup={}
        for coord, delta in enumerate(self.deltas):
            for sign in (-1, 1):
                value=np.zeros((len(self.deltas),), dtype=np.float64); value[coord]=sign*float(delta)
                lookup[(coord,sign)]=len(offsets); offsets.append(value)
        offsets=np.asarray(offsets)
        scales=np.asarray([derivative_physical_scale(d,self.deltas,d.state.scale) for d in derivatives])
        state_index={s.name:i for i,s in enumerate(system.differential_states)}
        def reconstruct(parts, normalized_derivatives):
            del normalized_derivatives
            return parts[2]
        def estimate(center, neighbors):
            vals=[]
            for i,spec in enumerate(derivatives):
                coord=spec.pure_coordinate; state=state_index[spec.state.name]
                if coord is None:
                    physical=jnp.asarray(0.0,center.dtype)
                else:
                    h=float(self.deltas[coord]); ym=neighbors[...,lookup[(coord,-1)],state]; yp=neighbors[...,lookup[(coord,1)],state]
                    if spec.order==1: physical=(yp-ym)/(2*h)
                    elif spec.order==2: physical=(yp-2*center[...,state]+ym)/(h*h)
                    else: physical=jnp.asarray(0.0,center.dtype)
                vals.append(physical/scales[i])
            return jnp.stack(vals,axis=-1)
        signature=sha256(json.dumps({"name":"central_difference","deltas":self.deltas},sort_keys=True).encode()).hexdigest()
        return CompiledDiscretization("central_difference",jnp.asarray(offsets),derivatives,jnp.asarray(scales),reconstruct,estimate,signature)
