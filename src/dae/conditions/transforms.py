"""Simple hard transforms for homogeneous or supplied Dirichlet conditions."""
from __future__ import annotations
import jax.numpy as jnp


def hard_initial_transform(coordinates, raw_values, target_values, *, coordinate_index: int, location: float = 0.0):
    factor=coordinates[...,coordinate_index:coordinate_index+1]-float(location)
    return target_values+factor*raw_values


def homogeneous_box_transform(coordinates, raw_values, *, spatial_indices, lower, upper):
    factor=jnp.ones(coordinates.shape[:-1]+(1,),dtype=coordinates.dtype)
    for index,lo,hi in zip(spatial_indices,lower,upper,strict=True):
        x=coordinates[...,index:index+1]
        factor=factor*(x-float(lo))*(float(hi)-x)
    return factor*raw_values
