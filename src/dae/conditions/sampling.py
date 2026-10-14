"""Deterministic condition-point sampling."""
from __future__ import annotations
import numpy as np


def sample_condition_coordinates(condition, lower, upper, count: int, *, seed: int = 0):
    lower=np.asarray(lower,dtype=np.float64); upper=np.asarray(upper,dtype=np.float64)
    if count < 1: raise ValueError("count must be positive")
    if condition.coordinate_index < 0 or condition.coordinate_index >= lower.size:
        raise ValueError("condition coordinate index is out of range")
    rng=np.random.default_rng(seed)
    points=rng.uniform(lower,upper,size=(count,lower.size))
    points[:,condition.coordinate_index]=condition.location
    return points
