"""Static variable and residual scaling for DAE projection problems."""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
import numpy as np
from .derivatives import DerivativeSpec


@dataclass(frozen=True)
class ScalingConfig:
    residual_mode: str = "row_norm"
    metric: str = "normalized"
    minimum_scale: float = 1e-12
    maximum_scale: float = 1e12
    condition_warning: float = 1e10

    def __post_init__(self):
        if self.residual_mode not in {"none", "row_norm"}:
            raise ValueError("residual_mode must be none or row_norm")
        if self.metric not in {"normalized", "preserve_physical"}:
            raise ValueError("metric must be normalized or preserve_physical")
        if self.minimum_scale <= 0 or self.maximum_scale < self.minimum_scale:
            raise ValueError("invalid scale bounds")


def derivative_physical_scale(spec: DerivativeSpec, deltas, state_scale: float) -> float:
    h_alpha = prod(float(deltas[i]) ** power for i, power in enumerate(spec.multi_index))
    normalized = h_alpha / (float(spec.multi_factorial) * float(state_scale))
    if normalized == 0:
        raise ValueError("derivative scaling requires nonzero neighborhood deltas")
    # physical derivative = scale * normalized derivative
    return 1.0 / normalized


def safe_row_scales(matrix, config: ScalingConfig):
    matrix = np.asarray(matrix, dtype=np.float64)
    if config.residual_mode == "none" or matrix.size == 0:
        return np.ones((matrix.shape[0],), dtype=np.float64)
    scales = np.linalg.norm(matrix, ord=np.inf, axis=1)
    scales = np.clip(scales, config.minimum_scale, config.maximum_scale)
    return scales
