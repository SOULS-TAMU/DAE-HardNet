"""Deterministic array datasets for DAE training, conditions, and examples."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class DAEData:
    coordinates: np.ndarray
    parameters: np.ndarray
    states: np.ndarray
    derivatives: np.ndarray | None = None
    initial_coordinates: np.ndarray | None = None
    initial_targets: np.ndarray | None = None
    boundary_coordinates: np.ndarray | None = None
    boundary_targets: np.ndarray | None = None
    initial_mask: np.ndarray | None = None
    boundary_mask: np.ndarray | None = None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        def value(x): return np.asarray([]) if x is None else np.asarray(x)
        np.savez(
            path,
            coordinates=self.coordinates,
            parameters=self.parameters,
            states=self.states,
            derivatives=value(self.derivatives),
            initial_coordinates=value(self.initial_coordinates),
            initial_targets=value(self.initial_targets),
            boundary_coordinates=value(self.boundary_coordinates),
            boundary_targets=value(self.boundary_targets),
            initial_mask=value(self.initial_mask),
            boundary_mask=value(self.boundary_mask),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        def optional(name):
            value = data[name]
            return None if value.size == 0 else value
        return cls(
            data["coordinates"], data["parameters"], data["states"],
            optional("derivatives"), optional("initial_coordinates"),
            optional("initial_targets"), optional("boundary_coordinates"),
            optional("boundary_targets"), optional("initial_mask"),
            optional("boundary_mask"),
        )
