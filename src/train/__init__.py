"""Functional training helpers and the preserved projected-neural implementation."""
from .steps import make_train_step, make_eval_step
from .schedules import ProjectionSchedule
from .logging import KKTTableLogger
from .model import ProjectedNeuralModel
from .projected_neural import (
    Adam,
    AdamState,
    BuildReport,
    CompiledProjectedGraphs,
    ImplicitProjectionLayer,
    MLPBackbone,
    PredictionOutput,
    ProjectedMetrics,
    ProjectionLayerOutput,
    apply_updates,
    configure_compilation_cache,
    make_implicit_projection_layer,
)

__all__ = [
    "make_train_step",
    "make_eval_step",
    "ProjectionSchedule",
    "KKTTableLogger",
    "Adam",
    "AdamState",
    "BuildReport",
    "CompiledProjectedGraphs",
    "ImplicitProjectionLayer",
    "MLPBackbone",
    "PredictionOutput",
    "ProjectedMetrics",
    "ProjectedNeuralModel",
    "ProjectionLayerOutput",
    "apply_updates",
    "configure_compilation_cache",
    "make_implicit_projection_layer",
]
