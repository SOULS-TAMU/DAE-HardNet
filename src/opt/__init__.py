"""Optimization configuration and organized solver namespaces.

Subpackages are loaded lazily so low-level solver modules can depend on one
another without package-initialization cycles. This affects Python import time
only; prepared JAX computations call the organized implementation modules
directly.
"""
from importlib import import_module

from .config import ProjectionConfig, LinearSolverConfig, ImplicitGradientConfig
from .solvers import solve

_LAZY_SUBMODULES = {"layers", "solvers", "implicit", "sparse", "api", "linear_equality", "fischer_burmeister", "primal_dual_interior_point"}


def __getattr__(name):
    if name in {"prepare_projection", "registered_methods"}:
        return getattr(import_module(f"{__name__}.api"), name)
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = [
    "ProjectionConfig",
    "LinearSolverConfig",
    "ImplicitGradientConfig",
    "solve",
    "prepare_projection",
    "registered_methods",
    "layers",
    "solvers",
    "implicit",
    "sparse",
]
