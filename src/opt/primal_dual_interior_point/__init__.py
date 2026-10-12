"""Projection backend namespace."""
from .solver import SparseProjection, PreparedProjection

def __getattr__(name):
    from importlib import import_module
    return getattr(import_module("kkthn.opt.primal_dual_interior_point"), name)
