"""Direct access to KKTHN primal_dual_interior_point.assembly; no copied kernels."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.opt.primal_dual_interior_point.assembly"), name)
