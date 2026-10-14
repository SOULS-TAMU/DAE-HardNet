"""Projection backend namespace."""
from .solver import FischerBurmeisterProjection, PreparedFischerBurmeisterProjection

def __getattr__(name):
    from importlib import import_module
    return getattr(import_module("kkthn.opt.fischer_burmeister"), name)
