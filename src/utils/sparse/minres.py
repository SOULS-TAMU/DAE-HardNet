"""Shared KKTHN sparse minres."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.utils.sparse.minres"), name)
