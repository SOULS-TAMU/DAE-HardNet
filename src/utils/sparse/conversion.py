"""Shared KKTHN sparse conversion."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.utils.sparse.conversion"), name)
