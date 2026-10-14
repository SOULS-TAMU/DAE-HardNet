"""Shared KKTHN sparse assembly."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.utils.sparse.assembly"), name)
