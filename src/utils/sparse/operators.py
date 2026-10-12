"""Shared KKTHN sparse operators."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.utils.sparse.operators"), name)
