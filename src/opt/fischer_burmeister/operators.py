"""Direct access to KKTHN fischer_burmeister.operators; no copied kernels."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.opt.fischer_burmeister.operators"), name)
