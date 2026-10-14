"""Direct access to KKTHN fischer_burmeister.reduced; no copied kernels."""
from importlib import import_module

def __getattr__(name):
    return getattr(import_module("kkthn.opt.fischer_burmeister.reduced"), name)
