"""Activation layers wrapping :mod:`jax.nn` functions."""
import jax
import jax.numpy as jnp
from .core import Elementwise

def Relu(): return Elementwise(jax.nn.relu)
def ReLU(): return Relu()
def Gelu(): return Elementwise(jax.nn.gelu)
def GELU(): return Gelu()
def Silu(): return Elementwise(jax.nn.silu)
def SiLU(): return Silu()
def Sigmoid(): return Elementwise(jax.nn.sigmoid)
def Tanh(): return Elementwise(jnp.tanh)
def Softplus(): return Elementwise(jax.nn.softplus)
