"""Functional neural-network layers."""
from ...typing import Layer
from .core import Identity, Lambda, Elementwise, Dropout, LayerNorm
from .linear import Dense
from .activations import Relu, ReLU, Gelu, GELU, Silu, SiLU, Sigmoid, Tanh, Softplus
from .combinators import Serial, Residual, Parallel, FanOut, FanInSum

def MLP(output_size, hidden_sizes=(64,64), *, activation=Relu, **dense_kwargs):
    layers=[]
    for width in hidden_sizes:
        layers.extend((Dense(width, **dense_kwargs), activation()))
    layers.append(Dense(output_size, **dense_kwargs))
    return Serial(*layers)

__all__ = ['Layer','Dense','Identity','Lambda','Elementwise','Dropout','LayerNorm',
'Relu','ReLU','Gelu','GELU','Silu','SiLU','Sigmoid','Tanh','Softplus',
'Serial','Residual','Parallel','FanOut','FanInSum','MLP']
