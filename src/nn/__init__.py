"""Neural-network functions extending a controlled subset of :mod:`jax.nn`."""
from jax.nn import (celu, elu, gelu, glu, hard_sigmoid, hard_silu, hard_swish,
    leaky_relu, log_sigmoid, log_softmax, one_hot, relu, relu6, selu, sigmoid,
    silu, soft_sign, softmax, softplus, standardize, swish)
from jax.nn import initializers
from . import layers

__all__ = [
    'celu','elu','gelu','glu','hard_sigmoid','hard_silu','hard_swish',
    'leaky_relu','log_sigmoid','log_softmax','one_hot','relu','relu6','selu',
    'sigmoid','silu','soft_sign','softmax','softplus','standardize','swish',
    'initializers','layers'
]
