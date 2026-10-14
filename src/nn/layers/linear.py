"""Linear neural-network layers."""
from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from ...typing import Layer


def Dense(output_size: int, *, use_bias: bool = True,
          kernel_init: Callable | None = None,
          bias_init: Callable | None = None,
          dtype=jnp.float64) -> Layer:
    output_size = int(output_size)
    if output_size < 1:
        raise ValueError('output_size must be positive')
    kernel_init = kernel_init or jax.nn.initializers.glorot_uniform()
    bias_init = bias_init or jax.nn.initializers.zeros
    def init(key, input_shape):
        input_shape = tuple(map(int, input_shape))
        if not input_shape or input_shape[-1] < 1:
            raise ValueError('input_shape must end in a positive feature dimension')
        k1, k2 = jax.random.split(key)
        params = {'kernel': kernel_init(k1, (input_shape[-1], output_size), dtype)}
        if use_bias:
            params['bias'] = bias_init(k2, (output_size,), dtype)
        return (*input_shape[:-1], output_size), params
    def apply(params, inputs, **_kwargs):
        outputs = jnp.asarray(inputs) @ params['kernel']
        if use_bias:
            outputs = outputs + params['bias']
        return outputs
    return Layer(init, apply)
