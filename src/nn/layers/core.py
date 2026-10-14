"""Functional JAX layer protocol and basic layers."""
from __future__ import annotations
from typing import Any, Callable
import jax
import jax.numpy as jnp
from ...typing import Layer


def Identity() -> Layer:
    def init(_key, input_shape):
        return tuple(input_shape), ()
    def apply(_params, inputs, **_kwargs):
        return inputs
    return Layer(init, apply)


def Lambda(function: Callable[[jax.Array], jax.Array]) -> Layer:
    if not callable(function):
        raise TypeError('function must be callable')
    def init(_key, input_shape):
        return tuple(input_shape), ()
    def apply(_params, inputs, **kwargs):
        try:
            return function(inputs, **kwargs)
        except TypeError:
            return function(inputs)
    return Layer(init, apply)


def Elementwise(function: Callable[[jax.Array], jax.Array]) -> Layer:
    return Lambda(function)


def Dropout(rate: float) -> Layer:
    rate = float(rate)
    if not 0.0 <= rate < 1.0:
        raise ValueError('rate must satisfy 0 <= rate < 1')
    keep = 1.0 - rate
    def init(_key, input_shape):
        return tuple(input_shape), ()
    def apply(_params, inputs, *, rng=None, training=False, **_kwargs):
        if not training or rate == 0.0:
            return inputs
        if rng is None:
            raise ValueError('Dropout requires rng when training=True')
        mask = jax.random.bernoulli(rng, keep, inputs.shape)
        return jnp.where(mask, inputs / keep, 0.0)
    return Layer(init, apply)


def LayerNorm(*, epsilon: float = 1e-5) -> Layer:
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    def init(_key, input_shape):
        width = int(input_shape[-1])
        params = {'scale': jnp.ones((width,)), 'bias': jnp.zeros((width,))}
        return tuple(input_shape), params
    def apply(params, inputs, **_kwargs):
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
        normalized = (inputs - mean) * jax.lax.rsqrt(variance + epsilon)
        return normalized * params['scale'] + params['bias']
    return Layer(init, apply)
