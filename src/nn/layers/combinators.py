"""Functional layer combinators."""
from __future__ import annotations
import jax
from ...typing import Layer


def Serial(*layers: Layer) -> Layer:
    layers = tuple(layers)
    def init(key, input_shape):
        if not layers:
            return tuple(input_shape), ()
        keys = jax.random.split(key, len(layers))
        shape = tuple(input_shape)
        params = []
        for layer, subkey in zip(layers, keys, strict=True):
            shape, layer_params = layer.init(subkey, shape)
            params.append(layer_params)
        return shape, tuple(params)
    def apply(params, inputs, **kwargs):
        values = inputs
        for layer, layer_params in zip(layers, params, strict=True):
            values = layer.apply(layer_params, values, **kwargs)
        return values
    return Layer(init, apply)


def Residual(layer: Layer) -> Layer:
    def init(key, input_shape):
        output_shape, params = layer.init(key, input_shape)
        if tuple(output_shape) != tuple(input_shape):
            raise ValueError('Residual layer must preserve shape')
        return tuple(output_shape), params
    def apply(params, inputs, **kwargs):
        return inputs + layer.apply(params, inputs, **kwargs)
    return Layer(init, apply)


def Parallel(*layers: Layer) -> Layer:
    layers = tuple(layers)
    def init(key, input_shape):
        keys = jax.random.split(key, len(layers))
        shapes, params = [], []
        for layer, subkey in zip(layers, keys, strict=True):
            shape, p = layer.init(subkey, input_shape)
            shapes.append(shape); params.append(p)
        return tuple(shapes), tuple(params)
    def apply(params, inputs, **kwargs):
        return tuple(layer.apply(p, inputs, **kwargs) for layer, p in zip(layers, params, strict=True))
    return Layer(init, apply)


def FanOut(count: int) -> Layer:
    count = int(count)
    if count < 1: raise ValueError('count must be positive')
    def init(_key, input_shape): return tuple(tuple(input_shape) for _ in range(count)), ()
    def apply(_params, inputs, **_kwargs): return tuple(inputs for _ in range(count))
    return Layer(init, apply)


def FanInSum() -> Layer:
    def init(_key, input_shapes):
        shapes = tuple(tuple(s) for s in input_shapes)
        if not shapes or any(s != shapes[0] for s in shapes):
            raise ValueError('FanInSum input shapes must match')
        return shapes[0], ()
    def apply(_params, inputs, **_kwargs):
        first, *rest = inputs
        out = first
        for item in rest: out = out + item
        return out
    return Layer(init, apply)
