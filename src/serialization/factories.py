"""Reusable predictor factories for custom symbolic problems."""
from __future__ import annotations

import importlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..opt import ImplicitGradientConfig, LinearSolverConfig, ProjectionConfig
from ..opt.layers import SparseProjection
from .predictor import make_network_apply


def _import_factory(path: str):
    module_name, attribute = path.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"problem factory {path!r} is not callable")
    return value


def _projection_config(value):
    value = dict(value or {})
    linear = LinearSolverConfig(**value.pop("linear_solver", {}))
    implicit = ImplicitGradientConfig(**value.pop("implicit_gradient", {}))
    return ProjectionConfig(
        **value, linear_solver=linear, implicit_gradient=implicit
    )


def projected_mlp_factory(*, params, manifest, root: Path, backend, dtype):
    """Build a predictor from a registered/importable symbolic problem factory.

    The referenced problem factory receives ``factory_config['problem_config']``
    as keyword arguments and must return a projection problem or a problem with
    ``as_projection_problem()``.
    """
    config = dict(manifest.get("factory_config", {}))
    problem_factory = _import_factory(config["problem_factory"])
    problem = problem_factory(**dict(config.get("problem_config", {})))
    projection = SparseProjection(
        problem, config=_projection_config(config.get("projection_config", {}))
    ).prepare(
        batch_size=int(manifest["batch_size"]),
        dtype=dtype,
        backend=backend,
    )
    network_apply = make_network_apply(manifest["network"])
    transform = dict(config.get("input_transform", {"type": "identity"}))
    if transform.get("type", "identity") == "standard":
        mean = jnp.asarray(transform["mean"], dtype=dtype)
        scale = jnp.asarray(transform["scale"], dtype=dtype)

        def transform_fn(x):
            return (x - mean) / scale
    else:
        transform_fn = lambda x: x

    def predict(inputs):
        raw = network_apply(params, transform_fn(inputs))
        return projection(raw, inputs)

    return predict
