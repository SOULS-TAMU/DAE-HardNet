"""Portable JAX exported-predictor artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..serialization.predictor import Predictor
from ..runtime._jax_export import deserialize_exported


def _dtype(name: str):
    return jnp.float64 if str(name) == "float64" else jnp.float32


def load_portable_predictor(path: str | Path) -> Predictor:
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "daehn_portable_predictor_v1":
        raise ValueError("unsupported portable predictor manifest")
    _exported, exported_call, _api = deserialize_exported(path.read_bytes())
    dtype = _dtype(manifest["dtype"])

    def predict_fn(inputs):
        return exported_call(inputs)

    return Predictor(
        predict_fn=predict_fn,
        batch_size=int(manifest["batch_size"]),
        input_size=int(manifest["input_size"]),
        output_size=int(manifest["output_size"]),
        dtype=dtype,
        backend="portable",
        source="portable",
    )
