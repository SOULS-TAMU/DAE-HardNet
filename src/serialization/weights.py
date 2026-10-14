"""Pickle-free PyTree array serialization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _encode_tree(value: Any, arrays: dict[str, np.ndarray], counter: list[int]):
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                [str(key), _encode_tree(item, arrays, counter)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_tree(item, arrays, counter) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_tree(item, arrays, counter) for item in value],
        }
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    array = np.asarray(jax.device_get(value))
    key = f"array_{counter[0]:05d}"
    counter[0] += 1
    arrays[key] = array
    return {
        "kind": "array",
        "key": key,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def save_pytree(path: str | Path, tree: Any) -> dict[str, Any]:
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    structure = _encode_tree(tree, arrays, [0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "format": "daehn_pytree_npz_v1",
        "file": path.name,
        "structure": structure,
        "array_count": len(arrays),
    }


def _decode_tree(spec: dict[str, Any], arrays) -> Any:
    kind = spec["kind"]
    if kind == "dict":
        return {key: _decode_tree(item, arrays) for key, item in spec["items"]}
    if kind == "tuple":
        return tuple(_decode_tree(item, arrays) for item in spec["items"])
    if kind == "list":
        return [_decode_tree(item, arrays) for item in spec["items"]]
    if kind == "none":
        return None
    if kind == "scalar":
        return spec["value"]
    if kind == "array":
        value = np.asarray(arrays[spec["key"]])
        if tuple(value.shape) != tuple(spec["shape"]):
            raise ValueError(f"array shape mismatch for {spec['key']}")
        return jnp.asarray(value)
    raise ValueError(f"unsupported PyTree node kind: {kind}")


def load_pytree(path: str | Path, manifest: dict[str, Any]) -> Any:
    if manifest.get("format") != "daehn_pytree_npz_v1":
        raise ValueError("unsupported weights format")
    with np.load(Path(path), allow_pickle=False) as arrays:
        return _decode_tree(manifest["structure"], arrays)
