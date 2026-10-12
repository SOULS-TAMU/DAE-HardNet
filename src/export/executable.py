"""Environment-specific serialized JAX executables."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import serialize_executable

from ..serialization.predictor import Predictor
from ..runtime._jax_export import deserialize_exported


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _dtype(name: str):
    return jnp.float64 if str(name) == "float64" else jnp.float32


def _environment_manifest(*, batch_size, input_size, output_size, dtype):
    try:
        import jaxlib

        jaxlib_version = jaxlib.__version__
    except Exception:  # pragma: no cover
        jaxlib_version = "unknown"
    devices = jax.devices()
    device_kind = devices[0].device_kind if devices else "unknown"
    platform_name = devices[0].platform if devices else "unknown"
    return {
        "format": "daehn_native_predictor_v1",
        "os": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "backend": platform_name,
        "device_kind": device_kind,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib_version,
        "batch_size": int(batch_size),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "dtype": np.dtype(dtype).name,
        "signature": "single_array_to_array",
    }


def compile_portable_to_native(
    portable_path: str | Path,
    native_path: str | Path,
) -> dict:
    portable_path = Path(portable_path)
    with portable_path.with_suffix(portable_path.suffix + ".json").open(
        "r", encoding="utf-8"
    ) as handle:
        portable_manifest = json.load(handle)
    _exported, exported_call, export_api = deserialize_exported(
        portable_path.read_bytes()
    )
    dtype = _dtype(portable_manifest["dtype"])
    example = jax.ShapeDtypeStruct(
        (
            int(portable_manifest["batch_size"]),
            int(portable_manifest["input_size"]),
        ),
        dtype,
    )

    def call_exported(inputs):
        return exported_call(inputs)

    compiled = jax.jit(call_exported).lower(example).compile()
    serialized, _in_tree, _out_tree = serialize_executable.serialize(compiled)
    native_path = Path(native_path)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    native_path.write_bytes(serialized)
    manifest = _environment_manifest(
        batch_size=portable_manifest["batch_size"],
        input_size=portable_manifest["input_size"],
        output_size=portable_manifest["output_size"],
        dtype=dtype,
    )
    manifest["file"] = native_path.name
    manifest["sha256"] = _sha256(native_path)
    manifest["portable_source"] = portable_path.name
    manifest["portable_export_api"] = export_api.name
    with native_path.with_suffix(native_path.suffix + ".json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def _check_compatibility(manifest: dict):
    current = _environment_manifest(
        batch_size=manifest["batch_size"],
        input_size=manifest["input_size"],
        output_size=manifest["output_size"],
        dtype=manifest["dtype"],
    )
    required_equal = (
        "os",
        "architecture",
        "backend",
        "device_kind",
        "jax_version",
        "jaxlib_version",
    )
    differences = {
        key: (manifest.get(key), current.get(key))
        for key in required_equal
        if manifest.get(key) != current.get(key)
    }
    if differences:
        raise RuntimeError(
            "native executable is incompatible with this environment: "
            + ", ".join(
                f"{key}={expected!r} (current {actual!r})"
                for key, (expected, actual) in differences.items()
            )
        )


def load_native_predictor(path: str | Path) -> Predictor:
    path = Path(path)
    with path.with_suffix(path.suffix + ".json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "daehn_native_predictor_v1":
        raise ValueError("unsupported native predictor manifest")
    _check_compatibility(manifest)
    if _sha256(path) != manifest["sha256"]:
        raise RuntimeError("native executable checksum mismatch")
    in_tree = jax.tree_util.tree_structure(((0,), {}))
    out_tree = jax.tree_util.tree_structure(0)
    compiled = serialize_executable.deserialize_and_load(
        path.read_bytes(), in_tree, out_tree
    )
    dtype = _dtype(manifest["dtype"])
    return Predictor(
        predict_fn=compiled,
        batch_size=int(manifest["batch_size"]),
        input_size=int(manifest["input_size"]),
        output_size=int(manifest["output_size"]),
        dtype=dtype,
        backend=manifest["backend"],
        source="native",
        compile_on_load=False,
    )
