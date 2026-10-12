"""Save, load, compile, and run fixed-shape DAEHN predictors."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..opt import ImplicitGradientConfig, LinearSolverConfig, ProjectionConfig
from ..runtime._jax_export import export_and_serialize
from ..opt.layers import (
    BoxProjection,
    FischerBurmeisterProjection,
    LinearEqualityProjection,
    SparseProjection,
)
from .problem import load_problem, save_problem
from .weights import load_pytree, save_pytree


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _activation(name: str):
    normalized = str(name).lower()
    choices = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,
        "sigmoid": jax.nn.sigmoid,
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
    }
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported serialized activation {name!r}") from exc


def make_network_apply(spec: dict[str, Any]) -> Callable:
    network_type = spec.get("type", "mlp")
    if network_type != "mlp":
        raise ValueError(f"unsupported serialized network type {network_type!r}")
    activation = _activation(spec.get("activation", "relu"))
    parameter_format = spec.get("parameter_format", "auto")

    parameter_path = tuple(spec.get("parameter_path", ()))

    def apply(params, inputs):
        values = inputs
        selected = params
        if parameter_path:
            for key in parameter_path:
                selected = selected[key]
        elif isinstance(selected, dict) and "backbone" in selected:
            # ProjectedNeuralModel stores trainable parameters under a
            # ``backbone`` key. Keep direct layer tuples supported for the
            # low-level serialization API while transparently accepting the
            # model-owned PyTree.
            selected = selected["backbone"]
        layers = tuple(selected)
        for index, layer in enumerate(layers):
            if parameter_format == "legacy" or "W" in layer:
                values = values @ layer["W"] + layer["b"]
            else:
                values = values @ layer["kernel"]
                if "bias" in layer:
                    values = values + layer["bias"]
            if index + 1 < len(layers):
                values = activation(values)
        return values

    return apply


def _projection_config_from_dict(value: dict[str, Any]) -> ProjectionConfig:
    linear = LinearSolverConfig(**value.get("linear_solver", {}))
    implicit = ImplicitGradientConfig(**value.get("implicit_gradient", {}))
    fields = {
        key: item
        for key, item in value.items()
        if key not in {"linear_solver", "implicit_gradient"}
    }
    return ProjectionConfig(
        **fields, linear_solver=linear, implicit_gradient=implicit
    )


def _save_projection_arrays(path: Path, projection_spec: dict[str, Any]):
    arrays = {}
    serializable = dict(projection_spec)
    for name in (
        "A",
        "b",
        "b0",
        "b_parameter_map",
        "lower",
        "upper",
    ):
        if name in serializable and serializable[name] is not None:
            key = f"projection_{name}"
            arrays[key] = np.asarray(serializable.pop(name))
            serializable[f"{name}_key"] = key
    if arrays:
        np.savez(path, **arrays)
        serializable["arrays_file"] = path.name
    return serializable


def _load_projection_arrays(root: Path, spec: dict[str, Any]):
    restored = dict(spec)
    file = restored.pop("arrays_file", None)
    if file is None:
        return restored
    with np.load(root / file, allow_pickle=False) as arrays:
        for name in (
            "A",
            "b",
            "b0",
            "b_parameter_map",
            "lower",
            "upper",
        ):
            key = restored.pop(f"{name}_key", None)
            if key is not None:
                restored[name] = np.asarray(arrays[key])
    return restored


def _prepare_projection(
    spec: dict[str, Any],
    *,
    problem,
    root: Path,
    batch_size: int,
    dtype,
    backend: str,
):
    spec = _load_projection_arrays(root, spec)
    kind = spec.get("type", "none")
    if kind == "none":
        return None
    if kind == "box":
        return BoxProjection(spec["lower"], spec["upper"]).prepare()
    if kind == "linear_equality":
        return LinearEqualityProjection(
            A=spec["A"],
            b=spec.get("b"),
            b0=spec.get("b0"),
            b_parameter_map=spec.get("b_parameter_map"),
            regularization=float(spec.get("regularization", 0.0)),
        ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
    if problem is None:
        raise ValueError(f"projection {kind!r} requires a serialized problem")
    if kind == "sparse":
        config = _projection_config_from_dict(spec.get("config", {}))
        return SparseProjection(problem, config=config).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
    if kind == "fischer_burmeister":
        kwargs = {
            key: value for key, value in spec.items() if key != "type"
        }
        return FischerBurmeisterProjection(problem, **kwargs).prepare(
            batch_size=batch_size, dtype=dtype, backend=backend
        )
    raise ValueError(f"unsupported serialized projection type {kind!r}")


class Predictor:
    """A fixed-shape predictor whose steady-state calls use one compiled graph."""

    def __init__(
        self,
        *,
        predict_fn: Callable,
        batch_size: int,
        input_size: int,
        output_size: int,
        dtype,
        backend: str,
        source: str,
        compile_on_load: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.dtype = dtype
        self.backend = str(backend)
        self.source = str(source)
        self._predict_fn = predict_fn
        example = jax.ShapeDtypeStruct(
            (self.batch_size, self.input_size), self.dtype
        )
        if compile_on_load:
            self._compiled = jax.jit(predict_fn).lower(example).compile()
        else:
            self._compiled = predict_fn

    def predict_jax(self, inputs):
        inputs = jnp.asarray(inputs, dtype=self.dtype)
        if inputs.shape != (self.batch_size, self.input_size):
            raise ValueError(
                f"predict_jax requires shape "
                f"{(self.batch_size, self.input_size)}, got {inputs.shape}"
            )
        return self._compiled(inputs)

    def __call__(self, inputs):
        host = np.asarray(inputs)
        squeeze = host.ndim == 1
        if squeeze:
            host = host[None, :]
        if host.ndim != 2 or host.shape[1] != self.input_size:
            raise ValueError(
                f"inputs must have shape (N, {self.input_size}), got {host.shape}"
            )
        outputs = []
        for start in range(0, host.shape[0], self.batch_size):
            chunk = host[start : start + self.batch_size]
            count = chunk.shape[0]
            if count < self.batch_size:
                padded = np.zeros(
                    (self.batch_size, self.input_size), dtype=np.dtype(self.dtype)
                )
                padded[:count] = chunk
                chunk = padded
            result = self.predict_jax(jax.device_put(jnp.asarray(chunk)))
            outputs.append(np.asarray(jax.device_get(result[:count])))
        merged = np.concatenate(outputs, axis=0)
        return merged[0] if squeeze else merged

    def export_portable(
        self,
        path: str | Path,
        *,
        platforms: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        try:
            import flatbuffers  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "portable export requires the optional 'flatbuffers' package"
            ) from exc
        example = jax.ShapeDtypeStruct(
            (self.batch_size, self.input_size), self.dtype
        )
        _exported, serialized, export_api = export_and_serialize(
            self._predict_fn,
            example,
            platforms=platforms,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialized)
        manifest = {
            "format": "daehn_portable_predictor_v1",
            "file": path.name,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dtype": np.dtype(self.dtype).name,
            "platforms": list(platforms or []),
            "jax_version": jax.__version__,
            "export_api": export_api.name,
            "sha256": _sha256(path),
        }
        with (path.with_suffix(path.suffix + ".json")).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest

    def export_native(
        self,
        portable_path: str | Path,
        native_path: str | Path,
    ) -> dict[str, Any]:
        from ..export.executable import compile_portable_to_native

        return compile_portable_to_native(portable_path, native_path)


def _dtype(name):
    normalized = str(name).lower()
    if normalized in {"float64", "fp64", "64"}:
        return jnp.float64
    if normalized in {"float32", "fp32", "32"}:
        return jnp.float32
    raise ValueError(f"unsupported dtype {name!r}")


def save_predictor(
    directory: str | Path,
    *,
    network_spec: dict[str, Any],
    params,
    input_size: int,
    output_size: int,
    batch_size: int,
    projection_spec: dict[str, Any] | None = None,
    problem=None,
    input_transform: dict[str, Any] | None = None,
    dtype: str = "float64",
    backend: str = "cpu",
    metadata: dict[str, Any] | None = None,
    predictor_factory: str | None = None,
    factory_config: dict[str, Any] | None = None,
    export_portable: bool = False,
    export_native: bool = False,
    export_platforms: list[str] | tuple[str, ...] | None = None,
    strict_export: bool = False,
) -> Predictor:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    weights_manifest = save_pytree(root / "weights.npz", params)
    problem_manifest = (
        None if problem is None else save_problem(root / "problem_arrays.npz", problem)
    )
    projection_spec = _save_projection_arrays(
        root / "projection_arrays.npz", projection_spec or {"type": "none"}
    )
    transform = dict(input_transform or {"type": "identity"})
    transform_arrays = {}
    for name in ("mean", "scale"):
        if name in transform:
            key = f"transform_{name}"
            transform_arrays[key] = np.asarray(transform.pop(name))
            transform[f"{name}_key"] = key
    if transform_arrays:
        np.savez(root / "normalization.npz", **transform_arrays)
        transform["arrays_file"] = "normalization.npz"
    manifest = {
        "format": "daehn_predictor_v1",
        "network": _json_safe(network_spec),
        "weights": weights_manifest,
        "problem": problem_manifest,
        "projection": _json_safe(projection_spec),
        "input_transform": _json_safe(transform),
        "input_size": int(input_size),
        "output_size": int(output_size),
        "batch_size": int(batch_size),
        "dtype": str(dtype),
        "backend": str(backend),
        "metadata": _json_safe(metadata or {}),
        "predictor_factory": predictor_factory,
        "factory_config": _json_safe(factory_config or {}),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
    }
    with (root / "model.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    predictor = load_predictor(root / "model.json", backend=backend)
    if export_portable or export_native:
        export_errors: dict[str, str] = {}
        portable = root / f"predictor_b{batch_size}.daehn_export"
        portable_created = False
        try:
            predictor.export_portable(portable, platforms=export_platforms)
            manifest["portable_export"] = portable.name
            portable_created = True
        except Exception as exc:
            if strict_export:
                raise
            export_errors["portable"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "portable predictor export failed; JSON/NPZ model saving "
                f"completed successfully: {export_errors['portable']}",
                RuntimeWarning,
                stacklevel=2,
            )
        if export_native and portable_created:
            native = root / f"predictor_b{batch_size}.executable"
            try:
                predictor.export_native(portable, native)
                manifest["native_executable"] = native.name
            except Exception as exc:
                if strict_export:
                    raise
                export_errors["native"] = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    "native predictor export is unavailable in this "
                    f"environment; portable and JSON artifacts remain usable: "
                    f"{export_errors['native']}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif export_native and not portable_created:
            export_errors["native"] = (
                "skipped because the portable export was not created"
            )
        if export_errors:
            manifest["export_errors"] = export_errors
        with (root / "model.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return predictor


def _load_transform(root: Path, spec: dict[str, Any]):
    restored = dict(spec)
    file = restored.pop("arrays_file", None)
    if file is not None:
        with np.load(root / file, allow_pickle=False) as arrays:
            for name in ("mean", "scale"):
                key = restored.pop(f"{name}_key", None)
                if key is not None:
                    restored[name] = jnp.asarray(arrays[key])
    kind = restored.get("type", "identity")
    if kind == "identity":
        return lambda x: x
    if kind == "standard":
        mean = jnp.asarray(restored["mean"])
        scale = jnp.asarray(restored["scale"])
        return lambda x: (x - mean) / scale
    raise ValueError(f"unsupported input transform {kind!r}")


def _load_factory(path: str):
    if ":" not in path:
        from .registry import resolve
        return resolve(path)
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"serialized predictor factory {path!r} is not callable")
    return factory


def load_predictor(
    model_json: str | Path,
    *,
    backend: str | None = None,
    preference: str = "json",
) -> Predictor:
    model_json = Path(model_json)
    root = model_json.parent
    with model_json.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "daehn_predictor_v1":
        raise ValueError("unsupported predictor manifest")
    selected_backend = str(backend or manifest.get("backend", "cpu"))
    if preference in {"fastest", "native"} and manifest.get("native_executable"):
        from ..export.executable import load_native_predictor

        native_path = root / manifest["native_executable"]
        try:
            return load_native_predictor(native_path)
        except Exception:
            if preference == "native":
                raise
    if preference in {"fastest", "portable"} and manifest.get("portable_export"):
        from ..export.stablehlo import load_portable_predictor

        try:
            return load_portable_predictor(root / manifest["portable_export"])
        except Exception:
            if preference == "portable":
                raise
    dtype = _dtype(manifest["dtype"])
    params = load_pytree(
        root / manifest["weights"]["file"], manifest["weights"]
    )
    problem = (
        None
        if manifest.get("problem") is None
        else load_problem(
            root / manifest["problem"]["file"], manifest["problem"]
        )
    )
    if manifest.get("predictor_factory"):
        factory = _load_factory(manifest["predictor_factory"])
        predict_fn = factory(
            params=params,
            manifest=manifest,
            root=root,
            backend=selected_backend,
            dtype=dtype,
        )
        return Predictor(
            predict_fn=predict_fn,
            batch_size=int(manifest["batch_size"]),
            input_size=int(manifest["input_size"]),
            output_size=int(manifest["output_size"]),
            dtype=dtype,
            backend=selected_backend,
            source="json_factory",
        )
    projection = _prepare_projection(
        manifest["projection"],
        problem=problem,
        root=root,
        batch_size=int(manifest["batch_size"]),
        dtype=dtype,
        backend=selected_backend,
    )
    network_apply = make_network_apply(manifest["network"])
    transform = _load_transform(root, manifest.get("input_transform", {}))

    def predict_fn(inputs):
        raw = network_apply(params, transform(inputs))
        if projection is None:
            return raw
        return projection(raw, inputs)

    return Predictor(
        predict_fn=predict_fn,
        batch_size=int(manifest["batch_size"]),
        input_size=int(manifest["input_size"]),
        output_size=int(manifest["output_size"]),
        dtype=dtype,
        backend=selected_backend,
        source="json",
    )
