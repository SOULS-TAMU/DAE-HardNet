"""Compatibility adapter for JAX portable-export APIs.

JAX moved portable export from ``jax.experimental.export`` to ``jax.export``
in version 0.4.30.  DAEHN supports both APIs so model artifacts can be created
and consumed with older 0.4.x installations as well as current JAX releases.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any, Callable, Sequence

import jax


@dataclass(frozen=True)
class JaxExportApi:
    """Resolved JAX export API and its calling convention."""

    module: ModuleType
    name: str
    modern: bool


def resolve_export_api() -> JaxExportApi:
    """Resolve the installed JAX portable-export implementation.

    Prefer the stable ``jax.export`` module.  On JAX versions before 0.4.30,
    fall back to ``jax.experimental.export``.  Importing by module path is
    intentional: some JAX releases do not populate ``jax.export`` as a package
    attribute until the submodule is imported.
    """

    try:
        module = importlib.import_module("jax.export")
        return JaxExportApi(module=module, name="jax.export", modern=True)
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        module = importlib.import_module("jax.experimental.export")
        return JaxExportApi(
            module=module,
            name="jax.experimental.export",
            modern=False,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "portable export is unavailable in this JAX installation; "
            "install a JAX release that provides either 'jax.export' or "
            "'jax.experimental.export'"
        ) from exc


def export_and_serialize(
    function: Callable,
    *argument_specs: Any,
    platforms: Sequence[str] | None = None,
) -> tuple[Any, bytes, JaxExportApi]:
    """Export and serialize a JAX function with the installed API."""

    api = resolve_export_api()
    jitted = jax.jit(function)
    if api.modern:
        exported = api.module.export(
            jitted,
            platforms=None if platforms is None else tuple(platforms),
        )(*argument_specs)
        serialized = exported.serialize()
    else:
        exported = api.module.export(
            jitted,
            lowering_platforms=None if platforms is None else tuple(platforms),
        )(*argument_specs)
        serialized = api.module.serialize(exported)
    return exported, bytes(serialized), api


def deserialize_exported(serialized: bytes | bytearray) -> tuple[Any, Callable, JaxExportApi]:
    """Deserialize a portable program and return its callable entry point."""

    api = resolve_export_api()
    exported = api.module.deserialize(bytearray(serialized))
    if api.modern:
        call = exported.call
    elif hasattr(api.module, "call_exported"):
        call = api.module.call_exported(exported)
    elif hasattr(api.module, "call"):
        # Compatibility with early experimental-export releases.
        call = api.module.call(exported)
    else:  # pragma: no cover - defensive guard for unsupported JAX builds
        raise RuntimeError(
            f"{api.name} can deserialize artifacts but exposes no call API"
        )
    return exported, call, api
