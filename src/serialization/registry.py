"""Small explicit registry for custom serializable factories."""
from __future__ import annotations

from typing import Callable

_FACTORIES: dict[str, Callable] = {}


def register(name: str):
    name = str(name)
    if not name:
        raise ValueError("registry name must be nonempty")

    def decorator(factory: Callable):
        if name in _FACTORIES and _FACTORIES[name] is not factory:
            raise ValueError(f"factory {name!r} is already registered")
        _FACTORIES[name] = factory
        return factory

    return decorator


def resolve(name: str) -> Callable:
    try:
        return _FACTORIES[str(name)]
    except KeyError as exc:
        raise KeyError(
            f"no DAEHN serialization factory is registered as {name!r}"
        ) from exc
