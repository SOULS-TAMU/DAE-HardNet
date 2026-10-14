"""Serialization for structured optimization problems."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from ..problems import ParametricQP


def _save_matrix(arrays: dict[str, np.ndarray], name: str, value) -> dict[str, Any]:
    if sp is not None and sp.issparse(value):
        csr = value.tocsr()
        arrays[f"{name}_data"] = np.asarray(csr.data)
        arrays[f"{name}_indices"] = np.asarray(csr.indices, dtype=np.int64)
        arrays[f"{name}_indptr"] = np.asarray(csr.indptr, dtype=np.int64)
        return {
            "kind": "csr",
            "shape": list(csr.shape),
            "data": f"{name}_data",
            "indices": f"{name}_indices",
            "indptr": f"{name}_indptr",
        }
    array = np.asarray(value)
    arrays[name] = array
    return {"kind": "dense", "shape": list(array.shape), "key": name}


def _load_matrix(spec: dict[str, Any], arrays):
    if spec["kind"] == "dense":
        value = np.asarray(arrays[spec["key"]])
        if tuple(value.shape) != tuple(spec["shape"]):
            raise ValueError("stored dense matrix shape mismatch")
        return value
    if spec["kind"] == "csr":
        if sp is None:
            raise RuntimeError("SciPy is required to load sparse problems")
        return sp.csr_matrix(
            (
                np.asarray(arrays[spec["data"]]),
                np.asarray(arrays[spec["indices"]], dtype=np.int64),
                np.asarray(arrays[spec["indptr"]], dtype=np.int64),
            ),
            shape=tuple(spec["shape"]),
        )
    raise ValueError(f"unsupported matrix kind {spec['kind']}")


def save_problem(path: str | Path, problem) -> dict[str, Any]:
    if not isinstance(problem, ParametricQP):
        raise TypeError(
            "portable problem serialization currently supports ParametricQP; "
            "custom functional problems should use a registered factory"
        )
    arrays: dict[str, np.ndarray] = {}
    fields = {}
    for name in (
        "P",
        "q0",
        "q_parameter_map",
        "A_eq",
        "b_eq0",
        "b_eq_parameter_map",
        "A_ineq",
        "b_ineq0",
        "b_ineq_parameter_map",
    ):
        fields[name] = _save_matrix(arrays, name, getattr(problem, name))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "format": "daehn_problem_v1",
        "type": "ParametricQP",
        "file": path.name,
        "name": problem.name,
        "fields": fields,
    }


def load_problem(path: str | Path, manifest: dict[str, Any]):
    if manifest.get("format") != "daehn_problem_v1":
        raise ValueError("unsupported problem format")
    if manifest.get("type") != "ParametricQP":
        raise ValueError("unsupported structured problem type")
    with np.load(Path(path), allow_pickle=False) as arrays:
        values = {
            name: _load_matrix(spec, arrays)
            for name, spec in manifest["fields"].items()
        }
    return ParametricQP(name=manifest.get("name", "parametric_qp"), **values)
