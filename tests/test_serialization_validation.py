"""Malformed artifact checks without executing pickle payloads."""
import copy
import jax.numpy as jnp
import numpy as np
import pytest
from daehn.serialization.weights import save_pytree, load_pytree


def test_corrupted_array_shape_is_rejected(tmp_path):
    path = tmp_path/"weights.npz"
    manifest = save_pytree(path, {"weights": jnp.ones((2, 3))})
    np.savez(path, array_00000=np.ones((3, 2)))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_pytree(path, manifest)


def test_unknown_structure_node_is_rejected(tmp_path):
    path = tmp_path/"weights.npz"
    manifest = save_pytree(path, jnp.ones(2))
    broken = copy.deepcopy(manifest)
    broken["structure"]["kind"] = "invalid"
    with pytest.raises(ValueError, match="unsupported PyTree"):
        load_pytree(path, broken)
