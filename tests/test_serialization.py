import jax
import jax.numpy as jnp
import numpy as np
import pytest
from daehn.serialization.weights import save_pytree, load_pytree


def test_npz_roundtrip_preserves_predictions(tmp_path):
    tree = {"weights": (jnp.array([[2., 3.]]), jnp.array([.25])), "label": "linear", "optional": None}
    manifest = save_pytree(tmp_path/"weights.npz", tree)
    restored = load_pytree(tmp_path/"weights.npz", manifest)
    value = jnp.array([.5, -.2])
    def predict(p):
        return p["weights"][0] @ value + p["weights"][1]
    np.testing.assert_array_equal(predict(tree), predict(restored))
    assert jax.tree_util.tree_structure(tree) == jax.tree_util.tree_structure(restored)
    with np.load(tmp_path/"weights.npz", allow_pickle=False) as archive:
        assert all(archive[k].dtype != object for k in archive.files)
    with pytest.raises(ValueError, match="unsupported"):
        load_pytree(tmp_path/"weights.npz", {"format": "invalid"})
