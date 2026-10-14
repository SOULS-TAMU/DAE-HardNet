"""Numerical properties independent of solver implementation details."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from daehn.opt.linear_equality import LinearEqualityProjection


def test_randomized_feasibility_idempotence_and_orthogonal_correction():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(2, 5))
    rhs = rng.normal(size=2)
    raw = rng.normal(size=(16, 5))
    layer = LinearEqualityProjection(A=matrix, b=rhs).prepare(batch_size=16)
    result = np.asarray(jax.jit(layer)(jnp.asarray(raw)))
    np.testing.assert_allclose(result @ matrix.T, np.broadcast_to(rhs, (16, 2)), atol=1e-11)
    np.testing.assert_allclose(layer(result), result, atol=1e-11)
    # Corrections must be normal to the null space of the constraints.
    _, _, vectors = np.linalg.svd(matrix, full_matrices=True)
    np.testing.assert_allclose((raw-result) @ vectors[2:].T, 0., atol=1e-11)


def test_dense_and_sparse_definitions_agree():
    matrix = np.array([[1., 2., -1.], [0., 1., 1.]])
    values = jnp.array([[2., -.5, .25], [-1., 3., 2.]])
    dense = LinearEqualityProjection(A=matrix, b=np.array([1., 0.])).prepare(batch_size=2)
    sparse = LinearEqualityProjection(A=csr_matrix(matrix), b=np.array([1., 0.])).prepare(batch_size=2)
    np.testing.assert_allclose(dense(values), sparse(values), atol=1e-12)


def test_parameter_jacobian_matches_analytical_solution():
    layer = LinearEqualityProjection(A=np.array([[1., 1.]]), b_parameter_map=np.array([[2.]])).prepare(batch_size=1)
    raw = jnp.array([.7, -.3])
    fn = lambda p: layer(raw, p.reshape(1, 1))
    np.testing.assert_allclose(jax.jacfwd(fn)(jnp.array([.4])), np.ones((2, 1)), atol=1e-12)
    np.testing.assert_allclose(jax.jacrev(fn)(jnp.array([.4])), np.ones((2, 1)), atol=1e-12)


def test_float32_feasibility_uses_precision_appropriate_tolerance():
    layer = LinearEqualityProjection(A=np.array([[1., 1.]]), b=np.array([1.])).prepare(batch_size=1, dtype=jnp.float32)
    result = layer(jnp.array([2., .1], dtype=jnp.float32))
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(result.sum(), 1., atol=5e-7)


@pytest.mark.parametrize("case", ["rank", "batch", "regularization", "rhs", "parameter-map"])
def test_invalid_preparation_is_rejected(case):
    kwargs = {"A": np.array([[1., 1.]]), "b": np.array([1.])}
    batch = 1
    if case == "rank":
        kwargs = {"A": np.array([[1., 1.], [2., 2.]]), "b": np.array([1., 2.])}
    elif case == "batch":
        batch = 0
    elif case == "regularization":
        kwargs["regularization"] = -1.
    elif case == "rhs":
        kwargs["b"] = np.array([1., 2.])
    else:
        kwargs["b_parameter_map"] = np.ones((2, 1))
    with pytest.raises(ValueError):
        LinearEqualityProjection(**kwargs).prepare(batch_size=batch)
