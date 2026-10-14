import jax
import jax.numpy as jnp
import numpy as np
import pytest
from daehn.opt.linear_equality import LinearEqualityProjection
from daehn.opt.layers.projection import LinearEqualityProjection as LegacyProjection


def test_analytical_projection_jit_and_gradient():
    layer = LinearEqualityProjection(A=np.array([[1., 1.]]), b=np.array([1.])).prepare(batch_size=1)
    value = jnp.array([2., 0.])
    np.testing.assert_allclose(jax.jit(layer)(value), [1.5, -.5], atol=1e-12)
    expected = np.array([[.5, -.5], [-.5, .5]])
    np.testing.assert_allclose(jax.jacfwd(layer)(value), expected, atol=1e-12)
    eps = 1e-5
    finite = np.column_stack([(np.asarray(layer(value+jnp.eye(2)[i]*eps))-np.asarray(layer(value-jnp.eye(2)[i]*eps)))/(2*eps) for i in range(2)])
    np.testing.assert_allclose(finite, expected, atol=1e-10)
    assert LegacyProjection is LinearEqualityProjection


def test_batch_parameter_rhs_and_diagnostics():
    layer = LinearEqualityProjection(A=np.array([[1., 1.]]), b0=np.zeros(1), b_parameter_map=np.ones((1, 1))).prepare(batch_size=2)
    result, info = layer.with_info(jnp.array([[2., 0.], [0., 0.]]), jnp.array([[1.], [2.]]))
    np.testing.assert_allclose(result, [[1.5, -.5], [1., 1.]], atol=1e-12)
    assert np.all(info.forward_state_finite)
    assert np.max(info.forward_residual_inf_norm) < 1e-12
    with pytest.raises(ValueError, match="batch size"):
        layer(jnp.array([2., 0.]), jnp.array([[1.]]))
