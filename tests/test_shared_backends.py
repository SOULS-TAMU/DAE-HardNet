import jax
import jax.numpy as jnp
import numpy as np
import pytest
from daehn.problems import FunctionalProblem
from daehn.opt.api import prepare_projection


@pytest.mark.integration
@pytest.mark.parametrize("method", ["fischer-burmeister-newton", "fischer-burmeister-reduced", "primal-dual-interior-point"])
def test_shared_backend_analytical_equality(method):
    problem = FunctionalProblem(parameter_size=0, variable_size=2,
        objective=lambda p, y: jnp.asarray(0., dtype=y.dtype),
        equalities=lambda p, y: jnp.array([y.sum()-1.]), equality_size=1)
    layer = prepare_projection(problem, method=method, batch_size=1)
    target = jnp.array([[2., 0.]])
    parameters = jnp.empty((1, 0))
    value, info = layer.with_info(target, parameters)
    np.testing.assert_allclose(value, [[1.5, -.5]], atol=1e-7)
    assert np.all(np.isfinite(value))
    gradient = jax.jacrev(lambda v: layer(v, parameters))(target)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(np.asarray(gradient).reshape(2, 2), [[.5, -.5], [-.5, .5]], atol=1e-6)
