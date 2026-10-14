import jax
import jax.numpy as jnp
import numpy as np
import pytest
from daehn.dae import Coordinate, DifferentialState, AlgebraicState, DerivativeSpec, EquationBlock, DAESystem, TaylorNeighborhood, compile_dae


def make_problem():
    x = DifferentialState("x")
    system = DAESystem(coordinates=(Coordinate("t"),), differential_states=(x,),
        algebraic_states=(AlgebraicState("z"),), derivatives=(DerivativeSpec(x, (1,)),),
        differential_equations=EquationBlock(lambda c, s, v, p: jnp.array([v[0]+s[0]]), size=1, affine_in_projection_variables=True),
        algebraic_equalities=EquationBlock(lambda c, s, v, p: jnp.array([s[1]-s[0]]), size=1, affine_in_projection_variables=True), algebraic_inequalities=EquationBlock(affine_in_projection_variables=True), name="decay")
    return compile_dae(system, discretization=TaylorNeighborhood(deltas=(.01,), order=1))


def test_compiled_differential_and_algebraic_residuals():
    compiled = make_problem()
    context = compiled.pack_context(jnp.array([[0.]]), jnp.empty((1, 0)), jnp.array([[1.]]), jnp.array([[[np.exp(-.01)]]]))
    projection = compiled.prepare_projection(batch_size=1, method="linear")
    result = jax.jit(projection)(jnp.zeros((1, compiled.variable_size)), context)
    residual = jax.vmap(compiled.raw_equalities)(context, result)
    np.testing.assert_allclose(residual, 0., atol=1e-10)
    decoded = compiled.decode(context, result)
    np.testing.assert_allclose(decoded["derivatives"][..., 0], -decoded["states"][..., 0], atol=1e-10)
    np.testing.assert_allclose(decoded["states"][..., 1], decoded["states"][..., 0], atol=1e-10)


def test_invalid_system_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        DAESystem(coordinates=(Coordinate("x"),), differential_states=(DifferentialState("x"),))
