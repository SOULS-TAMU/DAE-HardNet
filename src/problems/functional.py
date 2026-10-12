"""Function-defined JAX optimization problems."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable
import jax.numpy as jnp
from .problem import Expression, Constraint
from .canonical import CanonicalProblem
from .symbolic import SymbolicProblem


def _pattern(size: int, variable_size: int, supplied):
    if supplied is None:
        return tuple(tuple(range(variable_size)) for _ in range(size))
    rows = tuple(tuple(map(int, row)) for row in supplied)
    if len(rows) != size:
        raise ValueError('sparsity row count does not match constraint size')
    for row in rows:
        if any(i < 0 or i >= variable_size for i in row):
            raise ValueError('sparsity contains an invalid variable index')
    return rows

@dataclass(frozen=True)
class FunctionalProblem:
    parameter_size: int
    variable_size: int
    objective: Callable
    equalities: Callable = lambda theta, y: jnp.empty((0,), dtype=y.dtype)
    inequalities: Callable = lambda theta, y: jnp.empty((0,), dtype=y.dtype)
    equality_size: int = 0
    inequality_size: int = 0
    equality_sparsity: Any = None
    inequality_sparsity: Any = None
    name: str = 'functional_problem'

    def __post_init__(self):
        if self.parameter_size < 0 or self.variable_size < 1:
            raise ValueError('parameter_size must be nonnegative and variable_size positive')
        if self.equality_size < 0 or self.inequality_size < 0:
            raise ValueError('constraint sizes must be nonnegative')
        if not all(callable(f) for f in (self.objective, self.equalities, self.inequalities)):
            raise TypeError('objective and constraint functions must be callable')

    def canonicalize(self):
        return CanonicalProblem(
            parameter_size=self.parameter_size, variable_size=self.variable_size,
            equality_size=self.equality_size, inequality_size=self.inequality_size,
            objective=self.objective, equalities=self.equalities, inequalities=self.inequalities,
            equality_sparsity=_pattern(self.equality_size, self.variable_size, self.equality_sparsity),
            inequality_sparsity=_pattern(self.inequality_size, self.variable_size, self.inequality_sparsity),
            metadata={'name': self.name, 'kind': 'functional'},
        )

    def to_symbolic_problem(self):
        canonical = self.canonicalize()
        model = SymbolicProblem(self.name)
        p_names = [f'theta_{i}' for i in range(self.parameter_size)]
        y_names = [f'y_{i}' for i in range(self.variable_size)]
        if p_names: model.add_parameter(p_names)
        model.add_variable(y_names)
        all_params = frozenset(p_names)
        all_vars = frozenset(y_names)
        model.objective = Expression(
            lambda ctx, fn=self.objective: fn(ctx.parameters, ctx.variables),
            'functional objective', variable_dependencies=all_vars,
            parameter_dependencies=all_params, affine_in_variables=False,
        )
        eq_pattern = canonical.equality_sparsity
        ineq_pattern = canonical.inequality_sparsity
        for i in range(self.equality_size):
            deps = frozenset(y_names[j] for j in eq_pattern[i])
            residual = Expression(
                lambda ctx, i=i, fn=self.equalities: fn(ctx.parameters, ctx.variables)[i],
                f'equalities[{i}]', variable_dependencies=deps,
                parameter_dependencies=all_params, affine_in_variables=False,
            )
            model.constraints.add(Constraint(residual, 'eq', f'equalities[{i}] == 0'))
        for i in range(self.inequality_size):
            deps = frozenset(y_names[j] for j in ineq_pattern[i])
            residual = Expression(
                lambda ctx, i=i, fn=self.inequalities: fn(ctx.parameters, ctx.variables)[i],
                f'inequalities[{i}]', variable_dependencies=deps,
                parameter_dependencies=all_params, affine_in_variables=False,
            )
            model.constraints.add(Constraint(residual, 'ineq', f'inequalities[{i}] <= 0'))
        return model


    def as_projection_problem(self):
        canonical = self.canonicalize()
        model = SymbolicProblem(self.name + '_projection')
        target_names = [f'__target_{i}' for i in range(self.variable_size)]
        theta_names = [f'theta_{i}' for i in range(self.parameter_size)]
        p = model.add_parameter(target_names + theta_names)
        y = model.add_variable([f'y_{i}' for i in range(self.variable_size)])
        target_components = [getattr(p, name) for name in target_names]
        from .problem import VectorExpression
        model.set_projection_objective(
            variables=y,
            target=VectorExpression(target_components, 'projection target'),
        )
        all_params = frozenset(theta_names)
        y_names = [f'y_{i}' for i in range(self.variable_size)]
        eq_pattern = canonical.equality_sparsity
        ineq_pattern = canonical.inequality_sparsity
        offset = self.variable_size
        for i in range(self.equality_size):
            deps = frozenset(y_names[j] for j in eq_pattern[i])
            residual = Expression(
                lambda ctx, i=i, fn=self.equalities, offset=offset: fn(
                    ctx.parameters[offset:], ctx.variables
                )[i],
                f'equalities[{i}]', variable_dependencies=deps,
                parameter_dependencies=all_params, affine_in_variables=False,
            )
            model.constraints.add(Constraint(residual, 'eq', f'equalities[{i}] == 0'))
        for i in range(self.inequality_size):
            deps = frozenset(y_names[j] for j in ineq_pattern[i])
            residual = Expression(
                lambda ctx, i=i, fn=self.inequalities, offset=offset: fn(
                    ctx.parameters[offset:], ctx.variables
                )[i],
                f'inequalities[{i}]', variable_dependencies=deps,
                parameter_dependencies=all_params, affine_in_variables=False,
            )
            model.constraints.add(Constraint(residual, 'ineq', f'inequalities[{i}] <= 0'))
        return model.to_symbolic_problem()

    def solve(self, *, parameters=None, **kwargs):
        return self.to_symbolic_problem().solve(parameters=parameters, **kwargs)


def functional_problem(**metadata):
    def decorator(factory):
        definition = factory()
        if isinstance(definition, FunctionalProblem):
            return definition
        if not isinstance(definition, dict):
            raise TypeError('functional_problem factory must return FunctionalProblem or dict')
        return FunctionalProblem(**metadata, **definition)
    return decorator
