"""Serializable reference problems used by the ED-column and pooling examples."""
from __future__ import annotations

from .problem import Problem


def pooling_projection_problem():
    target_names = tuple(f"y_hat_{i + 1}" for i in range(5))
    parameter_names = ("x1", "x2", "x3", "x4")
    variable_names = ("y1", "y2", "y3", "y4", "y5")
    problem = Problem(name="pooling_projected_neural")
    p = problem.add_parameter(target_names + parameter_names)
    y = problem.add_variable(variable_names)
    problem.set_projection_objective(
        variables=y,
        target=[getattr(p, name) for name in target_names],
    )
    problem.constraints.add(
        y.y2 + y.y3 - p.x1 - p.x2 == 0,
        p.x3 - y.y2 - y.y4 == 0,
        p.x4 - y.y3 - y.y5 == 0,
        y.y1 * y.y2 + y.y1 * y.y3 - 3.0 * p.x1 - p.x2 == 0,
        y.y1 * y.y2 + 2.0 * y.y4 - 2.5 * p.x3 <= 0,
        y.y1 * y.y3 + 2.0 * y.y5 - 1.5 * p.x4 <= 0,
    )
    return problem


def ed_column_projection_problem():
    target_names = tuple(f"y_hat_{i + 1}" for i in range(9))
    parameter_names = ("x1", "x2", "x3")
    variable_names = tuple(f"y{i}" for i in range(1, 10))
    problem = Problem(name="ed_column_projected_neural")
    p = problem.add_parameter(target_names + parameter_names)
    y = problem.add_variable(variable_names)
    problem.set_projection_objective(
        variables=y,
        target=[getattr(p, name) for name in target_names],
    )
    problem.constraints.add(
        p.x1 + p.x2 - y.y1 - y.y2 == 0,
        p.x1 * 0.697616946 - y.y1 * y.y3 - y.y2 * y.y6 == 0,
        p.x1 * 0.302383054 - y.y1 * y.y4 - y.y2 * y.y7 == 0,
        y.y3 + y.y4 + y.y5 - 1.0 == 0,
        y.y6 + y.y7 + y.y8 - 1.0 == 0,
        p.x3 * y.y1 - y.y9 == 0,
    )
    return problem
