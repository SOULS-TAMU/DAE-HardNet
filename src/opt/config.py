"""Static solver configurations used before JAX tracing."""
from __future__ import annotations
from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class LinearSolverConfig:
    tolerance: float = 1e-9
    atol: float = 0.0
    restart: int = 30
    max_iterations: int = 50
    use_preconditioner: bool = True
    mode: str = 'auto'
    def __post_init__(self):
        if self.tolerance <= 0 or self.atol < 0: raise ValueError('linear tolerances are invalid')
        if self.restart < 1 or self.max_iterations < 1: raise ValueError('linear iteration limits must be positive')

@dataclass(frozen=True)
class ImplicitGradientConfig:
    policy: str = 'residual_checked'
    inexact_tolerance: float = 1e-6
    adjoint_tolerance: float = 1e-8
    invalid_batch_policy: str = 'skip'
    differentiate_problem_parameters: bool = False
    def __post_init__(self):
        if self.policy not in {'strict','residual_checked','always'}: raise ValueError('invalid gradient policy')
        if self.invalid_batch_policy not in {'skip','use_valid'}: raise ValueError('invalid batch policy')
        if self.inexact_tolerance < 0 or self.adjoint_tolerance < 0: raise ValueError('gradient tolerances must be nonnegative')

@dataclass(frozen=True)
class ProjectionConfig:
    tolerance: float = 1e-9
    max_iterations: int = 100
    mu: float = 1e-4
    backtracking: bool = False
    max_backtracking_steps: int = 8
    backtracking_factor: float = 0.5
    interior_margin: float = 1.0
    linear_solver: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    implicit_gradient: ImplicitGradientConfig = field(default_factory=ImplicitGradientConfig)
    store_residual_history: bool = False
    transpose_mode: str = 'auto'
    def __post_init__(self):
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0: raise ValueError('max_iterations must be a nonnegative integer')
        if not math.isfinite(self.tolerance) or self.tolerance <= 0: raise ValueError('tolerance must be positive and finite')
        if self.mu < 0: raise ValueError('mu must be nonnegative')
        if self.max_backtracking_steps < 0: raise ValueError('max_backtracking_steps must be nonnegative')
        if not 0 < self.backtracking_factor < 1: raise ValueError('backtracking_factor must be in (0,1)')
