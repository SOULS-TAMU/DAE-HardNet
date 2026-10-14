"""Compiled continuous-to-algebraic DAE representation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from daehn.problems import FunctionalProblem
from daehn.opt import ProjectionConfig, ImplicitGradientConfig
from daehn.opt.layers import LinearEqualityProjection, SparseProjection, FischerBurmeisterProjection


@dataclass(frozen=True)
class ProjectionDiagnostics:
    selected_method: str
    condition_estimate: float
    equality_rows: int
    projection_variables: int


@dataclass(frozen=True)
class CompiledDAE:
    system: Any
    discretization: Any
    projection_layout: Any
    context_layout: Any
    residual_scales: Any
    variable_physical_scales: Any
    raw_equalities: Any
    raw_inequalities: Any
    scaled_equalities: Any
    scaled_inequalities: Any
    reconstruct: Any
    build_target: Any
    decode: Any
    linear_matrix: Any | None
    signature: str
    diagnostics: ProjectionDiagnostics

    @property
    def parameter_size(self) -> int:
        return int(self.context_layout.size)

    @property
    def variable_size(self) -> int:
        return int(self.projection_layout.size)

    @property
    def is_explicit_linear(self) -> bool:
        return self.linear_matrix is not None and self.system.algebraic_inequalities.size == 0

    def pack_context(self, coordinates, parameters, center_states, neighbor_states):
        return self.context_layout.pack(coordinates, parameters, center_states, neighbor_states)

    def physical_residuals(self, context, normalized_variables):
        return self.raw_equalities(context, normalized_variables), self.raw_inequalities(context, normalized_variables)

    def prepare_projection(
        self,
        *,
        batch_size: int,
        method: str = "auto",
        config: ProjectionConfig | None = None,
        dtype=jnp.float64,
        backend: str = "cpu",
    ):
        selected = str(method).lower()
        if selected == "auto":
            selected = "linear" if self.is_explicit_linear else "sparse"
        if config is None:
            config = ProjectionConfig(
                tolerance=1e-9,
                max_iterations=100,
                implicit_gradient=ImplicitGradientConfig(
                    differentiate_problem_parameters=True
                ),
            )
        if selected == "linear":
            if not self.is_explicit_linear:
                raise ValueError("explicit linear projection is not available for this compiled DAE")
            rows = int(self.linear_matrix.shape[0])
            layer = LinearEqualityProjection(
                A=np.asarray(self.linear_matrix),
                b0=np.zeros((rows,), dtype=np.float64),
                b_parameter_map=np.eye(rows, dtype=np.float64),
            ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
            return PreparedDAEProjection(
                self, layer, "linear", config, dtype, backend, int(batch_size)
            )
        problem = FunctionalProblem(
            parameter_size=self.parameter_size,
            variable_size=self.variable_size,
            objective=lambda theta, y: jnp.asarray(0.0, dtype=y.dtype),
            equalities=self.scaled_equalities,
            inequalities=self.scaled_inequalities,
            equality_size=self.system.equality_size,
            inequality_size=self.system.algebraic_inequalities.size,
            name=f"{self.system.name}_compiled",
        )
        if selected == "sparse":
            layer = SparseProjection(problem, config=config).prepare(
                batch_size=batch_size, dtype=dtype, backend=backend
            )
        elif selected in {"fb", "fischer_burmeister"}:
            layer = FischerBurmeisterProjection(
                problem,
                tolerance=config.tolerance,
                max_iterations=config.max_iterations,
                line_search=config.backtracking,
                gradient_policy=config.implicit_gradient.policy,
                inexact_gradient_tolerance=config.implicit_gradient.inexact_tolerance,
                adjoint_gradient_tolerance=config.implicit_gradient.adjoint_tolerance,
                differentiate_problem_parameters=(
                    config.implicit_gradient.differentiate_problem_parameters
                ),
            ).prepare(batch_size=batch_size, dtype=dtype, backend=backend)
        elif selected in {"fischer-burmeister-newton", "fischer-burmeister-reduced", "primal-dual-interior-point"}:
            from daehn.opt.api import prepare_projection
            from kkthn.opt.config import FischerBurmeisterConfig, ProjectionConfig as SharedConfig, ImplicitGradientConfig as SharedImplicit
            implicit = SharedImplicit(**vars(config.implicit_gradient))
            config_type = SharedConfig if selected == "primal-dual-interior-point" else FischerBurmeisterConfig
            if type(config).__module__.startswith("kkthn."):
                shared_config = config
            else:
                from kkthn.opt.config import LinearSolverConfig as SharedLinear
                shared_linear = SharedLinear(**vars(config.linear_solver))
                options = dict(tolerance=config.tolerance, max_iterations=config.max_iterations,
                               implicit_gradient=implicit, linear_solver=shared_linear,
                               transpose_mode=config.transpose_mode)
                if selected == "primal-dual-interior-point":
                    options.update(mu=config.mu, backtracking=config.backtracking,
                                   max_backtracking_steps=config.max_backtracking_steps,
                                   backtracking_factor=config.backtracking_factor,
                                   interior_margin=config.interior_margin,
                                   store_residual_history=config.store_residual_history)
                shared_config = config_type(**options)
            layer = prepare_projection(problem, method=selected, config=shared_config, batch_size=batch_size, dtype=dtype, backend=backend)
        else:
            raise ValueError("unknown projection method")
        return PreparedDAEProjection(
            self, layer, selected, config, dtype, backend, int(batch_size)
        )


@dataclass(frozen=True, eq=False)
class PreparedDAEProjection:
    compiled: CompiledDAE
    layer: Any
    method: str
    config: ProjectionConfig
    dtype: Any
    backend: str
    batch_size: int

    def _parameters(self, context):
        if self.method == "linear":
            zeros = jnp.zeros((context.shape[0], self.compiled.variable_size), dtype=context.dtype)
            return -jax.vmap(self.compiled.scaled_equalities)(context, zeros)
        return context

    def __call__(self, target, context):
        return self.layer(target, self._parameters(context))

    def with_info(self, target, context):
        return self.layer.with_info(target, self._parameters(context))
