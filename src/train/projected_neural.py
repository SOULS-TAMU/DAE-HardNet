"""JIT-first projected neural models built on the sparse PyOpt solver.

The module composes a user-supplied JAX backbone with the prepared Euclidean
projection solver.  Projection derivatives use the final-KKT implicit adjoint;
backbone derivatives use normal JAX reverse-mode autodiff.  Complete train,
evaluation, and prediction entry points are lowered and compiled during
:meth:`ProjectedNeuralModel.build`, so :meth:`fit` performs no JAX transforms or
model recompilation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Callable, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..opt.sparse.batch_iteration import BATCH_STATUS_CONVERGED
from ..problems.problem import PreparedSparseProjectionBatchSolver

Array = jax.Array
PyTree = Any
BackboneInit = Callable[[Array, int, int, Any], PyTree]
BackboneApply = Callable[[PyTree, Array], Array]
ProjectionFactory = Callable[[int], PreparedSparseProjectionBatchSolver]
GradientPolicy = Literal["strict", "residual_checked", "always"]
InvalidBatchPolicy = Literal["skip", "use_valid"]


def _normalize_gradient_policy(value: str) -> GradientPolicy:
    selected = str(value).strip().lower()
    if selected not in {"strict", "residual_checked", "always"}:
        raise ValueError(
            "gradient_policy must be strict, residual_checked, or always."
        )
    return selected  # type: ignore[return-value]


def _normalize_invalid_batch_policy(value: str) -> InvalidBatchPolicy:
    selected = str(value).strip().lower()
    if selected not in {"skip", "use_valid"}:
        raise ValueError("invalid_batch_policy must be skip or use_valid.")
    return selected  # type: ignore[return-value]


def configure_compilation_cache(path: str | Path | None) -> None:
    """Configure JAX's persistent compilation cache before model compilation."""

    if path is None:
        return
    cache_path = Path(path).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


class AdamState(NamedTuple):
    count: Array
    first_moment: PyTree
    second_moment: PyTree


@dataclass(frozen=True)
class Adam:
    """Small dependency-free Adam optimizer with an Optax-like interface."""

    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def init(self, parameters: PyTree) -> AdamState:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
        return AdamState(
            count=jnp.asarray(0, dtype=jnp.int32),
            first_moment=zeros,
            second_moment=zeros,
        )

    def update(
        self,
        gradients: PyTree,
        state: AdamState,
        parameters: PyTree | None = None,
    ) -> tuple[PyTree, AdamState]:
        del parameters
        count = state.count + jnp.asarray(1, dtype=state.count.dtype)
        first = jax.tree_util.tree_map(
            lambda m, g: self.beta1 * m + (1.0 - self.beta1) * g,
            state.first_moment,
            gradients,
        )
        second = jax.tree_util.tree_map(
            lambda v, g: self.beta2 * v + (1.0 - self.beta2) * jnp.square(g),
            state.second_moment,
            gradients,
        )
        count_f = count.astype(jnp.float64)
        first_scale = 1.0 - self.beta1**count_f
        second_scale = 1.0 - self.beta2**count_f
        updates = jax.tree_util.tree_map(
            lambda m, v: -self.learning_rate
            * (m / first_scale)
            / (jnp.sqrt(v / second_scale) + self.epsilon),
            first,
            second,
        )
        return updates, AdamState(count, first, second)


def apply_updates(parameters: PyTree, updates: PyTree) -> PyTree:
    """Apply additive optimizer updates to an arbitrary parameter pytree."""

    return jax.tree_util.tree_map(lambda p, u: p + u, parameters, updates)


@dataclass(frozen=True)
class MLPBackbone:
    """Optional pure-JAX MLP backbone compatible with the generic framework."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: Callable[[Array], Array] = jax.nn.tanh

    def init(
        self,
        key: Array,
        input_dim: int,
        output_dim: int,
        dtype: Any = jnp.float64,
    ) -> tuple[dict[str, Array], ...]:
        sizes = (int(input_dim), *map(int, self.hidden_sizes), int(output_dim))
        keys = jax.random.split(key, len(sizes) - 1)
        layers = []
        for layer_key, fan_in, fan_out in zip(
            keys, sizes[:-1], sizes[1:], strict=True
        ):
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            weights = jax.random.uniform(
                layer_key,
                (fan_in, fan_out),
                minval=-limit,
                maxval=limit,
                dtype=dtype,
            )
            bias = jnp.zeros((fan_out,), dtype=dtype)
            layers.append({"W": weights, "b": bias})
        return tuple(layers)

    def apply(self, parameters: Sequence[dict[str, Array]], inputs: Array) -> Array:
        values = inputs
        for layer_index, layer in enumerate(parameters):
            values = values @ layer["W"] + layer["b"]
            if layer_index + 1 < len(parameters):
                values = self.activation(values)
        return values


class ProjectionLayerOutput(NamedTuple):
    y_tilde: Array
    forward_status_codes: Array
    forward_iterations: Array
    forward_residual_inf_norm: Array
    forward_state_finite: Array
    forward_gradient_eligible: Array


@dataclass(frozen=True)
class ImplicitProjectionLayer:
    project: Callable[[Array, Array], Array]
    project_with_info: Callable[[Array, Array], ProjectionLayerOutput]
    pack_parameters: Callable[[Array, Array], Array]
    physical_parameter_count: int
    output_dim: int


def make_implicit_projection_layer(
    prepared: PreparedSparseProjectionBatchSolver,
    *,
    mu: float = 1e-4,
    differentiate_problem_parameters: bool = False,
    gradient_policy: str = "residual_checked",
    inexact_gradient_tolerance: float = 1e-6,
    adjoint_gradient_tolerance: float = 1e-8,
) -> ImplicitProjectionLayer:
    """Create a custom-VJP projection callable for a prepared sparse solver.

    Arguments are ``(problem_parameters, y_hat)``.  The backward rule solves
    one final sparse KKT adjoint.  It returns the adjoint primal block as the
    projection-target VJP and, when requested, contracts the same adjoint with
    the parameterized KKT residual for inverse-parameter gradients.
    """

    selected_gradient_policy = _normalize_gradient_policy(gradient_policy)
    inexact_gradient_tolerance = float(inexact_gradient_tolerance)
    adjoint_gradient_tolerance = float(adjoint_gradient_tolerance)
    if not np.isfinite(inexact_gradient_tolerance) or inexact_gradient_tolerance < 0.0:
        raise ValueError("inexact_gradient_tolerance must be finite and nonnegative.")
    if not np.isfinite(adjoint_gradient_tolerance) or adjoint_gradient_tolerance < 0.0:
        raise ValueError("adjoint_gradient_tolerance must be finite and nonnegative.")

    physical_indices = prepared.physical_parameter_indices
    target_indices = prepared.target_parameter_indices
    batch_size = int(prepared.batch_size)
    parameter_count = int(prepared.parameter_count)
    output_dim = int(len(prepared.compiled.variable_names))
    physical_count = int(parameter_count - output_dim)
    mu_values = jnp.full((batch_size,), float(mu), dtype=jnp.float64)

    def pack_parameters(problem_parameters: Array, y_hat: Array) -> Array:
        problem_parameters = jnp.asarray(problem_parameters, dtype=jnp.float64)
        y_hat = jnp.asarray(y_hat, dtype=jnp.float64)
        packed = jnp.zeros((batch_size, parameter_count), dtype=y_hat.dtype)
        packed = packed.at[:, target_indices].set(y_hat)
        return packed.at[:, physical_indices].set(problem_parameters)

    def _state_is_finite(state) -> Array:
        blocks = (state.x, state.s, state.y_c, state.y_d, state.z_L, state.z_U)
        finite = jnp.ones((batch_size,), dtype=jnp.bool_)
        for block in blocks:
            finite = finite & jnp.all(jnp.isfinite(block), axis=1)
        return finite

    def _forward_eligible(status, residual, state_finite) -> Array:
        residual_finite = jnp.isfinite(residual)
        if selected_gradient_policy == "strict":
            policy_ok = status == BATCH_STATUS_CONVERGED
        elif selected_gradient_policy == "residual_checked":
            policy_ok = jnp.logical_or(
                status == BATCH_STATUS_CONVERGED,
                residual <= inexact_gradient_tolerance,
            )
        else:
            policy_ok = jnp.ones_like(state_finite)
        return state_finite & residual_finite & policy_ok

    def solve_projection(problem_parameters: Array, y_hat: Array):
        packed = pack_parameters(problem_parameters, y_hat)
        forward = prepared.solve_with_linearization_jax(packed, mu_values)
        state_finite = _state_is_finite(forward.state)
        gradient_eligible = _forward_eligible(
            forward.status_codes, forward.final_residual_inf_norm, state_finite
        )
        output = ProjectionLayerOutput(
            y_tilde=forward.state.x,
            forward_status_codes=forward.status_codes,
            forward_iterations=forward.iterations,
            forward_residual_inf_norm=forward.final_residual_inf_norm,
            forward_state_finite=state_finite,
            forward_gradient_eligible=gradient_eligible,
        )
        return output, packed, forward

    @jax.custom_vjp
    def project_with_info(
        problem_parameters: Array, y_hat: Array
    ) -> ProjectionLayerOutput:
        output, _packed, _forward = solve_projection(problem_parameters, y_hat)
        return output

    def project_forward(problem_parameters: Array, y_hat: Array):
        output, packed, forward = solve_projection(problem_parameters, y_hat)
        saved = (
            forward.state,
            forward.final_linearization,
            forward.status_codes,
            forward.final_residual_inf_norm,
            output.forward_state_finite,
            packed,
        )
        return output, saved

    def project_backward(saved, output_cotangent):
        (
            final_state,
            final_linearization,
            forward_status,
            forward_residual,
            forward_state_finite,
            packed,
        ) = saved
        grad_y_tilde = output_cotangent.y_tilde
        adjoint = prepared.solve_yhat_adjoint_jax(
            final_state,
            final_linearization,
            grad_y_tilde,
        )
        forward_valid = _forward_eligible(
            forward_status, forward_residual, forward_state_finite
        )
        adjoint_finite = (
            jnp.all(jnp.isfinite(adjoint.x), axis=1)
            & jnp.all(jnp.isfinite(adjoint.y_c), axis=1)
            & jnp.all(jnp.isfinite(adjoint.y_d), axis=1)
            & jnp.isfinite(adjoint.linear_residual_inf_norm)
        )
        if selected_gradient_policy == "strict":
            adjoint_valid = adjoint.linear_status == 0
        elif selected_gradient_policy == "residual_checked":
            adjoint_valid = jnp.logical_or(
                adjoint.linear_status == 0,
                adjoint.linear_residual_inf_norm <= adjoint_gradient_tolerance,
            )
        else:
            adjoint_valid = jnp.ones_like(adjoint_finite)
        valid = forward_valid & adjoint_finite & adjoint_valid
        grad_y_hat = adjoint.x
        if differentiate_problem_parameters:
            full_parameter_gradient = prepared.parameter_vjp_from_adjoint_jax(
                final_state,
                packed,
                adjoint,
            )
            grad_problem_parameters = full_parameter_gradient[:, physical_indices]
        else:
            grad_problem_parameters = jnp.zeros(
                (batch_size, physical_count), dtype=grad_y_hat.dtype
            )
        grad_y_hat = jnp.nan_to_num(grad_y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        grad_problem_parameters = jnp.nan_to_num(
            grad_problem_parameters, nan=0.0, posinf=0.0, neginf=0.0
        )
        grad_y_hat = jnp.where(valid[:, None], grad_y_hat, 0.0)
        grad_problem_parameters = jnp.where(
            valid[:, None], grad_problem_parameters, 0.0
        )
        return grad_problem_parameters, grad_y_hat

    project_with_info.defvjp(project_forward, project_backward)

    def project(problem_parameters: Array, y_hat: Array) -> Array:
        return project_with_info(problem_parameters, y_hat).y_tilde

    return ImplicitProjectionLayer(
        project=project,
        project_with_info=project_with_info,
        pack_parameters=pack_parameters,
        physical_parameter_count=physical_count,
        output_dim=output_dim,
    )


class ProjectedMetrics(NamedTuple):
    loss: Array
    task_loss: Array
    raw_mse: Array
    consistency_loss: Array
    inverse_regularization: Array
    projection_distance: Array
    forward_converged_fraction: Array
    forward_finite_fraction: Array
    gradient_eligible_fraction: Array
    update_accepted: Array
    max_forward_residual: Array
    mean_forward_iterations: Array


class PredictionOutput(NamedTuple):
    y_hat: Array
    y_tilde: Array
    forward_status_codes: Array
    forward_iterations: Array
    forward_residual_inf_norm: Array


@dataclass(frozen=True)
class BuildReport:
    mode: str
    backend: str
    device: str
    batch_size: int
    input_dim: int
    output_dim: int
    physical_parameter_count: int
    projection_prepare_seconds: float
    backbone_initialize_seconds: float
    graph_construct_seconds: float
    raw_train_compile_seconds: float
    raw_eval_compile_seconds: float
    train_compile_seconds: float
    eval_compile_seconds: float
    predict_compile_seconds: float
    validation_seconds: float
    total_build_seconds: float
    persistent_cache_directory: str | None
    validated: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledProjectedGraphs:
    raw_train_step: Any
    raw_eval_step: Any
    train_step: Any
    eval_step: Any
    predict_step: Any
    raw_predict_step: Any


def _copy_tree_to_device(tree: PyTree, device: Any) -> PyTree:
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jnp.array(value), device), tree
    )


def _call_optimizer_update(optimizer, gradients, state, parameters):
    try:
        return optimizer.update(gradients, state, parameters)
    except TypeError:
        return optimizer.update(gradients, state)


def _per_sample(values: Array, batch_size: int) -> Array:
    values = jnp.asarray(values, dtype=jnp.float64)
    if values.ndim == 0:
        return jnp.broadcast_to(values, (batch_size,))
    if values.shape[0] != batch_size:
        raise ValueError(
            "A loss/objective function must return a scalar or an array whose "
            f"first dimension is the fixed batch size {batch_size}."
        )
    if values.ndim == 1:
        return values
    return jnp.mean(values, axis=tuple(range(1, values.ndim)))


def _masked_mean(values: Array, valid_mask: Array) -> Array:
    weights = valid_mask.astype(values.dtype)
    return jnp.sum(weights * values) / jnp.maximum(jnp.sum(weights), 1.0)


def _tree_all_finite(tree: PyTree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    result = jnp.asarray(True)
    for leaf in leaves:
        result = result & jnp.all(jnp.isfinite(leaf))
    return result


def _tree_select(condition: Array, candidate: PyTree, original: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(condition, new, old), candidate, original
    )


class ProjectedNeuralModel:
    """Projected neural model with build-time AOT compilation.

    ``projection`` may be a prepared fixed-batch solver or a factory receiving
    ``batch_size``.  A factory is preferred because compilation-cache setup can
    occur before the projection solver is prepared.
    """

    def __init__(
        self,
        *,
        projection: PreparedSparseProjectionBatchSolver | ProjectionFactory,
        backbone_init: BackboneInit,
        backbone_apply: BackboneApply,
        input_dim: int,
        backbone_input_dim: int | None = None,
        mode: str = "supervised",
        optimizer: Any | None = None,
        data_loss_fn: Callable[[Array, Array, Array], Array] | None = None,
        objective_fn: Callable[[Array, Array], Array] | None = None,
        problem_parameter_fn: Callable[[Array, Array | None], Array] | None = None,
        backbone_input_fn: Callable[[Array, Array | None], Array] | None = None,
        inverse_init: Any | None = None,
        inverse_regularizer: Callable[[Array], Array] | None = None,
        consistency_weight: float = 0.0,
        mu: float = 1e-4,
        optimizer_apply_updates: Callable[[PyTree, PyTree], PyTree] = apply_updates,
        gradient_policy: str = "residual_checked",
        inexact_gradient_tolerance: float = 1e-6,
        adjoint_gradient_tolerance: float = 1e-8,
        invalid_batch_policy: str = "skip",
        invalidate_failed_gradients: bool | None = None,
    ) -> None:
        selected_mode = str(mode).strip().lower()
        if selected_mode not in {"supervised", "inverse", "optimization"}:
            raise ValueError(
                "mode must be 'supervised', 'inverse', or 'optimization'."
            )
        if selected_mode == "optimization" and objective_fn is None:
            raise ValueError("optimization mode requires objective_fn.")
        if selected_mode == "inverse" and inverse_init is None:
            raise ValueError("inverse mode requires inverse_init.")
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive.")
        if float(consistency_weight) < 0.0:
            raise ValueError("consistency_weight must be nonnegative.")
        self.projection_definition = projection
        self.backbone_init = backbone_init
        self.backbone_apply = backbone_apply
        self.input_dim = int(input_dim)
        self.backbone_input_dim = (
            None if backbone_input_dim is None else int(backbone_input_dim)
        )
        if self.backbone_input_dim is not None and self.backbone_input_dim < 1:
            raise ValueError("backbone_input_dim must be positive when supplied.")
        self.mode = selected_mode
        self.optimizer = optimizer or Adam()
        self.optimizer_apply_updates = optimizer_apply_updates
        self.data_loss_fn = data_loss_fn
        self.objective_fn = objective_fn
        self.problem_parameter_fn = problem_parameter_fn
        self.backbone_input_fn = backbone_input_fn
        self.inverse_init = (
            None
            if inverse_init is None
            else np.asarray(inverse_init, dtype=np.float64).reshape(-1)
        )
        self.inverse_regularizer = inverse_regularizer
        self.consistency_weight = float(consistency_weight)
        self.mu = float(mu)
        self.gradient_policy = _normalize_gradient_policy(gradient_policy)
        self.invalid_batch_policy = _normalize_invalid_batch_policy(
            invalid_batch_policy
        )
        self.inexact_gradient_tolerance = float(inexact_gradient_tolerance)
        self.adjoint_gradient_tolerance = float(adjoint_gradient_tolerance)
        if (
            not np.isfinite(self.inexact_gradient_tolerance)
            or self.inexact_gradient_tolerance < 0.0
        ):
            raise ValueError(
                "inexact_gradient_tolerance must be finite and nonnegative."
            )
        if (
            not np.isfinite(self.adjoint_gradient_tolerance)
            or self.adjoint_gradient_tolerance < 0.0
        ):
            raise ValueError(
                "adjoint_gradient_tolerance must be finite and nonnegative."
            )
        # Deprecated compatibility argument. Failed gradients are never filled
        # with NaN; they are sanitized and controlled by gradient_policy.
        del invalidate_failed_gradients

        self.prepared_projection: PreparedSparseProjectionBatchSolver | None = None
        self.projection_layer: ImplicitProjectionLayer | None = None
        self.parameters: PyTree | None = None
        self.optimizer_state: PyTree | None = None
        self.graphs: CompiledProjectedGraphs | None = None
        self.build_report: BuildReport | None = None
        self.batch_size: int | None = None
        self.dtype = jnp.float64

    @property
    def is_built(self) -> bool:
        return self.graphs is not None

    def _resolve_projection(self, batch_size: int):
        definition = self.projection_definition
        if isinstance(definition, PreparedSparseProjectionBatchSolver):
            prepared = definition
        else:
            prepared = definition(int(batch_size))
        if prepared.batch_size != int(batch_size):
            raise ValueError(
                f"Prepared projection batch size {prepared.batch_size} does not "
                f"match requested model batch size {batch_size}."
            )
        return prepared

    def _default_problem_parameters(
        self,
        x_batch: Array,
        inverse_parameters: Array | None,
        physical_count: int,
    ) -> Array:
        if inverse_parameters is None:
            values = x_batch
        else:
            inverse_batch = jnp.broadcast_to(
                inverse_parameters,
                (x_batch.shape[0], inverse_parameters.shape[0]),
            )
            values = jnp.concatenate((x_batch, inverse_batch), axis=1)
        if values.shape[1] != physical_count:
            raise ValueError(
                "Default problem-parameter packing produced "
                f"{values.shape[1]} columns; the projection expects "
                f"{physical_count}. Supply problem_parameter_fn to customize "
                "the mapping."
            )
        return values

    def _default_backbone_inputs(
        self, x_batch: Array, inverse_parameters: Array | None
    ) -> Array:
        if inverse_parameters is None:
            return x_batch
        inverse_batch = jnp.broadcast_to(
            inverse_parameters,
            (x_batch.shape[0], inverse_parameters.shape[0]),
        )
        return jnp.concatenate((x_batch, inverse_batch), axis=1)

    def build(
        self,
        *,
        batch_size: int,
        seed: int = 0,
        cache_dir: str | Path | None = None,
        validate: bool = True,
        compile_raw_predict: bool = True,
        example_x: Any | None = None,
        example_y: Any | None = None,
    ) -> BuildReport:
        """Prepare the solver and AOT-compile train/eval/predict entry points."""

        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive.")
        if self.is_built:
            raise RuntimeError("This model is already built.")
        total_start = time.perf_counter()
        configure_compilation_cache(cache_dir)

        projection_start = time.perf_counter()
        prepared = self._resolve_projection(int(batch_size))
        projection_seconds = time.perf_counter() - projection_start
        self.prepared_projection = prepared
        self.batch_size = int(batch_size)
        physical_count = int(
            prepared.parameter_count - len(prepared.compiled.variable_names)
        )
        output_dim = int(len(prepared.compiled.variable_names))
        differentiate_problem_parameters = self.mode == "inverse"
        projection_layer = make_implicit_projection_layer(
            prepared,
            mu=self.mu,
            differentiate_problem_parameters=differentiate_problem_parameters,
            gradient_policy=self.gradient_policy,
            inexact_gradient_tolerance=self.inexact_gradient_tolerance,
            adjoint_gradient_tolerance=self.adjoint_gradient_tolerance,
        )
        self.projection_layer = projection_layer

        inverse_size = 0 if self.inverse_init is None else int(self.inverse_init.size)
        default_backbone_input_dim = (
            self.input_dim + inverse_size
            if self.backbone_input_dim is None
            else self.backbone_input_dim
        )
        backbone_initialize_start = time.perf_counter()
        key = jax.random.PRNGKey(int(seed))
        backbone_parameters = self.backbone_init(
            key,
            default_backbone_input_dim,
            output_dim,
            self.dtype,
        )
        if self.mode == "inverse":
            trainable_parameters = {
                "backbone": backbone_parameters,
                "inverse": jnp.asarray(self.inverse_init, dtype=self.dtype),
            }
        else:
            trainable_parameters = {"backbone": backbone_parameters}
        optimizer_state = self.optimizer.init(trainable_parameters)
        backbone_initialize_seconds = time.perf_counter() - backbone_initialize_start
        graph_construct_start = time.perf_counter()

        batch_size_static = int(batch_size)
        consistency_weight = self.consistency_weight
        data_loss_fn = self.data_loss_fn
        objective_fn = self.objective_fn
        inverse_regularizer = self.inverse_regularizer
        problem_parameter_fn = self.problem_parameter_fn
        backbone_input_fn = self.backbone_input_fn
        backbone_apply = self.backbone_apply
        optimizer = self.optimizer
        optimizer_apply = self.optimizer_apply_updates
        mode = self.mode
        invalid_batch_policy = self.invalid_batch_policy

        def split_trainable(parameters):
            inverse = parameters.get("inverse") if mode == "inverse" else None
            return parameters["backbone"], inverse

        def build_problem_parameters(x_batch, inverse):
            if problem_parameter_fn is not None:
                return problem_parameter_fn(x_batch, inverse)
            return self._default_problem_parameters(
                x_batch, inverse, physical_count
            )

        def build_backbone_inputs(x_batch, inverse):
            if backbone_input_fn is not None:
                return backbone_input_fn(x_batch, inverse)
            return self._default_backbone_inputs(x_batch, inverse)

        def forward_outputs(parameters, x_batch):
            network_parameters, inverse = split_trainable(parameters)
            network_inputs = build_backbone_inputs(x_batch, inverse)
            y_hat = backbone_apply(network_parameters, network_inputs)
            problem_parameters = build_problem_parameters(x_batch, inverse)
            projection_output = projection_layer.project_with_info(
                problem_parameters, y_hat
            )
            return y_hat, projection_output

        def inverse_regularization_value(inverse):
            if inverse is not None and inverse_regularizer is not None:
                return jnp.asarray(
                    inverse_regularizer(inverse), dtype=jnp.float64
                )
            return jnp.asarray(0.0, dtype=jnp.float64)

        def metrics_from_losses(
            task_per_sample,
            raw_mse_per_sample,
            y_hat,
            projection_output,
            valid_mask,
            inverse,
        ):
            finite_outputs = (
                projection_output.forward_state_finite
                & jnp.all(jnp.isfinite(y_hat), axis=1)
                & jnp.all(jnp.isfinite(projection_output.y_tilde), axis=1)
            )
            gradient_eligible = (
                valid_mask
                & finite_outputs
                & projection_output.forward_gradient_eligible
            )
            safe_task = jnp.nan_to_num(
                task_per_sample, nan=0.0, posinf=0.0, neginf=0.0
            )
            safe_raw_mse = jnp.nan_to_num(
                raw_mse_per_sample, nan=0.0, posinf=0.0, neginf=0.0
            )
            report_mask = valid_mask & finite_outputs
            task_loss = _masked_mean(safe_task, report_mask)
            raw_mse = _masked_mean(safe_raw_mse, report_mask)
            projection_distance_per_sample = jnp.mean(
                jnp.square(
                    jnp.nan_to_num(y_hat - projection_output.y_tilde)
                ),
                axis=1,
            )
            projection_distance = _masked_mean(
                projection_distance_per_sample, report_mask
            )
            consistency_loss = consistency_weight * projection_distance
            inverse_reg = inverse_regularization_value(inverse)
            reported_total_loss = task_loss + consistency_loss + inverse_reg

            # The differentiable scalar uses only rows accepted by the selected
            # inexact-gradient policy. Metrics continue to report all finite
            # final iterates so skipped batches never appear as zero error.
            training_task_loss = _masked_mean(safe_task, gradient_eligible)
            training_projection_distance = _masked_mean(
                projection_distance_per_sample, gradient_eligible
            )
            training_total_loss = (
                training_task_loss
                + consistency_weight * training_projection_distance
                + inverse_reg
            )
            valid_weight = valid_mask.astype(jnp.float64)
            denominator = jnp.maximum(jnp.sum(valid_weight), 1.0)
            converged = jnp.logical_and(
                valid_mask,
                projection_output.forward_status_codes
                == BATCH_STATUS_CONVERGED,
            )
            converged_fraction = (
                jnp.sum(converged.astype(jnp.float64)) / denominator
            )
            finite_fraction = (
                jnp.sum((valid_mask & finite_outputs).astype(jnp.float64))
                / denominator
            )
            eligible_fraction = (
                jnp.sum(gradient_eligible.astype(jnp.float64)) / denominator
            )
            masked_residual = jnp.where(
                valid_mask,
                projection_output.forward_residual_inf_norm,
                -jnp.inf,
            )
            max_residual = jnp.max(masked_residual)
            mean_iterations = jnp.sum(
                valid_weight
                * projection_output.forward_iterations.astype(jnp.float64)
            ) / denominator
            if invalid_batch_policy == "skip":
                default_update_accepted = eligible_fraction >= (1.0 - 1e-15)
            else:
                default_update_accepted = eligible_fraction > 0.0
            return training_total_loss, ProjectedMetrics(
                loss=reported_total_loss,
                task_loss=task_loss,
                raw_mse=raw_mse,
                consistency_loss=consistency_loss,
                inverse_regularization=inverse_reg,
                projection_distance=projection_distance,
                forward_converged_fraction=converged_fraction,
                forward_finite_fraction=finite_fraction,
                gradient_eligible_fraction=eligible_fraction,
                update_accepted=default_update_accepted.astype(jnp.float64),
                max_forward_residual=max_residual,
                mean_forward_iterations=mean_iterations,
            )

        def raw_metrics_from_losses(
            task_per_sample,
            raw_mse_per_sample,
            valid_mask,
            inverse,
        ):
            safe_task = jnp.nan_to_num(
                task_per_sample, nan=0.0, posinf=0.0, neginf=0.0
            )
            safe_raw = jnp.nan_to_num(
                raw_mse_per_sample, nan=0.0, posinf=0.0, neginf=0.0
            )
            task_loss = _masked_mean(safe_task, valid_mask)
            raw_mse = _masked_mean(safe_raw, valid_mask)
            inverse_reg = inverse_regularization_value(inverse)
            total_loss = task_loss + inverse_reg
            zero = jnp.asarray(0.0, dtype=jnp.float64)
            one = jnp.asarray(1.0, dtype=jnp.float64)
            return total_loss, ProjectedMetrics(
                loss=total_loss,
                task_loss=task_loss,
                raw_mse=raw_mse,
                consistency_loss=zero,
                inverse_regularization=inverse_reg,
                projection_distance=zero,
                forward_converged_fraction=one,
                forward_finite_fraction=one,
                gradient_eligible_fraction=one,
                update_accepted=one,
                max_forward_residual=zero,
                mean_forward_iterations=zero,
            )

        def accept_or_reject_update(
            parameters, opt_state, gradients, loss, metrics, valid_mask
        ):
            gradients_finite = _tree_all_finite(gradients)
            loss_finite = jnp.isfinite(loss)
            if invalid_batch_policy == "skip":
                projection_valid = metrics.gradient_eligible_fraction >= (
                    1.0 - 1e-15
                )
            else:
                projection_valid = metrics.gradient_eligible_fraction > 0.0
            update_valid = gradients_finite & loss_finite & projection_valid
            safe_gradients = jax.tree_util.tree_map(
                lambda value: jnp.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                ),
                gradients,
            )
            updates, candidate_opt_state = _call_optimizer_update(
                optimizer, safe_gradients, opt_state, parameters
            )
            candidate_parameters = optimizer_apply(parameters, updates)
            candidate_finite = (
                _tree_all_finite(candidate_parameters)
                & _tree_all_finite(candidate_opt_state)
            )
            update_valid = update_valid & candidate_finite
            new_parameters = _tree_select(
                update_valid, candidate_parameters, parameters
            )
            new_opt_state = _tree_select(
                update_valid, candidate_opt_state, opt_state
            )
            metrics = metrics._replace(
                update_accepted=update_valid.astype(jnp.float64)
            )
            return new_parameters, new_opt_state, metrics

        if mode in {"supervised", "inverse"}:

            def raw_loss_with_aux(parameters, x_batch, y_data, valid_mask):
                network_parameters, inverse = split_trainable(parameters)
                network_inputs = build_backbone_inputs(x_batch, inverse)
                y_hat = backbone_apply(network_parameters, network_inputs)
                if data_loss_fn is None:
                    task_values = jnp.mean(jnp.square(y_hat - y_data), axis=1)
                else:
                    task_values = data_loss_fn(y_hat, y_data, x_batch)
                task_per_sample = _per_sample(task_values, batch_size_static)
                raw_mse_per_sample = jnp.mean(
                    jnp.square(y_hat - y_data), axis=1
                )
                return raw_metrics_from_losses(
                    task_per_sample,
                    raw_mse_per_sample,
                    valid_mask,
                    inverse,
                )

            def raw_train_step(parameters, opt_state, x_batch, y_data, valid_mask):
                (loss, metrics), gradients = jax.value_and_grad(
                    raw_loss_with_aux, has_aux=True
                )(parameters, x_batch, y_data, valid_mask)
                return accept_or_reject_update(
                    parameters,
                    opt_state,
                    gradients,
                    loss,
                    metrics,
                    valid_mask,
                )

            def raw_eval_step(parameters, x_batch, y_data, valid_mask):
                _loss, metrics = raw_loss_with_aux(
                    parameters, x_batch, y_data, valid_mask
                )
                return metrics

            def loss_with_aux(parameters, x_batch, y_data, valid_mask):
                _network_parameters, inverse = split_trainable(parameters)
                y_hat, projection_output = forward_outputs(parameters, x_batch)
                if data_loss_fn is None:
                    task_values = jnp.mean(
                        jnp.square(projection_output.y_tilde - y_data), axis=1
                    )
                else:
                    task_values = data_loss_fn(
                        projection_output.y_tilde, y_data, x_batch
                    )
                task_per_sample = _per_sample(task_values, batch_size_static)
                raw_mse_per_sample = jnp.mean(
                    jnp.square(projection_output.y_tilde - y_data), axis=1
                )
                return metrics_from_losses(
                    task_per_sample,
                    raw_mse_per_sample,
                    y_hat,
                    projection_output,
                    valid_mask,
                    inverse,
                )

            def train_step(parameters, opt_state, x_batch, y_data, valid_mask):
                (loss, metrics), gradients = jax.value_and_grad(
                    loss_with_aux, has_aux=True
                )(parameters, x_batch, y_data, valid_mask)
                return accept_or_reject_update(
                    parameters,
                    opt_state,
                    gradients,
                    loss,
                    metrics,
                    valid_mask,
                )

            def eval_step(parameters, x_batch, y_data, valid_mask):
                _loss, metrics = loss_with_aux(
                    parameters, x_batch, y_data, valid_mask
                )
                return metrics

        else:
            raw_train_step = None
            raw_eval_step = None

            def loss_with_aux(parameters, x_batch, valid_mask):
                _network_parameters, inverse = split_trainable(parameters)
                y_hat, projection_output = forward_outputs(parameters, x_batch)
                assert objective_fn is not None
                task_values = objective_fn(projection_output.y_tilde, x_batch)
                task_per_sample = _per_sample(task_values, batch_size_static)
                return metrics_from_losses(
                    task_per_sample,
                    task_per_sample,
                    y_hat,
                    projection_output,
                    valid_mask,
                    inverse,
                )

            def train_step(parameters, opt_state, x_batch, valid_mask):
                (loss, metrics), gradients = jax.value_and_grad(
                    loss_with_aux, has_aux=True
                )(parameters, x_batch, valid_mask)
                return accept_or_reject_update(
                    parameters,
                    opt_state,
                    gradients,
                    loss,
                    metrics,
                    valid_mask,
                )

            def eval_step(parameters, x_batch, valid_mask):
                _loss, metrics = loss_with_aux(parameters, x_batch, valid_mask)
                return metrics

        def predict_step(parameters, x_batch):
            y_hat, projection_output = forward_outputs(parameters, x_batch)
            return PredictionOutput(
                y_hat=y_hat,
                y_tilde=projection_output.y_tilde,
                forward_status_codes=projection_output.forward_status_codes,
                forward_iterations=projection_output.forward_iterations,
                forward_residual_inf_norm=projection_output.forward_residual_inf_norm,
            )

        def raw_predict_step(parameters, x_batch):
            network_parameters, inverse = split_trainable(parameters)
            network_inputs = build_backbone_inputs(x_batch, inverse)
            return backbone_apply(network_parameters, network_inputs)

        graph_construct_seconds = time.perf_counter() - graph_construct_start

        device = prepared.device
        parameters_device = _copy_tree_to_device(trainable_parameters, device)
        optimizer_state_device = _copy_tree_to_device(optimizer_state, device)
        def fixed_example(values, width):
            if values is None:
                host = np.zeros((batch_size_static, width), dtype=np.float64)
            else:
                raw = np.asarray(values, dtype=np.float64)
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                if raw.ndim != 2 or raw.shape[1] != width or raw.shape[0] < 1:
                    raise ValueError(
                        f"Build example must have shape (N, {width}) with N >= 1; "
                        f"got {raw.shape}."
                    )
                host = np.repeat(raw[-1:, :], batch_size_static, axis=0)
                count = min(batch_size_static, raw.shape[0])
                host[:count] = raw[:count]
            return jax.device_put(jnp.asarray(host, dtype=self.dtype), device)

        dummy_x = fixed_example(example_x, self.input_dim)
        dummy_y = fixed_example(example_y, output_dim)
        dummy_mask = jax.device_put(
            jnp.ones((batch_size_static,), dtype=jnp.bool_), device
        )

        raw_train_compile_seconds = 0.0
        raw_eval_compile_seconds = 0.0
        raw_train_executable = None
        raw_eval_executable = None
        if mode in {"supervised", "inverse"}:
            assert raw_train_step is not None
            assert raw_eval_step is not None
            raw_train_compile_start = time.perf_counter()
            raw_train_executable = jax.jit(
                raw_train_step, donate_argnums=(0, 1)
            ).lower(
                parameters_device,
                optimizer_state_device,
                dummy_x,
                dummy_y,
                dummy_mask,
            ).compile()
            raw_train_compile_seconds = (
                time.perf_counter() - raw_train_compile_start
            )

            raw_eval_compile_start = time.perf_counter()
            raw_eval_executable = jax.jit(raw_eval_step).lower(
                parameters_device, dummy_x, dummy_y, dummy_mask
            ).compile()
            raw_eval_compile_seconds = (
                time.perf_counter() - raw_eval_compile_start
            )

        train_compile_start = time.perf_counter()
        train_jit = jax.jit(train_step, donate_argnums=(0, 1))
        if mode in {"supervised", "inverse"}:
            train_executable = train_jit.lower(
                parameters_device,
                optimizer_state_device,
                dummy_x,
                dummy_y,
                dummy_mask,
            ).compile()
        else:
            train_executable = train_jit.lower(
                parameters_device,
                optimizer_state_device,
                dummy_x,
                dummy_mask,
            ).compile()
        train_compile_seconds = time.perf_counter() - train_compile_start

        eval_compile_start = time.perf_counter()
        eval_jit = jax.jit(eval_step)
        if mode in {"supervised", "inverse"}:
            eval_executable = eval_jit.lower(
                parameters_device, dummy_x, dummy_y, dummy_mask
            ).compile()
        else:
            eval_executable = eval_jit.lower(
                parameters_device, dummy_x, dummy_mask
            ).compile()
        eval_compile_seconds = time.perf_counter() - eval_compile_start

        predict_compile_start = time.perf_counter()
        predict_executable = jax.jit(predict_step).lower(
            parameters_device, dummy_x
        ).compile()
        if compile_raw_predict:
            raw_predict_executable = jax.jit(raw_predict_step).lower(
                parameters_device, dummy_x
            ).compile()
        else:
            raw_predict_executable = None
        predict_compile_seconds = time.perf_counter() - predict_compile_start

        validation_seconds = 0.0
        validated = False
        if validate:
            validation_start = time.perf_counter()
            prediction = predict_executable(parameters_device, dummy_x)
            jax.block_until_ready(prediction.y_tilde)
            validation_parameters = _copy_tree_to_device(
                parameters_device, device
            )
            validation_optimizer_state = _copy_tree_to_device(
                optimizer_state_device, device
            )
            if mode in {"supervised", "inverse"}:
                validation_result = train_executable(
                    validation_parameters,
                    validation_optimizer_state,
                    dummy_x,
                    dummy_y,
                    dummy_mask,
                )
            else:
                validation_result = train_executable(
                    validation_parameters,
                    validation_optimizer_state,
                    dummy_x,
                    dummy_mask,
                )
            jax.block_until_ready(validation_result[2].loss)
            validation_metrics = jax.device_get(validation_result[2])
            if not np.isfinite(float(validation_metrics.loss)):
                raise RuntimeError("Build validation produced a nonfinite loss.")
            if float(validation_metrics.forward_converged_fraction) < 1.0:
                raise RuntimeError(
                    "Build validation projection did not converge for all "
                    "synthetic rows. Provide a projection factory whose default "
                    "zero batch is feasible or call build(validate=False)."
                )
            validation_seconds = time.perf_counter() - validation_start
            validated = True

        self.parameters = parameters_device
        self.optimizer_state = optimizer_state_device
        self.graphs = CompiledProjectedGraphs(
            raw_train_step=raw_train_executable,
            raw_eval_step=raw_eval_executable,
            train_step=train_executable,
            eval_step=eval_executable,
            predict_step=predict_executable,
            raw_predict_step=raw_predict_executable,
        )
        report = BuildReport(
            mode=mode,
            backend=prepared.backend,
            device=prepared.device.device_kind,
            batch_size=batch_size_static,
            input_dim=self.input_dim,
            output_dim=output_dim,
            physical_parameter_count=physical_count,
            projection_prepare_seconds=projection_seconds,
            backbone_initialize_seconds=backbone_initialize_seconds,
            graph_construct_seconds=max(0.0, graph_construct_seconds),
            raw_train_compile_seconds=raw_train_compile_seconds,
            raw_eval_compile_seconds=raw_eval_compile_seconds,
            train_compile_seconds=train_compile_seconds,
            eval_compile_seconds=eval_compile_seconds,
            predict_compile_seconds=predict_compile_seconds,
            validation_seconds=validation_seconds,
            total_build_seconds=time.perf_counter() - total_start,
            persistent_cache_directory=(
                None
                if cache_dir is None
                else str(Path(cache_dir).expanduser().resolve())
            ),
            validated=validated,
        )
        self.build_report = report
        return report

    def _require_built(self):
        if not self.is_built:
            raise RuntimeError("Call model.build(...) before fit/evaluate/predict.")
        assert self.graphs is not None
        assert self.parameters is not None
        assert self.optimizer_state is not None
        assert self.prepared_projection is not None
        assert self.batch_size is not None

    def _fixed_batches(
        self,
        x: np.ndarray,
        y: np.ndarray | None,
        *,
        indices: np.ndarray,
    ):
        assert self.batch_size is not None
        batch_size = self.batch_size
        for start in range(0, indices.size, batch_size):
            selected = indices[start : start + batch_size]
            count = int(selected.size)
            if count == 0:
                continue
            x_batch = np.repeat(x[selected[-1] : selected[-1] + 1], batch_size, axis=0)
            x_batch[:count] = x[selected]
            y_batch = None
            if y is not None:
                y_batch = np.repeat(
                    y[selected[-1] : selected[-1] + 1], batch_size, axis=0
                )
                y_batch[:count] = y[selected]
            valid_mask = np.zeros((batch_size,), dtype=bool)
            valid_mask[:count] = True
            yield x_batch, y_batch, valid_mask, count

    @staticmethod
    def _aggregate_metric_batches(
        metrics_with_counts: list[tuple[ProjectedMetrics, int]],
    ) -> dict[str, float]:
        if not metrics_with_counts:
            raise RuntimeError("No metric batches were produced.")
        host_metrics = jax.device_get(
            tuple(metrics for metrics, _count in metrics_with_counts)
        )
        counts = np.asarray(
            [count for _metrics, count in metrics_with_counts],
            dtype=np.float64,
        )
        total = float(np.sum(counts))
        if total <= 0.0:
            raise RuntimeError("Metric aggregation received no valid samples.")

        def weighted(name: str) -> float:
            values = np.asarray(
                [float(getattr(item, name)) for item in host_metrics],
                dtype=np.float64,
            )
            return float(np.sum(counts * values) / total)

        return {
            "loss": weighted("loss"),
            "task_loss": weighted("task_loss"),
            "raw_mse": weighted("raw_mse"),
            "consistency_loss": weighted("consistency_loss"),
            "inverse_regularization": weighted("inverse_regularization"),
            "projection_distance": weighted("projection_distance"),
            "forward_converged_fraction": weighted(
                "forward_converged_fraction"
            ),
            "forward_finite_fraction": weighted("forward_finite_fraction"),
            "gradient_eligible_fraction": weighted(
                "gradient_eligible_fraction"
            ),
            "update_accepted_fraction": weighted("update_accepted"),
            "skipped_update_fraction": 1.0 - weighted("update_accepted"),
            "max_forward_residual": float(
                np.max(
                    [float(item.max_forward_residual) for item in host_metrics]
                )
            ),
            "mean_forward_iterations": weighted("mean_forward_iterations"),
        }

    def _evaluate_compiled_graph(
        self,
        graph: Any,
        x_host: np.ndarray,
        y_host: np.ndarray | None,
    ) -> dict[str, float]:
        assert self.prepared_projection is not None
        indices = np.arange(x_host.shape[0])
        device = self.prepared_projection.device
        outputs: list[tuple[ProjectedMetrics, int]] = []
        for x_batch, y_batch, valid_mask, count in self._fixed_batches(
            x_host, y_host, indices=indices
        ):
            x_device = jax.device_put(jnp.asarray(x_batch), device)
            mask_device = jax.device_put(jnp.asarray(valid_mask), device)
            if self.mode in {"supervised", "inverse"}:
                if y_batch is None:
                    raise ValueError(f"{self.mode} evaluation requires y data.")
                metrics = graph(
                    self.parameters,
                    x_device,
                    jax.device_put(jnp.asarray(y_batch), device),
                    mask_device,
                )
            else:
                metrics = graph(self.parameters, x_device, mask_device)
            outputs.append((metrics, count))
        return self._aggregate_metric_batches(outputs)

    def fit(
        self,
        x: Any,
        y: Any | None = None,
        *,
        epochs: int,
        shuffle: bool = True,
        seed: int = 0,
        print_every: int = 0,
        pretrain_epochs: int = 0,
        eta: float | None = None,
        switch_metric: str = "raw_mse",
    ) -> list[dict[str, Any]]:
        """Train using only executables produced by :meth:`build`.

        Supervised and inverse models can begin with the compiled raw-backbone
        graph. Projection is activated for the following epoch when either the
        epoch-end raw metric reaches ``eta`` or ``pretrain_epochs`` is reached.
        The same parameter and optimizer-state buffers continue across phases.
        """

        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        x_host = np.asarray(x, dtype=np.float64)
        if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
            )
        if self.mode in {"supervised", "inverse"}:
            if y is None:
                raise ValueError(f"{self.mode} mode requires y data.")
            y_host = np.asarray(y, dtype=np.float64)
            output_dim = len(self.prepared_projection.compiled.variable_names)
            if y_host.shape != (x_host.shape[0], output_dim):
                raise ValueError(
                    f"y must have shape {(x_host.shape[0], output_dim)}; "
                    f"got {y_host.shape}."
                )
        else:
            y_host = None
        if int(epochs) < 1:
            raise ValueError("epochs must be positive.")
        if int(pretrain_epochs) < 0:
            raise ValueError("pretrain_epochs must be nonnegative.")
        if eta is not None and (not np.isfinite(float(eta)) or float(eta) < 0.0):
            raise ValueError("eta must be finite and nonnegative when supplied.")
        allowed_switch_metrics = {"raw_mse", "task_loss", "loss"}
        if switch_metric not in allowed_switch_metrics:
            raise ValueError(
                "switch_metric must be one of raw_mse, task_loss, or loss."
            )

        pretraining_requested = int(pretrain_epochs) > 0 or eta is not None
        if pretraining_requested and self.mode == "optimization":
            raise ValueError(
                "Backbone-only pretraining currently requires supervised or "
                "inverse data. Optimization mode has no raw-label loss."
            )
        if pretraining_requested:
            if self.graphs.raw_train_step is None or self.graphs.raw_eval_step is None:
                raise RuntimeError(
                    "Raw pretraining graphs were not compiled during build()."
                )

        rng = np.random.default_rng(int(seed))
        history: list[dict[str, Any]] = []
        device = self.prepared_projection.device
        indices = np.arange(x_host.shape[0])
        projection_active = not pretraining_requested
        projection_start_epoch: int | None = 1 if projection_active else None
        switch_reason: str | None = None

        for epoch in range(1, int(epochs) + 1):
            epoch_projection_active = projection_active
            if shuffle:
                rng.shuffle(indices)
            epoch_metrics: list[tuple[ProjectedMetrics, int]] = []
            epoch_start = time.perf_counter()
            for x_batch, y_batch, valid_mask, count in self._fixed_batches(
                x_host, y_host, indices=indices
            ):
                x_device = jax.device_put(jnp.asarray(x_batch), device)
                mask_device = jax.device_put(jnp.asarray(valid_mask), device)
                if self.mode in {"supervised", "inverse"}:
                    assert y_batch is not None
                    y_device = jax.device_put(jnp.asarray(y_batch), device)
                    selected_step = (
                        self.graphs.train_step
                        if epoch_projection_active
                        else self.graphs.raw_train_step
                    )
                    assert selected_step is not None
                    self.parameters, self.optimizer_state, metrics = selected_step(
                        self.parameters,
                        self.optimizer_state,
                        x_device,
                        y_device,
                        mask_device,
                    )
                else:
                    self.parameters, self.optimizer_state, metrics = (
                        self.graphs.train_step(
                            self.parameters,
                            self.optimizer_state,
                            x_device,
                            mask_device,
                        )
                    )
                epoch_metrics.append((metrics, count))

            row = self._aggregate_metric_batches(epoch_metrics)
            if not epoch_projection_active:
                assert self.graphs.raw_eval_step is not None
                row = self._evaluate_compiled_graph(
                    self.graphs.raw_eval_step, x_host, y_host
                )
            row.update(
                {
                    "epoch": epoch,
                    "phase": "projected" if epoch_projection_active else "pretrain",
                    "projection_active": int(epoch_projection_active),
                    "epoch_seconds": float(time.perf_counter() - epoch_start),
                    "projection_start_epoch": (
                        -1 if projection_start_epoch is None else projection_start_epoch
                    ),
                    "switch_reason": "" if switch_reason is None else switch_reason,
                }
            )
            history.append(row)

            should_print = print_every and (
                epoch == 1
                or epoch == int(epochs)
                or epoch % int(print_every) == 0
            )
            if should_print:
                print(
                    "projected_fit "
                    f"epoch={epoch} phase={row['phase']} "
                    f"loss={row['loss']:.6e} raw_mse={row['raw_mse']:.6e} "
                    f"projection_distance={row['projection_distance']:.3e} "
                    f"converged={row['forward_converged_fraction']:.3f} "
                    f"eligible={row['gradient_eligible_fraction']:.3f} "
                    f"updates={row['update_accepted_fraction']:.3f}"
                )

            if not epoch_projection_active:
                reached_eta = eta is not None and row[switch_metric] <= float(eta)
                reached_epoch_limit = (
                    int(pretrain_epochs) > 0 and epoch >= int(pretrain_epochs)
                )
                if reached_eta or reached_epoch_limit:
                    projection_active = True
                    if epoch < int(epochs):
                        projection_start_epoch = epoch + 1
                    reasons = []
                    if reached_eta:
                        reasons.append("eta")
                    if reached_epoch_limit:
                        reasons.append("pretrain_epochs")
                    switch_reason = "+".join(reasons)
                    if projection_start_epoch is not None:
                        print(
                            "Activating projection at "
                            f"epoch {projection_start_epoch} "
                            f"(trigger: {switch_reason})."
                        )
        return history

    def evaluate(
        self,
        x: Any,
        y: Any | None = None,
        *,
        use_projection: bool = True,
    ) -> dict[str, float]:
        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        x_host = np.asarray(x, dtype=np.float64)
        if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
            )
        y_host = None if y is None else np.asarray(y, dtype=np.float64)
        if use_projection:
            graph = self.graphs.eval_step
        else:
            if self.graphs.raw_eval_step is None:
                raise RuntimeError(
                    "Raw evaluation is unavailable for this model mode."
                )
            graph = self.graphs.raw_eval_step
        return self._evaluate_compiled_graph(graph, x_host, y_host)

    def evaluate_raw(self, x: Any, y: Any) -> dict[str, float]:
        """Evaluate the backbone without invoking the projection solver."""

        return self.evaluate(x, y, use_projection=False)

    def predict(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        self._require_built()
        assert self.graphs is not None
        assert self.prepared_projection is not None
        x_host = np.asarray(x, dtype=np.float64)
        squeeze = x_host.ndim == 1
        if squeeze:
            x_host = x_host.reshape(1, -1)
        if x_host.ndim != 2 or x_host.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (N, {self.input_dim}); got {x_host.shape}."
            )
        indices = np.arange(x_host.shape[0])
        device = self.prepared_projection.device
        hats = []
        projected = []
        for x_batch, _y_batch, _valid_mask, count in self._fixed_batches(
            x_host, None, indices=indices
        ):
            result = self.graphs.predict_step(
                self.parameters,
                jax.device_put(jnp.asarray(x_batch), device),
            )
            result_host = jax.device_get(result)
            hats.append(np.asarray(result_host.y_hat[:count]))
            projected.append(np.asarray(result_host.y_tilde[:count]))
        y_hat = np.concatenate(hats, axis=0)
        y_tilde = np.concatenate(projected, axis=0)
        if squeeze:
            return y_hat[0], y_tilde[0]
        return y_hat, y_tilde

    def predict_raw(self, x: Any) -> np.ndarray:
        self._require_built()
        assert self.graphs is not None
        if self.graphs.raw_predict_step is None:
            raise RuntimeError("Raw prediction was not compiled during build().")
        x_host = np.asarray(x, dtype=np.float64)
        squeeze = x_host.ndim == 1
        if squeeze:
            x_host = x_host.reshape(1, -1)
        indices = np.arange(x_host.shape[0])
        assert self.prepared_projection is not None
        device = self.prepared_projection.device
        outputs = []
        for x_batch, _y_batch, _mask, count in self._fixed_batches(
            x_host, None, indices=indices
        ):
            result = self.graphs.raw_predict_step(
                self.parameters,
                jax.device_put(jnp.asarray(x_batch), device),
            )
            outputs.append(np.asarray(jax.device_get(result[:count])))
        merged = np.concatenate(outputs, axis=0)
        return merged[0] if squeeze else merged
