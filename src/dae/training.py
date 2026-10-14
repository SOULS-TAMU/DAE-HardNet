"""End-to-end projected-neural training for compiled DAE systems."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from daehn.train.projected_neural import Adam, MLPBackbone, apply_updates
from .logging import DAETrainingLogger
from .model import (
    FullModelDerivativePlan,
    ProjectedDAEModel,
    extract_network_derivatives,
)

Array = Any
PyTree = Any


class DAETrainMetrics(NamedTuple):
    loss: Array
    data_loss: Array
    state_mse: Array
    derivative_loss: Array
    derivative_consistency_loss: Array
    violation_loss: Array
    violation: Array
    differential_violation: Array
    algebraic_violation: Array
    inequality_violation: Array
    consistency_loss: Array
    initial_loss: Array
    boundary_loss: Array
    forward_converged_fraction: Array
    gradient_eligible_fraction: Array
    mean_forward_iterations: Array
    update_accepted: Array


@dataclass(frozen=True)
class DAETrainingConfig:
    epochs: int = 200
    pretrain_epochs: int = 25
    eta: float | None = 1e-4
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (32, 32)
    batch_size: int = 16
    derivative_weight: float = 1.0
    derivative_consistency_weight: float = 0.0
    violation_weight: float = 1.0
    consistency_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    full_model_derivative_steps: tuple[float, ...] | None = None
    print_every: int = 10
    seed: int = 0
    shuffle: bool = True

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.eta is not None and self.eta < 0:
            raise ValueError("eta must be nonnegative")
        for name in (
            "derivative_weight",
            "derivative_consistency_weight",
            "violation_weight",
            "consistency_weight",
            "initial_weight",
            "boundary_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.full_model_derivative_steps is not None:
            if not self.full_model_derivative_steps:
                raise ValueError("full_model_derivative_steps cannot be empty")
            if any(float(value) <= 0.0 for value in self.full_model_derivative_steps):
                raise ValueError("full-model derivative steps must be positive")


@dataclass(frozen=True)
class DAETrainingResult:
    parameters: PyTree
    history: tuple[dict[str, Any], ...]
    projection_start_epoch: int | None
    switch_reason: str | None
    wall_seconds: float


def _tree_all_finite(tree: PyTree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    result = jnp.asarray(True)
    for leaf in leaves:
        result = result & jnp.all(jnp.isfinite(leaf))
    return result


def _tree_select(condition: Array, candidate: PyTree, original: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(condition, new, old), candidate, original
    )


def _masked_mean(values: Array, mask: Array) -> Array:
    values = jnp.asarray(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _fixed_batches(
    coordinates: np.ndarray,
    parameters: np.ndarray,
    targets: np.ndarray,
    derivative_targets: np.ndarray,
    *,
    batch_size: int,
    indices: np.ndarray | None = None,
):
    if indices is None:
        indices = np.arange(coordinates.shape[0])
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        count = len(selected)
        if count == 0:
            continue
        x = np.zeros((batch_size, coordinates.shape[1]), dtype=np.float64)
        p = np.zeros((batch_size, parameters.shape[1]), dtype=np.float64)
        y = np.zeros((batch_size, targets.shape[1]), dtype=np.float64)
        d = np.zeros((batch_size, derivative_targets.shape[1]), dtype=np.float64)
        mask = np.zeros((batch_size,), dtype=bool)
        x[:count] = coordinates[selected]
        p[:count] = parameters[selected]
        y[:count] = targets[selected]
        d[:count] = derivative_targets[selected]
        mask[:count] = True
        if count < batch_size:
            x[count:] = x[count - 1]
            p[count:] = p[count - 1]
            y[count:] = y[count - 1]
            d[count:] = d[count - 1]
        yield x, p, y, d, mask, count


def _aggregate(rows: list[tuple[dict[str, float], int]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("no metric batches were produced")
    total = float(sum(count for _row, count in rows))
    keys = rows[0][0].keys()
    result = {}
    for key in keys:
        if key.startswith("max_"):
            result[key] = max(float(row[key]) for row, _count in rows)
        else:
            result[key] = sum(float(row[key]) * count for row, count in rows) / total
    return result


class DAETrainer:
    """Train a neural DAE surrogate using raw pretraining then projection.

    Violation metrics are always computed from ``DAESystem`` differential,
    algebraic, and inequality functions. They are not the scaled solver/KKT
    residuals used internally by the discretized projection.
    """

    def __init__(
        self,
        *,
        compiled,
        prepared_projection,
        backbone: MLPBackbone,
        config: DAETrainingConfig,
        initial_coordinates: np.ndarray | None = None,
        initial_targets: np.ndarray | None = None,
        boundary_coordinates: np.ndarray | None = None,
        boundary_targets: np.ndarray | None = None,
    ) -> None:
        self.compiled = compiled
        self.prepared_projection = prepared_projection
        self.backbone = backbone
        self.config = config
        self.model = ProjectedDAEModel(compiled, backbone.apply, prepared_projection)
        ncoords = len(compiled.system.coordinates)
        nstates = len(compiled.system.state_names)
        if config.full_model_derivative_steps is None:
            offsets = np.abs(np.asarray(compiled.discretization.offsets, dtype=np.float64))
            inferred_steps = []
            for coordinate in range(ncoords):
                nonzero = offsets[:, coordinate][offsets[:, coordinate] > 0.0]
                inferred_steps.append(float(np.min(nonzero)) if nonzero.size else 1.0e-3)
            derivative_steps = tuple(inferred_steps)
        else:
            derivative_steps = tuple(config.full_model_derivative_steps)
        if len(derivative_steps) != ncoords:
            raise ValueError("one full-model derivative step is required per coordinate")
        self.full_model_derivative_plan = FullModelDerivativePlan.build(
            compiled.discretization.derivatives,
            compiled.system.differential_states,
            derivative_steps,
        )
        derivative_batch_size = (
            config.batch_size * int(self.full_model_derivative_plan.shifts.shape[0])
        )
        derivative_projection = compiled.prepare_projection(
            batch_size=derivative_batch_size,
            method=prepared_projection.method,
            config=prepared_projection.config,
            dtype=prepared_projection.dtype,
            backend=prepared_projection.backend,
        )
        self.full_model_derivative_model = ProjectedDAEModel(
            compiled, backbone.apply, derivative_projection
        )

        def empty_coordinates(value):
            if value is None:
                return np.empty((0, ncoords), dtype=np.float64)
            return np.asarray(value, dtype=np.float64).reshape((-1, ncoords))

        def empty_targets(value):
            if value is None:
                return np.empty((0, nstates), dtype=np.float64)
            return np.asarray(value, dtype=np.float64).reshape((-1, nstates))

        self.initial_coordinates = jnp.asarray(empty_coordinates(initial_coordinates))
        self.initial_targets = jnp.asarray(empty_targets(initial_targets))
        self.boundary_coordinates = jnp.asarray(empty_coordinates(boundary_coordinates))
        self.boundary_targets = jnp.asarray(empty_targets(boundary_targets))

        self.optimizer = Adam(learning_rate=config.learning_rate)
        self._build_steps()

    def _condition_losses(self, parameters):
        initial = jnp.asarray(0.0, dtype=jnp.float64)
        if self.initial_coordinates.shape[0]:
            prediction = self.backbone.apply(parameters, self.initial_coordinates)
            initial = jnp.mean((prediction - self.initial_targets) ** 2)
        boundary = jnp.asarray(0.0, dtype=jnp.float64)
        if self.boundary_coordinates.shape[0]:
            prediction = self.backbone.apply(parameters, self.boundary_coordinates)
            boundary = jnp.mean((prediction - self.boundary_targets) ** 2)
        return initial, boundary

    def _physical_blocks(self, coordinates, states, derivatives, physical_parameters):
        system = self.compiled.system
        differential = jax.vmap(system.differential_equations.function)(
            coordinates, states, derivatives, physical_parameters
        )
        algebraic = jax.vmap(system.algebraic_equalities.function)(
            coordinates, states, derivatives, physical_parameters
        )
        inequalities = jax.vmap(system.algebraic_inequalities.function)(
            coordinates, states, derivatives, physical_parameters
        )
        differential = jnp.asarray(differential).reshape((coordinates.shape[0], -1))
        algebraic = jnp.asarray(algebraic).reshape((coordinates.shape[0], -1))
        inequalities = jnp.asarray(inequalities).reshape((coordinates.shape[0], -1))
        positive_inequalities = jnp.maximum(inequalities, 0.0)
        combined = jnp.concatenate(
            (differential, algebraic, positive_inequalities), axis=-1
        )
        return differential, algebraic, positive_inequalities, combined

    def _raw_outputs(self, parameters, coordinates, physical_parameters):
        states = self.backbone.apply(parameters, coordinates)
        derivatives = extract_network_derivatives(
            self.backbone.apply,
            parameters,
            coordinates,
            self.compiled.discretization.derivatives,
            self.compiled.system.differential_states,
        )
        blocks = self._physical_blocks(
            coordinates, states, derivatives, physical_parameters
        )
        return states, derivatives, blocks

    @staticmethod
    def _violation_metrics(blocks, mask):
        differential, algebraic, inequalities, combined = blocks

        def mean_norm(values):
            if values.shape[-1] == 0:
                return jnp.asarray(0.0, dtype=values.dtype)
            return _masked_mean(jnp.linalg.norm(values, axis=-1), mask)

        return (
            mean_norm(combined),
            mean_norm(differential),
            mean_norm(algebraic),
            mean_norm(inequalities),
        )

    def _build_steps(self) -> None:
        config = self.config

        def derivative_mse(predicted, targets, mask):
            if predicted.shape[-1] == 0:
                return jnp.asarray(0.0, dtype=predicted.dtype)
            return _masked_mean(jnp.mean((predicted - targets) ** 2, axis=-1), mask)

        def violation_mse(blocks, mask):
            combined = blocks[-1]
            if combined.shape[-1] == 0:
                return jnp.asarray(0.0, dtype=jnp.float64)
            return _masked_mean(jnp.mean(combined**2, axis=-1), mask)

        def raw_compute(
            parameters,
            coordinates,
            physical_parameters,
            targets,
            derivative_targets,
            mask,
        ):
            states, derivatives, blocks = self._raw_outputs(
                parameters, coordinates, physical_parameters
            )
            per_sample_mse = jnp.mean((states - targets) ** 2, axis=-1)
            data_loss = _masked_mean(per_sample_mse, mask)
            derivative_loss = derivative_mse(derivatives, derivative_targets, mask)
            residual_loss = violation_mse(blocks, mask)
            initial_loss, boundary_loss = self._condition_losses(parameters)
            loss = (
                data_loss
                + config.derivative_weight * derivative_loss
                + config.violation_weight * residual_loss
                + config.initial_weight * initial_loss
                + config.boundary_weight * boundary_loss
            )
            violation, diff_v, alg_v, ineq_v = self._violation_metrics(blocks, mask)
            zero = jnp.asarray(0.0, dtype=states.dtype)
            one = jnp.asarray(1.0, dtype=states.dtype)
            metrics = DAETrainMetrics(
                loss,
                data_loss,
                data_loss,
                derivative_loss,
                zero,
                residual_loss,
                violation,
                diff_v,
                alg_v,
                ineq_v,
                zero,
                initial_loss,
                boundary_loss,
                one,
                one,
                zero,
                one,
            )
            batch = coordinates.shape[0]
            return (
                loss,
                metrics,
                states,
                derivatives,
                derivatives,
                jnp.ones((batch,), dtype=jnp.bool_),
                jnp.ones((batch,), dtype=jnp.bool_),
                jnp.zeros((batch,), dtype=jnp.int32),
                jnp.zeros((batch,), dtype=states.dtype),
            )

        def projected_compute(
            parameters,
            coordinates,
            physical_parameters,
            targets,
            derivative_targets,
            mask,
        ):
            decoded, info, _context, target = self.model.forward(
                parameters, coordinates, physical_parameters
            )
            states = decoded["states"]
            projected_derivatives = decoded["derivatives"]
            variables = decoded["normalized_variables"]
            finite_mask = mask & info.forward_state_finite
            eligible_mask = mask & info.gradient_eligible

            # Recompute derivatives from the complete NN -> projection ->
            # reconstruction mapping.  These values, rather than the independent
            # projection derivative variables, define derivative supervision and
            # the physical differential-equation residual.
            model_derivatives = self.full_model_derivative_plan.evaluate(
                self.full_model_derivative_model,
                parameters,
                coordinates,
                physical_parameters,
                center_states=states,
            )

            per_sample_mse = jnp.mean((states - targets) ** 2, axis=-1)
            data_loss = _masked_mean(per_sample_mse, finite_mask)
            derivative_loss = derivative_mse(
                model_derivatives, derivative_targets, finite_mask
            )
            derivative_consistency_loss = derivative_mse(
                model_derivatives, projected_derivatives, eligible_mask
            )
            consistency_per_sample = jnp.mean((variables - target) ** 2, axis=-1)
            consistency_loss = _masked_mean(consistency_per_sample, eligible_mask)
            initial_loss, boundary_loss = self._condition_losses(parameters)
            blocks = self._physical_blocks(
                coordinates, states, model_derivatives, physical_parameters
            )
            residual_loss = violation_mse(blocks, finite_mask)
            violation, diff_v, alg_v, ineq_v = self._violation_metrics(
                blocks, finite_mask
            )
            loss = (
                data_loss
                + config.derivative_weight * derivative_loss
                + config.derivative_consistency_weight
                * derivative_consistency_loss
                + config.violation_weight * residual_loss
                + config.consistency_weight * consistency_loss
                + config.initial_weight * initial_loss
                + config.boundary_weight * boundary_loss
            )
            converged = _masked_mean(
                info.forward_converged.astype(jnp.float64), mask
            )
            eligible = _masked_mean(
                info.gradient_eligible.astype(jnp.float64), mask
            )
            iterations = _masked_mean(
                info.forward_iterations.astype(jnp.float64), mask
            )
            metrics = DAETrainMetrics(
                loss,
                data_loss,
                data_loss,
                derivative_loss,
                derivative_consistency_loss,
                residual_loss,
                violation,
                diff_v,
                alg_v,
                ineq_v,
                consistency_loss,
                initial_loss,
                boundary_loss,
                converged,
                eligible,
                iterations,
                jnp.asarray(1.0),
            )
            return (
                loss,
                metrics,
                states,
                model_derivatives,
                projected_derivatives,
                info.forward_converged,
                info.gradient_eligible,
                info.forward_iterations,
                info.forward_residual_inf_norm,
            )

        def raw_objective(
            parameters,
            coordinates,
            physical_parameters,
            targets,
            derivative_targets,
            mask,
        ):
            loss, metrics, *_rest = raw_compute(
                parameters,
                coordinates,
                physical_parameters,
                targets,
                derivative_targets,
                mask,
            )
            return loss, metrics

        def projected_objective(
            parameters,
            coordinates,
            physical_parameters,
            targets,
            derivative_targets,
            mask,
        ):
            loss, metrics, *_rest = projected_compute(
                parameters,
                coordinates,
                physical_parameters,
                targets,
                derivative_targets,
                mask,
            )
            return loss, metrics

        def make_train_step(objective):
            def step(
                parameters,
                optimizer_state,
                coordinates,
                physical_parameters,
                targets,
                derivative_targets,
                mask,
            ):
                (loss, metrics), gradients = jax.value_and_grad(
                    objective, has_aux=True
                )(
                    parameters,
                    coordinates,
                    physical_parameters,
                    targets,
                    derivative_targets,
                    mask,
                )
                updates, candidate_optimizer_state = self.optimizer.update(
                    gradients, optimizer_state, parameters
                )
                candidate_parameters = apply_updates(parameters, updates)
                accepted = (
                    jnp.isfinite(loss)
                    & _tree_all_finite(gradients)
                    & _tree_all_finite(candidate_parameters)
                )
                next_parameters = _tree_select(
                    accepted, candidate_parameters, parameters
                )
                next_optimizer_state = _tree_select(
                    accepted, candidate_optimizer_state, optimizer_state
                )
                metrics = metrics._replace(update_accepted=accepted.astype(jnp.float64))
                return next_parameters, next_optimizer_state, metrics

            return jax.jit(step)

        self.raw_train_step = make_train_step(raw_objective)
        self.projected_train_step = make_train_step(projected_objective)
        # Evaluation returns both metrics and predictions. These graphs are
        # compiled during epoch validation and reused for final postprocessing.
        self.raw_eval_step = jax.jit(raw_compute)
        self.projected_eval_step = jax.jit(projected_compute)

    @staticmethod
    def _metrics_to_dict(metrics: DAETrainMetrics) -> dict[str, float]:
        host = jax.device_get(metrics)
        return {name: float(value) for name, value in zip(metrics._fields, host)}

    def _evaluate_batches(
        self,
        parameters,
        coordinates,
        physical_parameters,
        targets,
        derivative_targets,
        *,
        projected: bool,
    ) -> dict[str, float]:
        graph = self.projected_eval_step if projected else self.raw_eval_step
        rows = []
        for x, p, y, d, mask, count in _fixed_batches(
            coordinates,
            physical_parameters,
            targets,
            derivative_targets,
            batch_size=self.config.batch_size,
        ):
            output = graph(
                parameters,
                jnp.asarray(x),
                jnp.asarray(p),
                jnp.asarray(y),
                jnp.asarray(d),
                jnp.asarray(mask),
            )
            rows.append((self._metrics_to_dict(output[1]), count))
        return _aggregate(rows)

    def fit(
        self,
        train_coordinates,
        train_parameters,
        train_targets,
        train_derivative_targets,
        validation_coordinates,
        validation_parameters,
        validation_targets,
        validation_derivative_targets,
        *,
        initial_parameters: PyTree | None = None,
    ) -> DAETrainingResult:
        config = self.config
        train_coordinates = np.asarray(train_coordinates, dtype=np.float64)
        train_parameters = np.asarray(train_parameters, dtype=np.float64)
        train_targets = np.asarray(train_targets, dtype=np.float64)
        train_derivative_targets = np.asarray(
            train_derivative_targets, dtype=np.float64
        )
        validation_coordinates = np.asarray(validation_coordinates, dtype=np.float64)
        validation_parameters = np.asarray(validation_parameters, dtype=np.float64)
        validation_targets = np.asarray(validation_targets, dtype=np.float64)
        validation_derivative_targets = np.asarray(
            validation_derivative_targets, dtype=np.float64
        )

        if train_derivative_targets.shape != (
            train_coordinates.shape[0],
            len(self.compiled.discretization.derivatives),
        ):
            raise ValueError("training derivative targets have the wrong shape")
        if validation_derivative_targets.shape != (
            validation_coordinates.shape[0],
            len(self.compiled.discretization.derivatives),
        ):
            raise ValueError("validation derivative targets have the wrong shape")

        if initial_parameters is None:
            parameters = self.backbone.init(
                jax.random.PRNGKey(config.seed),
                train_coordinates.shape[1],
                train_targets.shape[1],
                jnp.float64,
            )
        else:
            parameters = initial_parameters
        optimizer_state = self.optimizer.init(parameters)

        pretraining_requested = config.pretrain_epochs > 0 or config.eta is not None
        projection_active = not pretraining_requested
        projection_start_epoch = 1 if projection_active else None
        switch_reason = None
        rng = np.random.default_rng(config.seed)
        indices = np.arange(train_coordinates.shape[0])
        history = []
        logger = DAETrainingLogger(config.epochs, config.print_every)
        logger.start(
            system=self.compiled.system.name,
            discretization=self.compiled.discretization.name,
            projection=self.prepared_projection.method,
            samples=len(train_coordinates),
            validation_samples=len(validation_coordinates),
            batch_size=config.batch_size,
            network=f"MLP{config.hidden_sizes}",
            pretrain_epochs=config.pretrain_epochs,
            eta=config.eta,
        )
        start_time = time.perf_counter()

        for epoch in range(1, config.epochs + 1):
            epoch_projected = projection_active
            if config.shuffle:
                rng.shuffle(indices)
            step = self.projected_train_step if epoch_projected else self.raw_train_step
            rows = []
            for x, p, y, d, mask, count in _fixed_batches(
                train_coordinates,
                train_parameters,
                train_targets,
                train_derivative_targets,
                batch_size=config.batch_size,
                indices=indices,
            ):
                parameters, optimizer_state, metrics = step(
                    parameters,
                    optimizer_state,
                    jnp.asarray(x),
                    jnp.asarray(p),
                    jnp.asarray(y),
                    jnp.asarray(d),
                    jnp.asarray(mask),
                )
                rows.append((self._metrics_to_dict(metrics), count))
            train_metrics = _aggregate(rows)
            validation_metrics = self._evaluate_batches(
                parameters,
                validation_coordinates,
                validation_parameters,
                validation_targets,
                validation_derivative_targets,
                projected=epoch_projected,
            )
            row = {
                "epoch": epoch,
                "phase": "projected" if epoch_projected else "pretrain",
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            }
            history.append(row)
            logger.epoch(epoch, train_metrics, validation_metrics)

            if not projection_active:
                reason = None
                if config.eta is not None and train_metrics["state_mse"] <= config.eta:
                    reason = f"state_mse <= eta ({train_metrics['state_mse']:.4e})"
                elif config.pretrain_epochs and epoch >= config.pretrain_epochs:
                    reason = f"pretrain_epochs={config.pretrain_epochs}"
                if reason is not None and epoch < config.epochs:
                    projection_active = True
                    projection_start_epoch = epoch + 1
                    switch_reason = reason
                    logger.phase_switch(projection_start_epoch, reason)

        # Block once at the end so wall time includes real device execution.
        jax.block_until_ready(jax.tree_util.tree_leaves(parameters)[0])
        wall_seconds = time.perf_counter() - start_time
        logger.finish(wall_seconds)
        return DAETrainingResult(
            parameters=parameters,
            history=tuple(history),
            projection_start_epoch=projection_start_epoch,
            switch_reason=switch_reason,
            wall_seconds=wall_seconds,
        )

    def evaluate(
        self,
        parameters,
        coordinates,
        physical_parameters,
        targets,
        derivative_targets,
        *,
        projected: bool = True,
    ) -> dict[str, float]:
        return self._evaluate_batches(
            parameters,
            np.asarray(coordinates, dtype=np.float64),
            np.asarray(physical_parameters, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            np.asarray(derivative_targets, dtype=np.float64),
            projected=projected,
        )

    def predict(self, parameters, coordinates, physical_parameters):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        physical_parameters = np.asarray(physical_parameters, dtype=np.float64)
        raw_outputs = []
        projected_outputs = []
        derivative_outputs = []
        projected_derivative_outputs = []
        info_rows = []
        dummy_targets = np.zeros(
            (coordinates.shape[0], len(self.compiled.system.state_names)),
            dtype=np.float64,
        )
        dummy_derivatives = np.zeros(
            (coordinates.shape[0], len(self.compiled.discretization.derivatives)),
            dtype=np.float64,
        )
        for x, p, _y, _d, mask, count in _fixed_batches(
            coordinates,
            physical_parameters,
            dummy_targets,
            dummy_derivatives,
            batch_size=self.config.batch_size,
        ):
            xj = jnp.asarray(x)
            pj = jnp.asarray(p)
            maskj = jnp.asarray(mask)
            targetj = jnp.asarray(_y)
            derivative_targetj = jnp.asarray(_d)
            raw_result = self.raw_eval_step(
                parameters, xj, pj, targetj, derivative_targetj, maskj
            )
            projected_result = self.projected_eval_step(
                parameters, xj, pj, targetj, derivative_targetj, maskj
            )
            raw_outputs.append(np.asarray(raw_result[2])[:count])
            projected_outputs.append(np.asarray(projected_result[2])[:count])
            derivative_outputs.append(np.asarray(projected_result[3])[:count])
            projected_derivative_outputs.append(
                np.asarray(projected_result[4])[:count]
            )
            info_rows.append(
                {
                    "converged": np.asarray(projected_result[5])[:count],
                    "eligible": np.asarray(projected_result[6])[:count],
                    "iterations": np.asarray(projected_result[7])[:count],
                    "residual": np.asarray(projected_result[8])[:count],
                }
            )
        info = {
            key: np.concatenate([row[key] for row in info_rows], axis=0)
            for key in info_rows[0]
        }
        return (
            np.concatenate(raw_outputs, axis=0),
            np.concatenate(projected_outputs, axis=0),
            np.concatenate(derivative_outputs, axis=0),
            np.concatenate(projected_derivative_outputs, axis=0),
            info,
        )
