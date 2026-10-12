"""Public projected-neural model façade with additive logging/serialization."""
from __future__ import annotations

import contextlib
import io
import re
import sys
from typing import Any

from .logging import KKTTableLogger
from .projected_neural import ProjectedNeuralModel as _CoreProjectedNeuralModel


_EPOCH_RE = re.compile(
    r"projected_fit epoch=(?P<epoch>\d+) phase=(?P<phase>\S+) "
    r"loss=(?P<loss>[-+0-9.eEnNaA]+) raw_mse=(?P<raw>[-+0-9.eEnNaA]+) "
    r"projection_distance=(?P<distance>[-+0-9.eEnNaA]+) "
    r"converged=(?P<converged>[-+0-9.eEnNaA]+) "
    r"eligible=(?P<eligible>[-+0-9.eEnNaA]+) "
    r"updates=(?P<updates>[-+0-9.eEnNaA]+)"
)
_SWITCH_RE = re.compile(
    r"Activating projection at epoch (?P<epoch>\d+) \(trigger: (?P<reason>.+)\)\."
)


class _KKTLogStream(io.TextIOBase):
    def __init__(self, logger: KKTTableLogger, passthrough, validation_callback=None):
        self.logger = logger
        self.passthrough = passthrough
        self.validation_callback = validation_callback
        self.buffer = ""

    def writable(self):
        return True

    def write(self, text):
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line)
        return len(text)

    def flush(self):
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""

    def _line(self, line: str):
        match = _EPOCH_RE.fullmatch(line.strip())
        if match:
            values = match.groupdict()
            validation = None
            if self.validation_callback is not None:
                validation = self.validation_callback(values["phase"] == "projected")
            self.logger.epoch(
                int(values["epoch"]),
                {
                    "loss": float(values["loss"]),
                    "forward_converged_fraction": float(values["converged"]),
                    "gradient_eligible_fraction": float(values["eligible"]),
                },
                validation,
            )
            return
        match = _SWITCH_RE.fullmatch(line.strip())
        if match:
            self.logger.phase_switch(
                int(match.group("epoch")), match.group("reason")
            )
            return
        if line.strip():
            print(line, file=self.passthrough)


class ProjectedNeuralModel(_CoreProjectedNeuralModel):
    """The unchanged core model plus opt-in public conveniences.

    Numerical graph construction, training steps, sparse projection and custom
    VJP execution are inherited unchanged. This façade only formats already
    produced host-side log lines and adds post-training serialization helpers.
    """

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
        validation_data: tuple[Any, Any | None] | None = None,
        log_style: str = "kkt",
    ):
        if log_style not in {"kkt", "compact", "none"}:
            raise ValueError("log_style must be kkt, compact, or none")
        if log_style == "compact":
            history = super().fit(
                x,
                y,
                epochs=epochs,
                shuffle=shuffle,
                seed=seed,
                print_every=print_every,
                pretrain_epochs=pretrain_epochs,
                eta=eta,
                switch_metric=switch_metric,
            )
        elif log_style == "none" or not print_every:
            history = super().fit(
                x,
                y,
                epochs=epochs,
                shuffle=shuffle,
                seed=seed,
                print_every=0,
                pretrain_epochs=pretrain_epochs,
                eta=eta,
                switch_metric=switch_metric,
            )
        else:
            self._require_built()
            assert self.prepared_projection is not None
            compiled = self.prepared_projection.compiled
            logger = KKTTableLogger(epochs=int(epochs), print_every=int(print_every))
            logger.start(
                dims={
                    "n_x": self.input_dim,
                    "n_y": len(compiled.variable_names),
                    "n_eq": int(compiled.initial_state.y_c.shape[1]),
                    "n_ineq": int(compiled.initial_state.y_d.shape[1]),
                },
                task=self.mode,
                mode="inverse" if self.mode == "inverse" else "forward",
                samples=len(x),
                validation_samples=(
                    0 if validation_data is None else len(validation_data[0])
                ),
                batch_size=self.batch_size,
                network=getattr(self.backbone_apply, "__qualname__", "functional"),
                eta=eta,
                pretrain_epochs=int(pretrain_epochs),
                consistency_weight=self.consistency_weight,
            )
            def validation_callback(use_projection):
                if validation_data is None:
                    return None
                return self.evaluate(
                    validation_data[0],
                    validation_data[1],
                    use_projection=use_projection,
                )

            stream = _KKTLogStream(
                logger,
                sys.stdout,
                validation_callback=validation_callback,
            )
            with contextlib.redirect_stdout(stream):
                history = super().fit(
                    x,
                    y,
                    epochs=epochs,
                    shuffle=shuffle,
                    seed=seed,
                    print_every=print_every,
                    pretrain_epochs=pretrain_epochs,
                    eta=eta,
                    switch_metric=switch_metric,
                )
            stream.flush()
            logger.finish()

        if validation_data is not None and history:
            validation = self.evaluate(validation_data[0], validation_data[1])
            history[-1].update(
                {f"final_validation_{key}": value for key, value in validation.items()}
            )
        return history

    def save_predictor(
        self,
        directory,
        *,
        network_spec: dict[str, Any],
        projection_spec: dict[str, Any] | None = None,
        problem=None,
        input_transform: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        predictor_factory: str | None = None,
        factory_config: dict[str, Any] | None = None,
        export_portable: bool = False,
        export_native: bool = False,
        export_platforms=None,
        strict_export: bool = False,
    ):
        self._require_built()
        assert self.prepared_projection is not None
        from ..serialization import save_predictor

        serialized_network_spec = dict(network_spec)
        if (
            isinstance(self.parameters, dict)
            and "backbone" in self.parameters
        ):
            serialized_network_spec.setdefault("parameter_path", ["backbone"])

        return save_predictor(
            directory,
            network_spec=serialized_network_spec,
            params=self.parameters,
            input_size=self.input_dim,
            output_size=len(self.prepared_projection.compiled.variable_names),
            batch_size=self.batch_size,
            projection_spec=projection_spec,
            problem=problem,
            input_transform=input_transform,
            dtype="float64",
            backend=self.prepared_projection.backend,
            metadata=metadata,
            predictor_factory=predictor_factory,
            factory_config=factory_config,
            export_portable=export_portable,
            export_native=export_native,
            export_platforms=export_platforms,
            strict_export=strict_export,
        )
