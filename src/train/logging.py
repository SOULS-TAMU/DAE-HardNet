"""KKT-HardNet-style fixed-width training logs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class KKTTableLogger:
    epochs: int
    print_every: int = 1
    width: int = 13
    title: str = "DAEHN"
    _started: bool = False
    _has_validation: bool = False

    def start(
        self,
        *,
        dims: dict,
        task: str,
        mode: str,
        samples: int,
        batch_size: int,
        network: str,
        eta,
        pretrain_epochs: int,
        consistency_weight: float,
        validation_samples: int = 0,
    ):
        if not self.print_every:
            return
        self._has_validation = int(validation_samples) > 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Model training started! [{now}]\n")
        print(self.title)
        print(
            "  dims: "
            f"n_x={dims.get('n_x')} n_y={dims.get('n_y')} "
            f"n_eq={dims.get('n_eq')} n_ineq={dims.get('n_ineq')}"
        )
        print(f"  task: {task}")
        print(f"  mode: {mode}")
        print(
            f"  samples: train={samples} val={validation_samples} "
            f"batch_size={batch_size}"
        )
        print(f"  network: {network}")
        print(f"  eta: {'n/a' if eta is None else eta}")
        print(f"  epoch_mlp: {pretrain_epochs if pretrain_epochs else 'n/a'}")
        print(f"  cons_alpha: {consistency_weight}")
        print("=" * 120)
        print(
            f"{'Epoch':>{self.width}} | "
            f"{'Training':^{self.width * 3 + 2}} | "
            f"{'Validation':^{self.width * 3 + 2}}"
        )
        print("-" * 120)
        print(
            f"{'':>{self.width}} | "
            f"{'Loss':>{self.width}} {'Converged':>{self.width}} "
            f"{'Eligible':>{self.width}} | "
            f"{'Loss':>{self.width}} {'Converged':>{self.width}} "
            f"{'Eligible':>{self.width}}"
        )
        print("=" * 120)
        self._started = True

    def should_print(self, epoch: int) -> bool:
        return bool(self.print_every) and (
            epoch == 1
            or epoch == self.epochs
            or epoch % max(1, int(self.print_every)) == 0
        )

    def epoch(self, epoch: int, train: dict, validation: dict | None = None):
        if not self.should_print(epoch):
            return
        validation = validation or {}

        def metric(row, name, default=0.0):
            value = row.get(name, default)
            return f"{float(value):>{self.width}.4e}"

        def unavailable():
            return f"{'n/a':>{self.width}}"

        if self._has_validation and validation:
            validation_columns = (
                f"{metric(validation, 'loss')} "
                f"{metric(validation, 'forward_converged_fraction')} "
                f"{metric(validation, 'gradient_eligible_fraction')}"
            )
        else:
            validation_columns = (
                f"{unavailable()} {unavailable()} {unavailable()}"
            )

        label = f"{epoch:03d}/{self.epochs:03d}"
        print(
            f"{label:>{self.width}} | "
            f"{metric(train, 'loss')} "
            f"{metric(train, 'forward_converged_fraction')} "
            f"{metric(train, 'gradient_eligible_fraction')} | "
            f"{validation_columns}"
        )

    def phase_switch(self, epoch: int, reason: str):
        if self.print_every:
            print(
                f"Switching to projection phase at epoch {epoch} "
                f"(trigger: {reason})."
            )

    def finish(self):
        if self.print_every and self._started:
            print("=" * 120)
            print("✅ Model training finished!")
