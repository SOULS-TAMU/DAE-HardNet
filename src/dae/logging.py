"""Fixed-width DAE-HardNet training logs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DAETrainingLogger:
    """Render data, derivative, and physical-violation training tables."""

    epochs: int
    print_every: int = 1
    width: int = 17
    title: str = "DAEHN"
    _started: bool = False

    @property
    def _rule_width(self) -> int:
        # Epoch + separators + eight metric columns.
        return self.width + 3 + (self.width * 4 + 3) * 2 + 3

    def start(
        self,
        *,
        system: str,
        discretization: str,
        projection: str,
        samples: int,
        validation_samples: int,
        batch_size: int,
        network: str,
        pretrain_epochs: int,
        eta: float | None,
    ) -> None:
        if not self.print_every:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Model training started! [{now}]\n")
        print(self.title)
        print(f"  system: {system}")
        print(f"  discretization: {discretization}")
        print(f"  projection: {projection}")
        print(
            f"  samples: train={samples} validation={validation_samples} "
            f"batch_size={batch_size}"
        )
        print(f"  network: {network}")
        print(f"  pretrain_epochs: {pretrain_epochs}")
        print(f"  eta: {'n/a' if eta is None else eta}")
        rule = "=" * self._rule_width
        group_width = self.width * 4 + 3
        print(rule)
        print(
            f"{'Epoch':>{self.width}} | "
            f"{'Training':^{group_width}} | "
            f"{'Validation':^{group_width}}"
        )
        print("-" * self._rule_width)
        print(
            f"{'':>{self.width}} | "
            f"{'Data Loss':>{self.width}} "
            f"{'Derivative Loss':>{self.width}} "
            f"{'DE Violation':>{self.width}} "
            f"{'Alg Violation':>{self.width}} | "
            f"{'Data Loss':>{self.width}} "
            f"{'Derivative Loss':>{self.width}} "
            f"{'DE Violation':>{self.width}} "
            f"{'Alg Violation':>{self.width}}"
        )
        print(rule)
        self._started = True

    def should_print(self, epoch: int) -> bool:
        return bool(self.print_every) and (
            epoch == 1
            or epoch == self.epochs
            or epoch % max(1, int(self.print_every)) == 0
        )

    def epoch(self, epoch: int, train: dict, validation: dict) -> None:
        if not self.should_print(epoch):
            return

        def fmt(row: dict, key: str) -> str:
            return f"{float(row[key]):>{self.width}.4e}"

        label = f"{epoch:03d}/{self.epochs:03d}"
        print(
            f"{label:>{self.width}} | "
            f"{fmt(train, 'data_loss')} "
            f"{fmt(train, 'derivative_loss')} "
            f"{fmt(train, 'differential_violation')} "
            f"{fmt(train, 'algebraic_violation')} | "
            f"{fmt(validation, 'data_loss')} "
            f"{fmt(validation, 'derivative_loss')} "
            f"{fmt(validation, 'differential_violation')} "
            f"{fmt(validation, 'algebraic_violation')}"
        )

    def phase_switch(self, epoch: int, reason: str) -> None:
        if self.print_every:
            print(
                f"Switching to projected DAE-HardNet phase at epoch {epoch} "
                f"(trigger: {reason})."
            )

    def finish(self, wall_seconds: float) -> None:
        if self.print_every and self._started:
            print("=" * self._rule_width)
            print(f"⏱️ Model training time (wall time) = {wall_seconds:.4f} seconds")
            print("✅ Model training finished!")
