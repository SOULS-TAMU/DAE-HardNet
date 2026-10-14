from dataclasses import dataclass
from typing import Any
from .sparse.batch_iteration import BATCH_STATUS_CONVERGED

@dataclass(frozen=True)
class ProjectionInfo:
    forward_status_codes: Any
    forward_iterations: Any
    forward_residual_inf_norm: Any
    forward_state_finite: Any
    gradient_eligible: Any

    @property
    def forward_converged(self):
        return self.forward_status_codes == BATCH_STATUS_CONVERGED


