"""Host-side schedules selecting among already compiled executables."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ProjectionSchedule:
    min_pretrain_epochs: int = 0
    max_pretrain_epochs: int = 100
    eta: float | None = None
    readiness_fraction: float = 1.0
    on_not_ready: str = 'continue'
    def __post_init__(self):
        if self.min_pretrain_epochs<0 or self.max_pretrain_epochs<self.min_pretrain_epochs: raise ValueError('invalid pretraining bounds')
        if not 0<=self.readiness_fraction<=1: raise ValueError('readiness_fraction must be in [0,1]')
        if self.on_not_ready not in {'continue','raise','activate'}: raise ValueError('invalid on_not_ready policy')
    def should_attempt(self, epoch, metric=None):
        if epoch < self.min_pretrain_epochs: return False
        if epoch >= self.max_pretrain_epochs: return True
        return self.eta is not None and metric is not None and metric <= self.eta
