"""NumPy orchestration and direct primal-dual solver components."""
from .bounds import BoundIndices, BoundSlacks, BoundValues, compute_bound_slacks
from .iteration import FixedMuOptions, FixedMuResult, run_fixed_mu_iterations
from .state import PrimalDualState


def solve(problem, **kwargs):
    """Solve a problem that exposes the DAEHN ``solve`` protocol."""
    if hasattr(problem, "solve"):
        return problem.solve(**kwargs)
    raise TypeError("problem does not provide solve()")


__all__ = [
    "BoundIndices",
    "BoundSlacks",
    "BoundValues",
    "compute_bound_slacks",
    "FixedMuOptions",
    "FixedMuResult",
    "run_fixed_mu_iterations",
    "PrimalDualState",
    "solve",
]
