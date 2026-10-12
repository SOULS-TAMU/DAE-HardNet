"""DAEHN: functional JAX neural and differentiable optimization layers."""
from . import nn, problems, opt, losses, train, runtime, tree, dae
from .problems import (
    Problem,
    SymbolicProblem,
    FunctionalProblem,
    MatrixProblem,
    ParametricLP,
    ParametricQP,
)
from .problems.problem import (
    BatchNamedValues,
    BatchProblemSolution,
    CompiledBatchProblem,
    CompiledProblem,
    Constraint,
    ConstraintList,
    Expression,
    NamedValues,
    OptimizationProblem,
    PrimalDualProblem,
    ProblemSolution,
    PreparedSparseProjectionBatchSolver,
    ProjectionStructure,
    VectorExpression,
)
from .opt import ProjectionConfig, LinearSolverConfig, ImplicitGradientConfig
from .train.model import ProjectedNeuralModel
from .train.projected_neural import (
    Adam,
    AdamState,
    BuildReport,
    CompiledProjectedGraphs,
    ImplicitProjectionLayer,
    MLPBackbone,
    PredictionOutput,
    ProjectedMetrics,
    ProjectionLayerOutput,
    apply_updates,
    configure_compilation_cache,
    make_implicit_projection_layer,
)

__all__ = [
    "nn", "opt", "problems", "losses", "train", "runtime", "tree", "dae",
    "Problem", "SymbolicProblem", "FunctionalProblem", "MatrixProblem",
    "ParametricLP", "ParametricQP", "ProjectionConfig", "LinearSolverConfig",
    "ImplicitGradientConfig", "Adam", "MLPBackbone", "ProjectedNeuralModel",
    "make_implicit_projection_layer",
    "save_predictor", "load_predictor",
]


def save_predictor(*args, **kwargs):
    """Lazily dispatch to :func:`daehn.serialization.save_predictor`."""
    from .serialization import save_predictor as implementation
    return implementation(*args, **kwargs)


def load_predictor(*args, **kwargs):
    """Lazily dispatch to :func:`daehn.serialization.load_predictor`."""
    from .serialization import load_predictor as implementation
    return implementation(*args, **kwargs)

__version__ = "0.5.3"
