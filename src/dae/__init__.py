"""Continuous DAE formulation and discretization frontend for DAEHN."""
from .symbols import Coordinate, DifferentialState, AlgebraicState, Parameter
from .derivatives import DerivativeSpec, derivative_closure
from .equations import EquationBlock
from .system import DAESystem
from .scaling import ScalingConfig
from .compiler import compile_dae
from .compiled import CompiledDAE, PreparedDAEProjection
from .model import (
    FullModelDerivativePlan,
    ProjectedDAEModel,
    extract_network_derivatives,
)
from .data import DAEData
from .training import DAETrainer, DAETrainingConfig, DAETrainingResult
from .logging import DAETrainingLogger
from .metrics import physical_residual_blocks, physical_violation_metrics, dae_metrics
from .discretization import TaylorNeighborhood, ForwardDifference, CentralDifference
from .conditions import Condition, InitialCondition, BoundaryCondition, sample_condition_coordinates, condition_mse, hard_initial_transform, homogeneous_box_transform

__all__=["Coordinate","DifferentialState","AlgebraicState","Parameter","DerivativeSpec","derivative_closure","EquationBlock","DAESystem","ScalingConfig","compile_dae","CompiledDAE","PreparedDAEProjection","FullModelDerivativePlan","ProjectedDAEModel","extract_network_derivatives","DAEData","DAETrainer","DAETrainingConfig","DAETrainingResult","DAETrainingLogger","physical_residual_blocks","physical_violation_metrics","dae_metrics","TaylorNeighborhood","ForwardDifference","CentralDifference","Condition","InitialCondition","BoundaryCondition","sample_condition_coordinates","condition_mse","hard_initial_transform","homogeneous_box_transform"]
