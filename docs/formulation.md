# DAE formulation

Create typed Coordinate, DifferentialState, AlgebraicState and Parameter objects from daehn.dae. Declare derivative multi-indices with DerivativeSpec. An EquationBlock function receives coordinates, reconstructed states, physical derivatives and parameters, and returns a residual vector of its declared size.

DAESystem combines differential equations, algebraic equalities and algebraic inequalities. Compile it with compile_dae and an explicit discretization, such as TaylorNeighborhood(deltas=(0.01,), order=1). Affine metadata must describe the actual dependence on projection variables; marking a nonlinear equation affine is incorrect.

Compiled context includes coordinates, parameters, center states and neighboring differential states. Use compiled.pack_context rather than constructing offsets manually. Physical residuals and reconstructed outputs are available through the compiled interface. The self-contained DAE regression in tests/test_dae.py checks this reconstruction and its physical residuals.

The formulation determines which variables projection can correct. Benchmark models that project only an invariant or algebraic rates still evaluate differential residuals separately. Do not interpret every use of the DAEHN name as a hard projection of the full continuous ODE.
