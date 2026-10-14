# Projection algorithms

The explicit linear-equality projector solves a Euclidean correction onto A*y=b(parameters). It is intended for affine equalities. Prepared layers have a fixed batch size. Reuse preparation and compilation when measuring inference cost.

Legacy selections `linear`, `sparse`, `fb`, and automatic compiled selection retain their established implementations. Explicit shared selections `fischer-burmeister-newton`, `fischer-burmeister-reduced`, and `primal-dual-interior-point` use KKTHN through daehn.opt.api.prepare_projection. The full and reduced Fischer-Burmeister algorithms address complementarity; the interior-point method uses a finite barrier. Switching algorithms can change residuals, solutions and gradients.

Use `with_info` when solver diagnostics are needed. Check convergence codes, finite state, residuals and gradient eligibility rather than assuming every returned point is valid. Nonlinear and nonconvex constraints may admit several solutions. Derivative checks should avoid active-set transitions and singular constraint Jacobians.

Configure JAX x64 before constructing arrays when float64 accuracy is required. Float32 constraint residuals should not be judged against float64 tolerances. State the solver configuration, dtype and stopping tolerance in numerical comparisons.

The shared implicit layers use custom reverse-mode derivatives. Use jax.grad or jax.jacrev for them; jax.jacfwd/JVP is not supported by those custom-VJP layers. The explicit linear layer supports both modes.
