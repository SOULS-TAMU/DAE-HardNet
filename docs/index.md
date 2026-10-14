# DAEHN

DAEHN provides JAX-based DAE formulation, discretization, neural models and differentiable projection layers. This checkout is a reorganized library implementation associated with SOULS-TAMU/DAE-HardNet; the original public research layout and this library layout may differ until the library revision is published.

Start with [installation](installation.md) and the [quick start](quickstart.md). Full manuscript studies and trained models live in the sibling daehn-benchmarks project, not in the library distribution.

Projection enforces the constraints encoded in the selected formulation, subject to solver convergence, numerical tolerances and precision. An algebraic/invariant projection does not automatically enforce every differential equation, guarantee trajectory accuracy or certify downstream control safety.
