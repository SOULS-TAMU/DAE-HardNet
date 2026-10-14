# API map

| Namespace | Purpose |
|---|---|
| daehn.dae | Symbols, systems, compilation, discretization, conditions and DAE training |
| daehn.problems | Symbolic, functional and structured optimization frontends |
| daehn.opt | Solver configuration, backend selection and differentiable projection |
| daehn.nn | Functional neural layers |
| daehn.train | Backbones, optimization and projected models |
| daehn.losses | Loss helpers |
| daehn.serialization | Predictor and weight persistence |
| daehn.export / daehn.runtime | Deployment artifacts and runtime helpers |
| daehn.utils.sparse | Shared KKTHN sparse operations |

Historical daehn.opt.layers imports are retained. New organized projection namespaces coexist with them. Source directly in src/ maps to the daehn package using pyproject.toml; users continue to import daehn. Detailed callable docstrings are available through Python help().
