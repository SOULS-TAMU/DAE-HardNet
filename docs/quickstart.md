# Quick start

After installing the repository, run this snippet in Python:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from daehn.opt.linear_equality import LinearEqualityProjection

project = LinearEqualityProjection(
    A=np.array([[1., 1.]]), b=np.array([1.])
).prepare(batch_size=1)

value = jnp.array([2., 0.])
print(jax.jit(project)(value))  # [1.5, -0.5]
print(jax.jacfwd(project)(value))
```

This projects onto x+y=1 and evaluates the explicit layer's derivative. It needs no dataset or saved model. See [DAE formulation](formulation.md) for the continuous-system API. Full case studies remain in daehn-benchmarks.
