"""JAX runtime and AOT compilation helpers."""
from pathlib import Path
import jax

def configure_cache(path):
    if path is None: return
    p=Path(path).expanduser().resolve(); p.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_compilation_cache_dir',str(p))
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes',-1)
def select_device(backend='cpu'):
    backend='gpu' if str(backend).lower() in {'cuda','nvidia'} else str(backend).lower()
    devices=jax.devices(backend)
    if not devices: raise ValueError(f'backend {backend!r} unavailable')
    return devices[0]
def aot_compile(function, *example_args, donate_argnums=()):
    return jax.jit(function,donate_argnums=tuple(donate_argnums)).lower(*example_args).compile()
__all__=['configure_cache','select_device','aot_compile']
