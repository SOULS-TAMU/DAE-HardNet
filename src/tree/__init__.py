"""Small PyTree utilities used by functional training steps."""
import jax
import jax.numpy as jnp

def tree_all_finite(tree):
    leaves=jax.tree_util.tree_leaves(tree)
    result=jnp.asarray(True)
    for leaf in leaves: result=result & jnp.all(jnp.isfinite(leaf))
    return result

def tree_select(condition, true_tree, false_tree):
    return jax.tree_util.tree_map(lambda a,b: jnp.where(condition, a, b), true_tree, false_tree)
__all__=['tree_all_finite','tree_select']
