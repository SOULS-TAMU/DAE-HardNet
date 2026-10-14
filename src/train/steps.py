"""Functional train and evaluation step builders."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from ..tree import tree_all_finite, tree_select

def _apply_updates(params, updates): return jax.tree_util.tree_map(lambda p,u:p+u, params, updates)

def _optimizer_update(optimizer, gradients, state, params):
    try: return optimizer.update(gradients,state,params)
    except TypeError: return optimizer.update(gradients,state)

def make_train_step(loss_fn, optimizer, *, invalid_batch_policy='skip'):
    if invalid_batch_policy not in {'skip','use_valid'}: raise ValueError('invalid_batch_policy must be skip or use_valid')
    value_and_grad=jax.value_and_grad(loss_fn,has_aux=True)
    def step(params, optimizer_state, batch):
        (loss,metrics),grads=value_and_grad(params,batch)
        valid=jnp.isfinite(loss)&tree_all_finite(grads)
        if 'gradient_valid' in metrics: valid=valid&jnp.asarray(metrics['gradient_valid'])
        updates,candidate_state=_optimizer_update(optimizer,grads,optimizer_state,params)
        candidate_params=_apply_updates(params,updates)
        if invalid_batch_policy=='skip':
            new_params=tree_select(valid,candidate_params,params); new_state=tree_select(valid,candidate_state,optimizer_state)
        else:
            new_params,new_state=candidate_params,candidate_state
        metrics={**metrics,'loss':loss,'update_accepted':valid}
        return new_params,new_state,metrics
    return step

def make_eval_step(loss_fn):
    def step(params,batch):
        loss,metrics=loss_fn(params,batch)
        return {**metrics,'loss':loss}
    return step
