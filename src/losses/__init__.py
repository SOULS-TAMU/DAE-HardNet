"""Functional losses for supervised, objective, and inverse learning."""
from __future__ import annotations
import jax
import jax.numpy as jnp

def mse(prediction, target, mask=None):
    values=jnp.mean(jnp.square(prediction-target), axis=-1)
    if mask is None: return jnp.mean(values)
    mask=jnp.asarray(mask,dtype=values.dtype); return jnp.sum(values*mask)/jnp.maximum(jnp.sum(mask),1.0)

def consistency(raw, projected, mask=None): return mse(projected, raw, mask)

def supervised_mse(*, prediction_key='projected', target_key='solution', consistency_weight=0.0):
    def loss(outputs, batch):
        task=mse(outputs[prediction_key], batch[target_key], batch.get('mask'))
        cons=consistency(outputs['raw'], outputs['projected'], batch.get('mask')) if consistency_weight else jnp.asarray(0.0,dtype=task.dtype)
        total=task+float(consistency_weight)*cons
        return total, {'task_loss':task,'consistency_loss':cons}
    return loss

def problem_objective(objective):
    def loss(outputs,batch):
        values=jax.vmap(objective)(batch['theta'], outputs['projected'])
        value=jnp.mean(values)
        return value, {'objective':value}
    return loss

def inverse_observation(observation_fn=lambda y:y):
    def loss(outputs,batch):
        value=mse(observation_fn(outputs['projected']), batch['observations'], batch.get('mask'))
        return value, {'observation_loss':value}
    return loss

def weighted_sum(terms):
    terms=dict(terms)
    def loss(outputs,batch):
        total=jnp.asarray(0.0); metrics={}
        for name,(weight,fn) in terms.items():
            value,aux=fn(outputs,batch); total=total+float(weight)*value
            metrics[name]=value; metrics.update({f'{name}/{k}':v for k,v in aux.items()})
        return total,metrics
    return loss
__all__=['mse','consistency','supervised_mse','problem_objective','inverse_observation','weighted_sum']
