"""Compile continuous DAE definitions into standard DAEHN projection problems."""
from __future__ import annotations
from hashlib import sha256
import json
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from .compiled import CompiledDAE, ProjectionDiagnostics
from .layouts import ProjectionLayout, ContextLayout
from .scaling import ScalingConfig, safe_row_scales


def compile_dae(system, *, discretization, scaling: ScalingConfig | None = None) -> CompiledDAE:
    scaling = ScalingConfig() if scaling is None else scaling
    disc = discretization.compile(system, scaling)
    nderiv = len(disc.derivatives)
    nalg = len(system.algebraic_states)
    projection_layout = ProjectionLayout(slice(0, nderiv), slice(nderiv, nderiv+nalg), nderiv+nalg)
    ncoords=len(system.coordinates); nparams=len(system.parameters); ndiff=len(system.differential_states); nneigh=int(disc.offsets.shape[0])
    start=0; coordinate_slice=slice(start,start+ncoords); start+=ncoords
    parameter_slice=slice(start,start+nparams); start+=nparams
    center_slice=slice(start,start+ndiff); start+=ndiff
    neighbor_slice=slice(start,start+nneigh*ndiff); start+=nneigh*ndiff
    context_layout=ContextLayout(coordinate_slice,parameter_slice,center_slice,neighbor_slice,ncoords,nparams,ndiff,nneigh,start)
    derivative_scales=jnp.asarray(disc.derivative_physical_scales)
    algebraic_scales=jnp.asarray([s.scale for s in system.algebraic_states], dtype=jnp.float64)
    variable_scales=jnp.concatenate((derivative_scales,algebraic_scales))

    def reconstruct(context, normalized_variables):
        parts=context_layout.unpack(context)
        dbar, abar=projection_layout.unpack(normalized_variables)
        physical_derivatives=dbar*derivative_scales
        differential_states=disc.reconstruct_differential_states(parts,dbar)
        algebraic_states=abar*algebraic_scales
        states=jnp.concatenate((differential_states,algebraic_states),axis=-1)
        return states,physical_derivatives

    def raw_equalities(context, normalized_variables):
        coords,params,_center,_neighbors=context_layout.unpack(context)
        states,derivatives=reconstruct(context,normalized_variables)
        differential=system.differential_equations.function(coords,states,derivatives,params)
        algebraic=system.algebraic_equalities.function(coords,states,derivatives,params)
        return jnp.concatenate((jnp.atleast_1d(differential),jnp.atleast_1d(algebraic)))

    def raw_inequalities(context, normalized_variables):
        coords,params,_center,_neighbors=context_layout.unpack(context)
        states,derivatives=reconstruct(context,normalized_variables)
        return jnp.atleast_1d(system.algebraic_inequalities.function(coords,states,derivatives,params))

    zero_context=jnp.zeros((context_layout.size,),dtype=jnp.float64)
    zero_variables=jnp.zeros((projection_layout.size,),dtype=jnp.float64)
    if system.equality_size:
        A0=np.asarray(jax.jacfwd(raw_equalities,argnums=1)(zero_context,zero_variables),dtype=np.float64)
        row_scales=safe_row_scales(A0,scaling)
    else:
        A0=np.zeros((0,projection_layout.size),dtype=np.float64); row_scales=np.ones((0,),dtype=np.float64)
    ineq_scales=np.ones((system.algebraic_inequalities.size,),dtype=np.float64)
    row_scales_j=jnp.asarray(row_scales)
    ineq_scales_j=jnp.asarray(ineq_scales)
    def scaled_equalities(context,normalized_variables):
        return raw_equalities(context,normalized_variables)/row_scales_j
    def scaled_inequalities(context,normalized_variables):
        return raw_inequalities(context,normalized_variables)/ineq_scales_j

    all_affine=(system.differential_equations.affine_in_projection_variables and system.algebraic_equalities.affine_in_projection_variables and system.algebraic_inequalities.affine_in_projection_variables)
    linear_matrix=None; cond=float("nan")
    if all_affine and system.algebraic_inequalities.size==0:
        linear_matrix=np.asarray(A0/row_scales[:,None],dtype=np.float64)
        # Check the variable Jacobian is context-independent at a nonzero representative context.
        probe_context=jnp.linspace(0.1,0.9,max(1,context_layout.size),dtype=jnp.float64)[:context_layout.size]
        A1=np.asarray(jax.jacfwd(scaled_equalities,argnums=1)(probe_context,zero_variables))
        if not np.allclose(linear_matrix,A1,rtol=1e-9,atol=1e-10):
            linear_matrix=None
        elif linear_matrix.shape[0] and np.linalg.matrix_rank(linear_matrix)<linear_matrix.shape[0]:
            warnings.warn("compiled affine equality matrix is row-rank deficient; using nonlinear sparse projection",RuntimeWarning)
            linear_matrix=None
        else:
            cond=float(np.linalg.cond(linear_matrix@linear_matrix.T)) if linear_matrix.shape[0] else 1.0
            if cond>scaling.condition_warning:
                warnings.warn(f"scaled projection matrix condition estimate is {cond:.3e}",RuntimeWarning)

    def build_target(center_all_states, normalized_derivative_anchor=None):
        center_all_states=jnp.asarray(center_all_states)
        if normalized_derivative_anchor is None:
            normalized_derivative_anchor=jnp.zeros(center_all_states.shape[:-1]+(nderiv,),dtype=center_all_states.dtype)
        alg=center_all_states[...,ndiff:ndiff+nalg]/algebraic_scales if nalg else jnp.empty(center_all_states.shape[:-1]+(0,),center_all_states.dtype)
        return jnp.concatenate((normalized_derivative_anchor,alg),axis=-1)

    def decode(context,normalized_variables):
        states,derivatives=reconstruct(context,normalized_variables)
        return {"states":states,"derivatives":derivatives,"normalized_variables":normalized_variables}

    signature_payload={"system":system.name,"states":system.state_names,"derivatives":[(d.state.name,d.multi_index) for d in disc.derivatives],"discretization":disc.signature,"scaling":scaling.__dict__}
    signature=sha256(json.dumps(signature_payload,sort_keys=True,default=str).encode()).hexdigest()
    diagnostics=ProjectionDiagnostics("linear" if linear_matrix is not None else "sparse",cond,system.equality_size,projection_layout.size)
    return CompiledDAE(system,disc,projection_layout,context_layout,row_scales_j,variable_scales,raw_equalities,raw_inequalities,scaled_equalities,scaled_inequalities,reconstruct,build_target,decode,linear_matrix,signature,diagnostics)
