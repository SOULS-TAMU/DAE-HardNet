"""Structured sparse linear and quadratic problems."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import jax.numpy as jnp
try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None
try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse = None
from .canonical import CanonicalQP
from .symbolic import SymbolicProblem


def _is_sparse(x):
    return sp is not None and sp.issparse(x)

def _shape(x): return tuple(x.shape)

def _dense_vector(x, size, name):
    if x is None: return np.zeros((size,), dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (size,): raise ValueError(f'{name} must have shape {(size,)}, got {arr.shape}')
    return arr

def _matrix(x, shape, name, *, zeros=False):
    if x is None:
        if sp is not None: return sp.csr_matrix(shape, dtype=np.float64)
        return np.zeros(shape, dtype=np.float64)
    if _shape(x) != shape: raise ValueError(f'{name} must have shape {shape}, got {_shape(x)}')
    if _is_sparse(x): return x.astype(np.float64).tocsr()
    return np.asarray(x, dtype=np.float64)

def _jax_matrix(x):
    if _is_sparse(x):
        if jsparse is None: return jnp.asarray(x.toarray())
        return jsparse.BCOO.from_scipy_sparse(x.tocoo())
    return jnp.asarray(x)

def _iter_row(matrix, row):
    if _is_sparse(matrix):
        csr = matrix.tocsr(); start, end = csr.indptr[row], csr.indptr[row+1]
        return zip(csr.indices[start:end].tolist(), csr.data[start:end].tolist())
    values = np.asarray(matrix[row])
    idx = np.flatnonzero(values)
    return zip(idx.tolist(), values[idx].tolist())

def _row_expr(matrix, row, symbols):
    return sum((float(v) * symbols[int(j)] for j, v in _iter_row(matrix, row)), 0.0)

def _affine_rhs(base, mapping, row, theta):
    return float(base[row]) + _row_expr(mapping, row, theta)

@dataclass(frozen=True)
class ParametricQP:
    P: Any
    q0: Any
    q_parameter_map: Any = None
    A_eq: Any = None
    b_eq0: Any = None
    b_eq_parameter_map: Any = None
    A_ineq: Any = None
    b_ineq0: Any = None
    b_ineq_parameter_map: Any = None
    name: str = 'parametric_qp'

    def __post_init__(self):
        if len(_shape(self.P)) != 2 or _shape(self.P)[0] != _shape(self.P)[1]:
            raise ValueError('P must be square')
        n = _shape(self.P)[0]
        q0 = _dense_vector(self.q0, n, 'q0')
        candidates = []
        for value in (self.q_parameter_map, self.b_eq_parameter_map, self.b_ineq_parameter_map):
            if value is not None: candidates.append(_shape(value)[1])
        m = candidates[0] if candidates else 0
        if any(c != m for c in candidates): raise ValueError('parameter maps must share one parameter dimension')
        P = _matrix(self.P, (n,n), 'P')
        Q = _matrix(self.q_parameter_map, (n,m), 'q_parameter_map')
        neq = 0 if self.A_eq is None else _shape(self.A_eq)[0]
        nineq = 0 if self.A_ineq is None else _shape(self.A_ineq)[0]
        Aeq = _matrix(self.A_eq, (neq,n), 'A_eq')
        Aineq = _matrix(self.A_ineq, (nineq,n), 'A_ineq')
        beq = _dense_vector(self.b_eq0, neq, 'b_eq0')
        bineq = _dense_vector(self.b_ineq0, nineq, 'b_ineq0')
        Beq = _matrix(self.b_eq_parameter_map, (neq,m), 'b_eq_parameter_map')
        Bineq = _matrix(self.b_ineq_parameter_map, (nineq,m), 'b_ineq_parameter_map')
        for name, value in [('P',P),('q0',q0),('q_parameter_map',Q),('A_eq',Aeq),('b_eq0',beq),('b_eq_parameter_map',Beq),('A_ineq',Aineq),('b_ineq0',bineq),('b_ineq_parameter_map',Bineq)]:
            object.__setattr__(self, name, value)
        object.__setattr__(self, '_parameter_size', m)

    @property
    def variable_size(self): return int(_shape(self.P)[0])
    @property
    def parameter_size(self): return int(self._parameter_size)
    @property
    def equality_size(self): return int(_shape(self.A_eq)[0])
    @property
    def inequality_size(self): return int(_shape(self.A_ineq)[0])

    def objective(self, theta, y):
        P = _jax_matrix(self.P); Q = _jax_matrix(self.q_parameter_map)
        q = jnp.asarray(self.q0) + Q @ theta
        return 0.5 * (y @ (P @ y)) + q @ y
    def equalities(self, theta, y):
        return _jax_matrix(self.A_eq) @ y - jnp.asarray(self.b_eq0) - _jax_matrix(self.b_eq_parameter_map) @ theta
    def inequalities(self, theta, y):
        return _jax_matrix(self.A_ineq) @ y - jnp.asarray(self.b_ineq0) - _jax_matrix(self.b_ineq_parameter_map) @ theta

    def canonicalize(self):
        return CanonicalQP(
            parameter_size=self.parameter_size, variable_size=self.variable_size,
            equality_size=self.equality_size, inequality_size=self.inequality_size,
            objective=self.objective, equalities=self.equalities, inequalities=self.inequalities,
            P=self.P, q0=self.q0, q_parameter_map=self.q_parameter_map,
            A_eq=self.A_eq, b_eq0=self.b_eq0, b_eq_parameter_map=self.b_eq_parameter_map,
            A_ineq=self.A_ineq, b_ineq0=self.b_ineq0, b_ineq_parameter_map=self.b_ineq_parameter_map,
            metadata={'name': self.name, 'kind': 'parametric_qp'},
        )

    def to_symbolic_problem(self, *, projection=False):
        model = SymbolicProblem(self.name + ('_projection' if projection else ''))
        target_names = [f'__target_{i}' for i in range(self.variable_size)] if projection else []
        theta_names = [f'theta_{i}' for i in range(self.parameter_size)]
        if target_names or theta_names: p = model.add_parameter(target_names + theta_names)
        else: p = model.parameter
        y = model.add_variable([f'y_{i}' for i in range(self.variable_size)])
        yv = y.vector(); theta = [getattr(p, name) for name in theta_names]
        if projection:
            targets = [getattr(p, name) for name in target_names]
            model.set_projection_objective(variables=yv, target=type(yv)(targets, 'target'))
        else:
            expr = 0.0
            for i in range(self.variable_size):
                for j, value in _iter_row(self.P, i):
                    expr = expr + 0.5 * float(value) * yv[i] * yv[j]
                expr = expr + float(self.q0[i]) * yv[i]
                for k, value in _iter_row(self.q_parameter_map, i):
                    expr = expr + float(value) * theta[k] * yv[i]
            model.objective = expr
        for i in range(self.equality_size):
            model.constraints.add(_row_expr(self.A_eq, i, yv) == _affine_rhs(self.b_eq0, self.b_eq_parameter_map, i, theta))
        for i in range(self.inequality_size):
            model.constraints.add(_row_expr(self.A_ineq, i, yv) <= _affine_rhs(self.b_ineq0, self.b_ineq_parameter_map, i, theta))
        return model

    def as_projection_problem(self): return self.to_symbolic_problem(projection=True)
    def solve(self, *, parameters=None, **kwargs):
        return self.to_symbolic_problem(projection=False).solve(parameters=parameters, **kwargs)

    @classmethod
    def from_bounds(cls, *, P, q0, q_parameter_map=None, A, lower0, upper0,
                    lower_parameter_map=None, upper_parameter_map=None, name='parametric_qp'):
        lower0 = np.asarray(lower0, dtype=np.float64); upper0 = np.asarray(upper0, dtype=np.float64)
        if lower0.shape != upper0.shape: raise ValueError('lower0 and upper0 shapes must match')
        eq_mask = np.isfinite(lower0) & np.isfinite(upper0) & (lower0 == upper0)
        upper_mask = np.isfinite(upper0) & ~eq_mask
        lower_mask = np.isfinite(lower0) & ~eq_mask
        rows=[]; rhs=[]; maps=[]
        A_arr = A.tocsr() if _is_sparse(A) else np.asarray(A)
        def subset(mat, mask): return mat[mask] if _is_sparse(mat) else np.asarray(mat)[mask]
        Aeq = subset(A_arr, eq_mask); beq=upper0[eq_mask]
        m = 0
        for candidate in (q_parameter_map, lower_parameter_map, upper_parameter_map):
            if candidate is not None: m=_shape(candidate)[1]; break
        Beq = np.zeros((eq_mask.sum(),m)) if upper_parameter_map is None else np.asarray(upper_parameter_map)[eq_mask]
        Aineq_parts=[]; b_parts=[]; B_parts=[]
        if upper_mask.any():
            Aineq_parts.append(subset(A_arr, upper_mask)); b_parts.append(upper0[upper_mask]); B_parts.append(np.zeros((upper_mask.sum(),m)) if upper_parameter_map is None else np.asarray(upper_parameter_map)[upper_mask])
        if lower_mask.any():
            Aineq_parts.append(-subset(A_arr, lower_mask)); b_parts.append(-lower0[lower_mask]); B_parts.append(np.zeros((lower_mask.sum(),m)) if lower_parameter_map is None else -np.asarray(lower_parameter_map)[lower_mask])
        if _is_sparse(A_arr): Aineq = sp.vstack(Aineq_parts, format='csr') if Aineq_parts else sp.csr_matrix((0,_shape(A_arr)[1]))
        else: Aineq = np.vstack(Aineq_parts) if Aineq_parts else np.zeros((0,_shape(A_arr)[1]))
        bineq=np.concatenate(b_parts) if b_parts else np.zeros((0,)); Bineq=np.vstack(B_parts) if B_parts else np.zeros((0,m))
        return cls(P=P,q0=q0,q_parameter_map=q_parameter_map,A_eq=Aeq,b_eq0=beq,b_eq_parameter_map=Beq,A_ineq=Aineq,b_ineq0=bineq,b_ineq_parameter_map=Bineq,name=name)

ParametricLP = ParametricQP
MatrixProblem = ParametricQP
