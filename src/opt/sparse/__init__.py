"""Sparse format conversion utilities."""
from __future__ import annotations
import numpy as np
try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp=None
try:
    from jax.experimental import sparse as jsparse
except Exception:  # pragma: no cover
    jsparse=None

def as_csr(value):
    if sp is None: raise ImportError('scipy is required')
    return value.tocsr() if sp.issparse(value) else sp.csr_matrix(np.asarray(value))
def as_csc(value):
    if sp is None: raise ImportError('scipy is required')
    return value.tocsc() if sp.issparse(value) else sp.csc_matrix(np.asarray(value))
def as_bcoo(value):
    if jsparse is None: raise ImportError('jax.experimental.sparse is required')
    if sp is not None and sp.issparse(value): return jsparse.BCOO.from_scipy_sparse(value.tocoo())
    return jsparse.BCOO.fromdense(value)
def validate_pattern(matrix):
    shape=tuple(matrix.shape)
    if len(shape)!=2: raise ValueError('sparse matrix must be rank two')
    return shape
__all__=['as_csr','as_csc','as_bcoo','validate_pattern']
