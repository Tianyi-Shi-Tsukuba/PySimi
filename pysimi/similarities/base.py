from typing import Optional
import numpy as np
try:
    from scipy import sparse as _sparse 
except Exception:
    _sparse = None

def ensure_symmetric(S, diag: Optional[float] = None):
    """Return 0.5*(S+S.T) and optionally set the diagonal to a constant."""
    if _sparse is not None and hasattr(_sparse, "issparse") and _sparse.issparse(S):
        S = 0.5 * (S + S.T)
        if diag is not None:
            S.setdiag(diag)
        return S
    S = 0.5 * (S + S.T)
    if diag is not None:
        np.fill_diagonal(S, diag)
    return S
