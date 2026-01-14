from typing import Optional
import numpy as np

def polynomial_kernel(
    X: np.ndarray,
    degree: int = 2,
    gamma: Optional[float] = None,
    coef0: float = 1.0
) -> np.ndarray:
    """Polynomial kernel: S = (gamma * X X^T + coef0)^degree."""
    X = np.asarray(X, dtype=float)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    S = (gamma * (X @ X.T) + coef0) ** degree
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S
