from typing import Optional
import numpy as np

def sigmoid_kernel(
    X: np.ndarray,
    gamma: Optional[float] = None,
    coef0: float = 0.0
) -> np.ndarray:
    """Sigmoid (tanh) kernel: S = tanh(gamma * X X^T + coef0)."""
    X = np.asarray(X, dtype=float)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    S = np.tanh(gamma * (X @ X.T) + coef0)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S
