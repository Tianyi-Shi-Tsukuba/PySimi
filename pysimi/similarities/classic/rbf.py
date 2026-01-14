from typing import Optional
import numpy as np
from ..pairwise import pairwise_sq_l2, pairwise_l2

def rbf_gaussian(
    X: np.ndarray,
    sigma: Optional[float] = None,
    gamma: Optional[float] = None,
    use_squared: bool = True
) -> np.ndarray:
    """Gaussian (RBF) similarity.

    S_ij = exp(-gamma * d_ij^2) if use_squared else exp(-gamma * d_ij).
    If sigma is provided and gamma is None, gamma = 1 / (2*sigma^2).
    """
    if gamma is None:
        if sigma is None:
            raise ValueError("Provide either gamma or sigma for RBF.")
        gamma = 1.0 / (2.0 * (sigma ** 2))

    if use_squared:
        D = pairwise_sq_l2(X)
        S = np.exp(-gamma * D)
    else:
        D = pairwise_l2(X)
        S = np.exp(-gamma * D)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S
