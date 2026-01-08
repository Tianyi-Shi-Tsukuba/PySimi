from typing import Optional
import numpy as np
from ..pairwise import pairwise_sq_l2

def close_shared_neighbors_similarity(A: np.ndarray, k: int, sigma: float, n: Optional[int] = None) -> np.ndarray:
    """Close-Shared Neighbors (CSNN) similarity.

    Parameters
    ----------
    A : ndarray of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of nearest neighbors to consider when computing shared-neighbors.

    sigma : float
        RBF kernel bandwidth. Controls how pairwise distances are transformed
        before ranking.

    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix in [0, 1]. Higher values indicate stronger
        similarity based on *rank-weighted shared neighborhood*. The diagonal is
        explicitly set to zero to avoid self-similarity.

    Note
    -----
    - The diagonal is explicitly set to zero to avoid self-similarity."""
    A = np.asarray(A, dtype=float)
    n_samples = A.shape[0]

    dist2 = pairwise_sq_l2(A)
    B = np.exp(-dist2 / (2.0 * (sigma ** 2)))
    idx_sorted = np.argsort(-B, axis=1)
    topk = idx_sorted[:, :k]

    w = (k - np.arange(k) + 1).astype(float)  # linear rank weights

    pos = np.zeros((n_samples, n_samples), dtype=int)
    rows = np.repeat(np.arange(n_samples), k)
    cols = topk.reshape(-1)
    ps = np.tile(np.arange(k,0, -1), n_samples)  # 1..k
    pos[rows, cols] = ps
    np.fill_diagonal(pos, 0)

    W = pos @ pos.T + pos * pos.T

    N = np.sum((k - np.arange(1, k + 1) ) ** 2)

    W = W / N

    np.fill_diagonal(W, 0)
    return W
