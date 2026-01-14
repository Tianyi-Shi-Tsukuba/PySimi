from typing import Optional
import numpy as np
from ..pairwise import pairwise_sq_l2

def adaptive_similarity(X: np.ndarray, k: int, n: Optional[int] = None) -> np.ndarray:
    """Adaptive neighbors similarity.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of neighbors to which each sample assigns similarity.

    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Symmetric adaptive similarity matrix with zeros on the diagonal.
        Higher values indicate stronger adaptive similarity between samples.

    Note
    -----
    - The diagonal is explicitly set to zero to avoid self-similarity.
    """
    dist2 = pairwise_sq_l2(X)
    dist_sorted = np.sort(dist2, axis=1)
    idx_sorted = np.argsort(dist2, axis=1)

    n_samples = dist2.shape[0]
    S = np.zeros((n_samples, n_samples), dtype=float)
    eps = np.finfo(float).eps

    di_all = dist_sorted[:, 1:k+2]
    id_all = idx_sorted[:, 1:k+2]

    di_k1 = di_all[:, k]

    numer = (di_k1[:, None] - di_all[:, :k])
    denom = (k * di_k1 - np.sum(di_all[:, :k], axis=1)) + eps

    rows = np.repeat(np.arange(n_samples), k)
    cols = id_all[:, :k].reshape(-1)
    vals = (numer / denom[:, None])[:, :k].reshape(-1)
    S[rows, cols] = vals

    W = 0.5 * (S + S.T)
    np.fill_diagonal(W, 0.0)
    return W
