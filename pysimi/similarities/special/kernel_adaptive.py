from typing import Optional
import numpy as np
from ..pairwise import pairwise_sq_l2

def kernel_adaptive_similarity(X: np.ndarray, k: int, sigma: float, n: Optional[int] = None) -> np.ndarray:
    """Kernel-induced adaptive neighbors similarity .

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.

    k : int
        Number of neighbors to assign adaptive similarity weights to.

    sigma : float
        RBF kernel bandwidth parameter.

    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Symmetric adaptive similarity matrix with zeros on the diagonal.

    Note
    -----
    - The diagonal is explicitly set to zero to avoid self-similarity.
    """
    X = np.asarray(X, dtype=float)

    dist2 = pairwise_sq_l2(X)
    K = np.exp(-dist2 / (2.0 * (sigma ** 2)))
    kdist = 2.0 - 2.0 * K

    np.maximum(kdist, 0.0, out=kdist)

    np.fill_diagonal(kdist, 0.0)

    kdist_sorted = np.sort(kdist, axis=1)
    kidx_sorted = np.argsort(kdist, axis=1)

    n_samples = kdist.shape[0]
    S = np.zeros((n_samples, n_samples), dtype=float)
    eps = np.finfo(float).eps

    kdi_all = kdist_sorted[:, 1:k+2]
    kid_all = kidx_sorted[:, 1:k+2]

    kdi_k1 = kdi_all[:, k]

    numer = (kdi_k1[:, None] - kdi_all[:, :k])
    denom = (k * kdi_k1 - np.sum(kdi_all[:, :k], axis=1)) + eps

    rows = np.repeat(np.arange(n_samples), k)
    cols = kid_all[:, :k].reshape(-1)
    vals = (numer / denom[:, None])[:, :k].reshape(-1)

    S[rows, cols] = vals

    W = 0.5 * (S + S.T)
    np.fill_diagonal(W, 0.0)

    return W
