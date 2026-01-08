from typing import Optional
import numpy as np
from ..pairwise import pairwise_sq_l2

def number_shared_neighbors_similarity(A: np.ndarray, k: int, sigma: float, n: Optional[int] = None) -> np.ndarray:
    """Number-Shared Neighbors (NSNN) similarity

    Parameters
    ----------
    A - ndarray of shape(n_samples, n_features)
        Array of data points

    k : int
        Number of neighbors

    sigma : float
        Bandwidth parameter of the RBF kernel used to compute the initial
        soft similarity before selecting k-nearest neighbors.

    n : int, optional
        Placeholder parameter for compatibility. Not used in this implementation.

    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix representing the normalized number of shared
        neighbors similarity. Larger values correspond to stronger structural similarity.
        The diagonal is explicitly set to zero to remove self-similarity.

    Note
    -----
    - The diagonal is zeroed out for compatibility with graph-based methods
      such as spectral clustering."""
    A = np.asarray(A, dtype=float)
    n_samples = A.shape[0]

    dist2 = pairwise_sq_l2(A)
    B = np.exp(-dist2 / (2.0 * (sigma ** 2)))

    idx_sorted = np.argsort(-B, axis=1)
    topk = idx_sorted[:, :k]

    G = np.zeros((n_samples, n_samples), dtype=int)
    rows = np.repeat(np.arange(n_samples), k)
    cols = topk.reshape(-1)
    G[rows, cols] = 1

    inter = G @ G.T
    W = inter.astype(float)

    mutual = (G & G.T).astype(int)
    W += mutual

    W /= float(k)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)
    return W
