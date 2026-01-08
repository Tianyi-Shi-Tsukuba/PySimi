import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Optional

def self_tuning_similarity(X: np.ndarray, k: int=7, n: Optional[int] = None)-> np.ndarray:
    """
    Construct similarity matrix W using the Self-Tuning method.

    Parameters
    ----------
    X : (n, d) data matrix

    k : number of neighbors used to determine local scale σ_i

    Returns
    -------
    W : ndarray of shape (n, n)
        Symmetric similarity matrix with zeros on the diagonal. Higher values
        indicate stronger similarity under the self tuning method.

    Note
    -----
    The diagonal is explicitly set to zero to avoid self-similarity.
    """
    D = pairwise_distances(X)

    sort_D = np.sort(D, axis=1)
    sigma = sort_D[:, k]    # σ_i

    sigma_matrix = np.outer(sigma, sigma)

    W = np.exp(- D**2 / sigma_matrix)

    # remove self-similarity
    np.fill_diagonal(W, 0)

    return W
