from sklearn.neighbors import kneighbors_graph
import numpy as np
from typing import Optional
def knn(X: np.ndarray, k_neighbors: int, symmetrize: str = "max", n: Optional[int] = None)-> np.ndarray:
    """
    Parameters
    ----------
    X - Array of data points

    sigma - param used in rbf for similarity computation

    k - Number of neighbors in kNN graph

    symmetrize - method of symmetrizing the similarity matrix

    Returns
    -------
    W - ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix with zeros on the diagonal. Higher values
        indicate stronger similarity under the k nearest neighbors graph.

    Notes
    -----
    - The diagonal is explicitly set to zero to avoid self-similarity.
    """

    # constructing a kNN graph
    A = kneighbors_graph(X, k_neighbors, mode='distance', metric='euclidean', include_self=True)
    A = A.tocsr().copy()

    d = A.data

    d_min, d_max = d.min(), d.max()
    d_norm = (d - d_min) / (d_max - d_min)

    A.data = 1 - d_norm


    W = A.toarray()

    # symmetrize：W = max(W, W.T) or (W + W.T)/2
    if symmetrize == 'max':
        W = np.maximum(W, W.T)
    elif symmetrize == 'mean':
        W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)
    return W


