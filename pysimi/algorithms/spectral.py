from typing import Optional
import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(similarity_matrix: np.ndarray, num_clusters: int, random_state: int = 42):
    """Run normalized-cut spectral clustering on a given similarity matrix.

    This follows the normalized Laplacian approach:
      L_sym = D^{-1/2} (D - W) D^{-1/2}
    Then take the first `num_clusters` eigenvectors and run KMeans.
    """
    S = np.asarray(similarity_matrix, dtype=float)
    # Degree vector
    degrees = np.sum(S, axis=1)
    # Safe inverse sqrt degrees
    with np.errstate(divide='ignore'):
        inv_sqrt_deg = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D = np.diag(degrees)
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    L = D - S
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(L_sym)
    # Sort by ascending eigenvalues
    order = np.argsort(eigvals.real)
    U = np.real(eigvecs[:, order[:num_clusters]])

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(U)
    return labels

def spectral_from_data(X: np.ndarray, builder, num_clusters: int, knn: Optional[int] = None, **simi_kw):
    """Compute similarity via `builder(X, **simi_kw)`, optional kNN sparsification, then spectral clustering.

    Parameters
    ----------
    X : array-like, shape (n, d)
    builder : callable
        A function like `build_similarity(X, method=..., **params)` that returns an (n x n) similarity matrix.
    num_clusters : int
    knn : int or None
        If given, keep for each row only the top-`knn` similarities (including self), then symmetrize.
    simi_kw : dict
        Extra kwargs forwarded to `builder`.
    """
    S = builder(X, **simi_kw)
    if knn is not None and knn > 0:
        idx = np.argsort(-S, axis=1)[:, :knn]
        S2 = np.zeros_like(S)
        rows = np.repeat(np.arange(S.shape[0]), knn)
        cols = idx.reshape(-1)
        S2[rows, cols] = S[rows, cols]
        # make symmetric by max
        S = np.maximum(S2, S2.T)
        np.fill_diagonal(S, 1.0)
    return spectral_clustering(S, num_clusters=num_clusters)
