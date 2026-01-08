from typing import Optional
import numpy as np

def cal_pairwise_dist(x: np.ndarray) -> np.ndarray:
    """Return pairwise squared Euclidean distances for rows of x."""
    sum_x = np.sum(np.square(x), axis=1, keepdims=True)
    dist = sum_x + sum_x.T - 2 * (x @ x.T)
    np.maximum(dist, 0.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist

def _rbf_from_sqdist(Dsq: np.ndarray, t: float) -> np.ndarray:
    """RBF on squared distances: exp(-Dsq / (2 t^2))."""
    t = float(t)
    if t <= 0:
        raise ValueError("t must be positive for RBF.")
    return np.exp(-Dsq / (2.0 * (t ** 2)))

def cal_rbf_dist(data: np.ndarray, n_neighbors: int = 10, t: float = 1.0) -> np.ndarray:
    """Build a symmetric kNN RBF similarity matrix using squared distances."""
    Dsq = cal_pairwise_dist(data)
    rbf_dist = _rbf_from_sqdist(Dsq, t)
    n = Dsq.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        idx = np.argsort(Dsq[i])[1:1 + n_neighbors]  # skip self at rank 0
        W[i, idx] = rbf_dist[i, idx]
        W[idx, i] = rbf_dist[idx, i]
    return W

def lpp(data: np.ndarray, n_dims: int = 2, n_neighbors: int = 30, t: float = 1.0, W: Optional[np.ndarray] = None):
    """Locality Preserving Projection (LPP) with interchangeable similarity.

    If W is None, a kNN RBF similarity is constructed via cal_rbf_dist(data, n_neighbors, t).
    Otherwise, supply a precomputed similarity matrix W (n x n).

    Returns
    -------
    data_ndim : np.ndarray, shape (n, n_dims)
        Low-dimensional embedding of the rows of `data`.
    eig_vec_picked : np.ndarray, shape (d, n_dims)
        Projection directions A (columns) such that Y = X A.
    """
    X = np.asarray(data, dtype=float)
    N = X.shape[0]
    if W is None:
        W = cal_rbf_dist(X, n_neighbors=n_neighbors, t=t)
    else:
        W = np.asarray(W, dtype=float)
        if W.shape != (N, N):
            raise ValueError("W must be (n x n) where n = data.shape[0]")

    # Degree matrix
    D = np.zeros_like(W)
    for i in range(N):
        D[i, i] = np.sum(W[i])

    L = D - W
    XDXT = X.T @ D @ X
    XLXT = X.T @ L @ X

    # Generalized eigenproblem: (X^T L X) a = lambda (X^T D X) a
    # Solve via pinv to avoid requiring a symmetric-definite solver
    M = np.linalg.pinv(XDXT) @ XLXT
    eig_val, eig_vec = np.linalg.eig(M)

    # Sort by ascending |eigenvalues| and skip near-zero ones
    order = np.argsort(np.abs(eig_val))
    eig_val = eig_val[order]

    j = 0
    while j < eig_val.size and eig_val[j].real < 1e-6:
        j += 1

    pick = order[j:j + n_dims]
    eig_vec_picked = np.real(eig_vec[:, pick])

    data_ndim = X @ eig_vec_picked
    return data_ndim, eig_vec_picked

def lpp_from_data(X: np.ndarray, builder, n_dims: int = 2, n_neighbors: Optional[int] = 30, t: float = 1.0, **simi_kw):
    """Build similarity via `builder` then run LPP.

    If `builder` returns a full similarity S and n_neighbors is not None, keep for each row the
    top-`n_neighbors` entries (including self) to sparsify before forming L and running LPP.
    """
    S = builder(X, **simi_kw)
    if n_neighbors is not None and n_neighbors > 0:
        idx = np.argsort(-S, axis=1)[:, :n_neighbors]
        W = np.zeros_like(S)
        rows = np.repeat(np.arange(S.shape[0]), n_neighbors)
        cols = idx.reshape(-1)
        W[rows, cols] = S[rows, cols]
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 1.0)
    else:
        W = S
    return lpp(X, n_dims=n_dims, n_neighbors=n_neighbors if n_neighbors is not None else 0, t=t, W=W)
