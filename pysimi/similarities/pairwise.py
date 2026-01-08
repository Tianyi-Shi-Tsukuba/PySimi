import numpy as np

def pairwise_sq_l2(X: np.ndarray) -> np.ndarray:
    """Return pairwise squared Euclidean distances between row vectors of X."""
    X = np.asarray(X, dtype=float)
    xx = np.sum(X * X, axis=1, keepdims=True)
    dist2 = xx + xx.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    np.fill_diagonal(dist2, 0.0)
    return dist2

def pairwise_l2(X: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance matrix."""
    return np.sqrt(pairwise_sq_l2(X))

def pairwise_l1(X: np.ndarray) -> np.ndarray:
    """Return the pairwise L1/Manhattan distance matrix between rows of X."""
    X = np.asarray(X, dtype=float)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sum(np.abs(diffs), axis=2)
