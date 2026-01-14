import numpy as np

def _row_standardize(X: np.ndarray) -> np.ndarray:
    """Row-standardize matrix: (X - mean_row) / std_row."""
    X = np.asarray(X, dtype=float)
    m = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + np.finfo(float).eps
    return (X - m) / sd

def pearson_correlation(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Pearson correlation between row vectors (features as columns)."""

    if center:
        Z = _row_standardize(X)
    else:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, np.finfo(float).eps)
        Z = X / norms
    S = Z @ Z.T / Z.shape[1]
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S

def spearman_correlation(X: np.ndarray) -> np.ndarray:
    """Spearman correlation between row vectors via per-row ranking."""
  
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    ranks = np.empty_like(X, dtype=float)
    for i in range(n):
        order = np.argsort(X[i], kind="mergesort")
        ranks[i, order] = np.arange(1, d + 1, dtype=float)
    m = ranks.mean(axis=1, keepdims=True)
    sd = ranks.std(axis=1, keepdims=True) + np.finfo(float).eps
    Z = (ranks - m) / sd
    S = Z @ Z.T / d
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S
