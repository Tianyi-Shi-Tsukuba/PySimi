import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from typing import Optional


def snn_importance_similarity(
    A: np.ndarray,
    k: int,
    sigma: float,
    n: Optional[int] = None
) -> np.ndarray:
    """
    Shared-Nearest-Neighbors Importance similarity.

    A: (n_samples, n_features)
    k: shared kNN size
    sigma: global scaling factor
    """
    A = np.asarray(A, dtype=float)
    n_samples = A.shape[0]
    if n_samples == 0:
        return np.zeros((0, 0), dtype=float)

    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= n_samples:
        k = n_samples - 1
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    D = pairwise_distances(A)
    np.fill_diagonal(D, 0.0)

    TH = float(np.mean(D[np.triu_indices(n_samples, k=1)]))
    B = (D < TH).astype(float)
    np.fill_diagonal(B, 0.0)

    h = np.ones(n_samples, dtype=float)
    a = np.ones(n_samples, dtype=float)

    def _norm(v):
        s = np.linalg.norm(v)
        return v if s == 0 else (v / s)

    h = _norm(h)
    a = _norm(a)

    for _ in range(100):
        h_new = B @ a
        a_new = B.T @ h_new
        h_new = _norm(h_new)
        a_new = _norm(a_new)

        if np.max(np.abs(h - h_new)) < 1e-8 and np.max(np.abs(a - a_new)) < 1e-8:
            h, a = h_new, a_new
            break
        h, a = h_new, a_new

    importance = h + a  # Im_i

    nn_idx = np.argsort(D, axis=1)[:, 1:k+1]
    nn_sets = [set(nn_idx[i]) for i in range(n_samples)]

    sort_D = np.sort(D, axis=1)
    sigma_i = sort_D[:, k]
    sigma_i = np.maximum(sigma_i, 1e-12)

    W = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            shared = nn_sets[i].intersection(nn_sets[j])
            im_max = 0.0 if len(shared) == 0 else float(np.max(importance[list(shared)]))

            denom = sigma_i[i] * sigma_i[j] * (1.0 + im_max) * sigma
            sij = np.exp(-(D[i, j] ** 2) / max(denom, 1e-12))
            W[i, j] = sij
            W[j, i] = sij

    np.fill_diagonal(W, 0.0)
    return W

