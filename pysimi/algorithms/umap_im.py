"""
umap_im.py — Robust UMAP wrapper supporting:
- Precomputed similarities or distance matrices
- Standard feature matrices
- Unified interface with similarity → distance conversion
- Compatibility with umap-learn

Key behaviors
-------------
1. If S (similarity matrix) is provided:
      - Converted to a distance matrix via `_sim_to_dist()`.
      - The UMAP metric is automatically set to 'precomputed' unless overridden.

2. If X (feature matrix) is provided:
      - Standard UMAP is applied (metric='Euclidean' by default).

3. Additional UMAP parameters (n_neighbors, min_dist, n_components, etc.) are fully exposed.
"""
import numpy as np
from typing import Optional, Literal, Any, Tuple

try:
    import umap
except Exception as e:
    raise ImportError("umap-learn is required for umap") from e


SimToDist = Literal["max_minus", "one_minus", "reciprocal", "exp_neg"]


def _sim_to_dist(S: np.ndarray, method: SimToDist = "max_minus") -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square similarity matrix")
    # Make symmetric
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, S.diagonal())

    if method == "max_minus":
        m = np.max(S)
        D = m - S
    elif method == "one_minus":
        S1 = np.clip(S, 0.0, 1.0)
        D = 1.0 - S1
    elif method == "reciprocal":
        eps = 1e-8
        D = 1.0 / (S + eps) - 1.0
        D[D < 0] = 0.0
    elif method == "exp_neg":
        D = np.exp(-S)
    else:
        raise ValueError(f"Unknown sim_to_dist method: {method}")

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    return D


def umap_rdc(
    S: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    n_components: int = 2,
    sim_to_dist: SimToDist = "max_minus",
    metric: Optional[str] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
)->Tuple[np.ndarray, Any]:
    if S is None and X is None:
        raise ValueError("Provide either S (similarity) or X (features).")

    if S is not None:
        D = _sim_to_dist(S, method=sim_to_dist)
        use_metric = "precomputed" if metric is None else metric
        data = D

    else:
        use_metric = 'euclidean' if metric is None else metric
        data = X

    umap_kwargs = dict(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=use_metric,
        random_state=random_state,
    )
    umap_kwargs.update(kwargs)

    reducer = umap.UMAP(**umap_kwargs)
    embedding = reducer.fit_transform(data)
    return embedding, reducer