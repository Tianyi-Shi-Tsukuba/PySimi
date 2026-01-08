"""
tsne.py — Robust t-SNE helper that supports precomputed similarities/distances
and is compatible across scikit-learn versions (n_iter -> max_iter transition).

Key behaviors:
- If S (similarity) is given, converts to a distance matrix and uses metric='precomputed'.
- When metric='precomputed', forces init='random' (init='pca' is invalid).
- Accepts both n_iter and max_iter; prefers max_iter in newer sklearn, falls back to n_iter otherwise.
"""

from typing import Optional, Tuple, Literal, Union, Any
import numpy as np

try:
    from sklearn.manifold import TSNE
except Exception as e:
    raise ImportError("scikit-learn is required for t-SNE") from e


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


def tsne(
    S: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    n_iter: int = 1000,
    max_iter: Optional[int] = None,
    init: Union[str, np.ndarray] = "auto",
    random_state: Optional[int] = 0,
    sim_to_dist: SimToDist = "max_minus",
    metric: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Any]:
    """
    Run t-SNE with either a feature matrix X or a similarity matrix S.
    """
    if S is None and X is None:
        raise ValueError("Provide either S (similarity) or X (features).")

    if S is not None:
        D = _sim_to_dist(S, method=sim_to_dist)
        use_metric = "precomputed" if metric is None else metric
        if use_metric == "precomputed":
            if isinstance(init, str) and init.lower() in ("pca", "auto"):
                init = "random"
        data = D
    else:
        use_metric = "euclidean" if metric is None else metric
        data = X
        if isinstance(init, str) and init.lower() == "auto":
            init = "pca"

    mi = max_iter if max_iter is not None else n_iter

    tsne_kwargs = dict(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        random_state=random_state,
        metric=use_metric,
    )
    tsne_kwargs.update(kwargs)

    model = None
    try:
        model = TSNE(max_iter=mi, **tsne_kwargs)  # sklearn >=1.5
    except TypeError:
        model = TSNE(n_iter=mi, **tsne_kwargs)   # older sklearn

    Y = model.fit_transform(data)
    return Y, model
