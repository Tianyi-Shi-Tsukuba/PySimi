import numpy as np

def cosine_similarity(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Cosine similarity between row vectors."""
    X = np.asarray(X, dtype=float)
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, np.finfo(float).eps)
        Xn = X / norms
    else:
        Xn = X
    S = Xn @ Xn.T
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S
