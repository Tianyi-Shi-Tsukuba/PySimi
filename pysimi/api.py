from importlib import import_module
from typing import Any, Callable, Dict

_REGISTRY: Dict[str, str] = {
    # Classic similarities
    "rbf":        "pysimi.similarities.classic.rbf:rbf_gaussian",
    "cosine":     "pysimi.similarities.classic.cosine:cosine_similarity",
    "pearson":    "pysimi.similarities.classic.correlation:pearson_correlation",
    "spearman":   "pysimi.similarities.classic.correlation:spearman_correlation",
    "poly":       "pysimi.similarities.classic.polynomial:polynomial_kernel",
    "sigmoid":    "pysimi.similarities.classic.sigmoid:sigmoid_kernel",

    # Special similarities 
    "nsnn":           "pysimi.similarities.special.nsnn:number_shared_neighbors_similarity",
    "csnn":           "pysimi.similarities.special.csnn:close_shared_neighbors_similarity",
    "adaptive":       "pysimi.similarities.special.adaptive:adaptive_similarity",
    "kerneladaptive": "pysimi.similarities.special.kernel_adaptive:kernel_adaptive_similarity",
    "knn":  "pysimi.similarities.special.sckNN:knn"
}

def _resolve(name: str) -> Callable[..., Any]:
    try:
        mod, func = _REGISTRY[name].split(":")
    except KeyError as e:
        raise ValueError(f"Unknown similarity method: {name}") from e
    return getattr(import_module(mod), func)

def build_similarity(X, method: str = "rbf", **kwargs):
    fn = _resolve(method)
    return fn(X, **kwargs)