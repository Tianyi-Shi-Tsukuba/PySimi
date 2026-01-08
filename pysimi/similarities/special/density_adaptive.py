import numpy as np
from scipy.spatial.distance import pdist, squareform



# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def cnn(n, dist_matrix, epsilon):
    leq_epsilon = (dist_matrix <= epsilon)
    count = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            count[i, j] = np.sum(leq_epsilon[i] & leq_epsilon[j]) + 1
    return count


def get_epsilon(n, euclidean_distance, dist_matrix):
    order_euclidean_distance = np.sort(euclidean_distance)
    min_d = order_euclidean_distance[0]
    max_d = order_euclidean_distance[-1]
    mean_d = np.mean(euclidean_distance)

    order_dist_matrix = np.sort(dist_matrix, axis=1)
    nearest_neighbor = np.sort(order_dist_matrix[:, 1])
    max_n = nearest_neighbor[-1]
    mean_n = np.mean(nearest_neighbor)

    epsilon = 20 * mean_d + 54 * min_d + 13 - max_n - 6 * max_d - 65 * mean_n
    return epsilon

# ------------------------------------------------------------
# Main spectral clustering code
# ------------------------------------------------------------

def local_density_similarity(X: np.ndarray, k: int=2, sigma: float=0.15)-> np.ndarray:
    """
        Compute a local-density-adaptive similarity matrix.


        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        k : int, optional (default=2)
            Unused parameter (kept for API compatibility).
            Included to match the standard interface of similarity functions.
        sigma : float, optional (default=0.15)
            Base kernel bandwidth parameter. Smaller values produce sharper decay
            for distances; the effective bandwidth is further adjusted by local
            density C[i, j].

        Returns
        -------
        W : ndarray of shape (n_samples, n_samples)
            Symmetric similarity matrix with zeros on the diagonal. Higher values
            indicate stronger similarity under the local-density-aware kernel.

        Notes
        -----.
        - The epsilon value is estimated automatically from data statistics.
        """
    n = X.shape[0]

    euclidean_distance = pdist(X)
    dist_matrix = squareform(euclidean_distance)

    epsilon = get_epsilon(n, euclidean_distance, dist_matrix)

    C = cnn(n, dist_matrix, epsilon)

    W_ = np.exp(-dist_matrix / (2 * sigma * sigma * C))
    W = W_ - np.diag(np.diag(W_))

    return W

