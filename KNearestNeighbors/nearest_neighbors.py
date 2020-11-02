import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    if top > ranks.shape[axis]:
        top = ranks.shape[axis]
    indices = np.take(np.argpartition(ranks, top - 1, axis=axis),
                      np.arange(top), axis=axis)
    top_ranks = np.take_along_axis(ranks, indices, axis=axis)
    indices = np.take_along_axis(indices, np.argsort(top_ranks), axis=axis)
    if return_ranks:
        distances = np.take_along_axis(ranks, indices, axis=axis)
        return (distances, indices)
    return indices


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        ranks = self._metric_func(X, self._X)
        result = get_best_ranks(ranks, self.n_neighbors, 1, return_distance)
        return result
