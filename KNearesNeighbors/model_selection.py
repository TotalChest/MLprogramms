from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import KNNClassifier
from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, batched=False, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    results = defaultdict(lambda: np.zeros(cv.get_n_splits(X)))

    for i, index in enumerate(cv.split(X)):
        train_index, test_index = index
        if batched:
            model = BatchedKNNClassifier(n_neighbors=max(k_list), **kwargs)
            model.set_batch_size(1000)
        else:
            model = KNNClassifier(n_neighbors=max(k_list), **kwargs)
        model.fit(X[train_index, :], y[train_index])
        distances, indices = model.kneighbors(X[test_index, :],
                                              return_distance=True)
        for k in k_list:
            pred = model._predict_precomputed(indices[:, :k], distances[:, :k])
            results[k][i] = scorer(y[test_index], pred)

    return results
