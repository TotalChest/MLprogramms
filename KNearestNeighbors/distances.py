import numpy as np


def euclidean_distance(x, y):
    d = np.dot(x, y.T)
    x_2 = np.sum(x**2, axis=1)
    y_2 = np.sum(y**2, axis=1)
    return np.sqrt(x_2[:, None] + y_2[None, :] - 2 * d)


def cosine_distance(x, y):
    d = np.dot(x, y.T)
    x_2 = np.sqrt(np.sum(x**2, axis=1))
    y_2 = np.sqrt(np.sum(y**2, axis=1))
    return 1 - d / x_2[:, None] / y_2[None, :]
