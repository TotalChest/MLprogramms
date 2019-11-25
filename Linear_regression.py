import os
import numpy as np

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

m = y.size


def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    m = y.size

    for i in range(X.shape[1]):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        X_norm[:, i] = (X_norm[:, i] - mu[i])
        X_norm[:, i] = (X_norm[:, i] / sigma[i])

    X_norm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    m = y.size
    return (1 / (2 * m)) * np.dot((np.dot(X, theta) - y).transpose(), (np.dot(X, theta) - y))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.size

    J_history = []

    for i in range(num_iters):
        q = [0, 0, 0]
        for j in range(m):
            for k in range(3):
                q[k] += (np.dot(theta, X[j]) - y[j]) * X[j, k]

        for l in range(3):
            theta[l] = theta[l] - (alpha / m) * q[l]

        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


alpha = 0.1
num_iters = 250
theta = np.ones(3)

X_norm, mu, sigma = featureNormalize(X)
theta, J_history = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)

m = int(input('Square: '))
k = int(input('Rooms : '))

m = (m - mu[0]) / sigma[0]
k = (k - mu[1]) / sigma[1]

print(np.dot(theta, [1, m, k]))
