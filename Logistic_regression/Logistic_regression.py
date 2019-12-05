#Prediction of passing to the university according to the scores of two exams

import os
import numpy as np
from scipy import optimize

data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
y = data[:, 2]
X = np.concatenate([np.ones((data[:, :2].shape[0], 1)), data[:, :2]], axis=1)


def sigmoid(z):
    z = np.array(z)
    g = 1 / (1 + np.exp(-1 * z))
    return g


def costFunction(theta, X, y):
    m = y.size
    n = theta.size
    J = 0
    grad = np.zeros(theta.shape)

    for i in range(m):
        J += -y[i] * np.log(sigmoid(np.dot(theta, X[i]))) - (1 - y[i]) * np.log(1 - sigmoid(np.dot(theta, X[i])))
        for j in range(n):
            grad[j] += (sigmoid(np.dot(theta, X[i])) - y[i]) * X[i, j]

    return J / m, grad / m


initial_theta = np.zeros(X.shape[1])
res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options={'maxiter': 400})

theta = res.x

x1 = int(input("First exam: "))
x2 = int(input("Second exam: "))

print("Yes!" if sigmoid(np.dot(np.array([1, x1, x2]), theta)) > 0.5 else "No.")
