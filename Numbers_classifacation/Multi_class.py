import numpy as np
from scipy import optimize
from PIL import Image

data = np.loadtxt('data.txt', delimiter=',')
X = data[:, :400]
y = data[:, 400]
y[y == 10] = 0
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def CostFunction(theta_, X_, y_, lambda_):
    m_ = y_.size

    if y_.dtype == bool:
        y_ = y_.astype(int)

    temp = theta_.copy()
    temp[0] = 0
    J = - np.sum(np.dot(y_, np.log(sigmoid(np.dot(X_, theta_))))) - np.sum(
        np.dot((1 - y_), np.log(1 - sigmoid(np.dot(X_, theta_))))) + (lambda_ / 2) * np.sum(temp ** 2)

    grad_ = np.dot(X_.transpose(), sigmoid(np.dot(X_, theta_)) - y_) + lambda_ * temp

    return J / m_, grad_ / m_


theta = np.ones((10, 401))
for i in range(10):
    initial_theta = np.zeros(X.shape[1])
    res = optimize.minimize(CostFunction,
                            initial_theta,
                            (X, (y == i), 0.9),
                            jac=True,
                            method='TNC',
                            options={'maxiter': 200})
    theta[i] = res.x.copy()

x = np.zeros(401)
x[0] = 1
img = str(input("Input image: "))
image = Image.open(img)
pix = image.load()
for i in range(20):
    for j in range(20):
        x[i * 20 + j + 1] = int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3)

predict = np.argmax(sigmoid(np.dot(theta, x)))

print(predict)
