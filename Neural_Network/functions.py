'''
Файл с основными функциями.
Функции помечены комментариями.
'''

import numpy as np
from PIL import Image
from matplotlib import pyplot


# Отображение фигуры
def displayData(X, figsize=(20, 20)):
    pyplot.imshow(X.reshape(figsize[0], figsize[1], order='F'), cmap='Greys')
    pyplot.axis('off')
    pyplot.show()


# Прямое рпспространение
def feedforward(inputs, Theta):
    x = inputs.copy()
    for i in range(len(Theta)):
        x = sigmoid(np.dot(Theta[i], np.concatenate([np.ones(1), x])))
    return x


# Функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Функция стоимости
def nnCostFunction(nn_params, layers, X, Y, lambda_=0.0):
    count = 0
    Theta = []
    for i in range(len(layers) - 1):
        Theta.append(
            np.reshape(nn_params[count: count + layers[i + 1] * (layers[i] + 1)], (layers[i + 1], (layers[i] + 1))))
        count += layers[i + 1] * (layers[i] + 1)

    Theta_grad = []
    for i in range(len(Theta)):
        Theta_grad.append(np.zeros(Theta[i].shape))

    m = Y.size
    J = 0
    Y1 = np.zeros((m, layers[-1]))
    for i in range(m):
        Y1[i, Y[i]] = 1

    for i in range(m):
        J += (-1 / m) * np.sum(np.log(feedforward(X[i, :], Theta)) * Y1[i, :] + np.log(
            1 - feedforward(X[i, :], Theta)) * (1 - Y1[i, :]))

    for i in range(len(Theta)):
        J += (lambda_ / (2 * m)) * (np.sum(Theta[i][:, 1:] ** 2))

    for i in range(m):
        a = []
        d = []

        a.append(np.concatenate([np.ones(1), X[i, :]]))
        for j in range(len(Theta)):
            a.append(np.concatenate([np.ones(1), sigmoid(np.dot(Theta[j], a[j]))]))

        d.append(a[len(Theta)] - np.concatenate([np.ones(1), Y1[i, :]]))
        for j in range(len(Theta) - 1):
            d.insert(0, np.dot(Theta[len(Theta) - j - 1].T, d[0][1:]) * a[len(Theta) - j - 1] * (
                    1 - a[len(Theta) - j - 1]))

        for j in range(len(Theta)):
            Theta_grad[j] += np.dot(d[j][1:].reshape(-1, 1), a[j].reshape(1, -1))

    for i in range(len(Theta)):
        Theta_grad[i] += lambda_ * np.concatenate([np.zeros((layers[i + 1], 1)), Theta[i][:, 1:]], axis=1)
        Theta_grad[i] /= m

    grad = np.array([])
    for i in range(len(Theta)):
        grad = np.concatenate([grad, Theta_grad[i].ravel()], axis=0)

    return J, grad


# Случайная инициализация матрицы
def randInitialize(In, Out, epsilon_init=1):
    W = np.zeros((In, Out))
    W = np.random.rand(In, Out) * 2 * epsilon_init - epsilon_init
    return W


# Классификация новых данных
def predict(inputs, Theta):
    o = feedforward(inputs, Theta)
    return np.argmax(o)


# Создание вектора из изображения
def generate_vector(path, size=(20, 20)):
    data = []
    image = Image.open(path)
    image.resize(size)
    pix = image.load()
    for i in range(size[0]):
        for j in range(size[1]):
            data.append(1 if int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3) < 128 else 0)
    data = np.array(data)
    return data
