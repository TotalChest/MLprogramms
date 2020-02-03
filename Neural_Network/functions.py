import numpy as np
from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# lambda = регуляризация
def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_1_size,
                   hidden_layer_2_size,
                   output_layer_size,
                   X, Y, lambda_=0.0):
    weights_h_1 = np.reshape(nn_params[:hidden_layer_1_size * (input_layer_size + 1)],
                             (hidden_layer_1_size, (input_layer_size + 1)))

    weights_h_2 = np.reshape(nn_params[(hidden_layer_1_size * (input_layer_size + 1)):(hidden_layer_1_size * (
            input_layer_size + 1)) + hidden_layer_2_size * (hidden_layer_1_size + 1)],
                             (hidden_layer_2_size, (hidden_layer_1_size + 1)))

    weights_o = np.reshape(
        nn_params[(hidden_layer_1_size * (input_layer_size + 1)) + hidden_layer_2_size * (hidden_layer_1_size + 1):],
        (output_layer_size, (hidden_layer_2_size + 1)))

    m = Y.size

    J = 0  # значение ошибки

    # градиент для каждого веса
    weights_h_1_grad = np.zeros(weights_h_1.shape)
    weights_h_2_grad = np.zeros(weights_h_2.shape)
    weights_o_grad = np.zeros(weights_o.shape)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)  # add biases
    Y1 = np.zeros((m, output_layer_size))
    for i in range(m):
        Y1[i, int(Y[i])] = 1

    J = (-1 / m) * np.sum(np.array([np.sum(np.log(
        sigmoid(np.dot(weights_o, np.concatenate([np.ones(1), sigmoid(
            np.dot(weights_h_2, np.concatenate([np.ones(1), sigmoid(np.dot(weights_h_1, X[i, :]))])))])))) * Y1[i, :] +
                                           np.log(1 - sigmoid(np.dot(weights_o, np.concatenate([np.ones(1), sigmoid(
                                               np.dot(weights_h_2, np.concatenate(
                                                   [np.ones(1), sigmoid(np.dot(weights_h_1, X[i, :]))])))])))) * (
                                                   1 - Y1[i, :]))
                                    for i in range(m)])) + (lambda_ / (2 * m)) * (
                np.sum(weights_h_1[:, 1:] ** 2) + np.sum(weights_h_2[:, 1:] ** 2) + np.sum(weights_o[:, 1:] ** 2))

    for i in range(m):
        a1 = X[i, :]
        a2 = sigmoid(np.dot(weights_h_1, a1))
        a2 = np.concatenate([np.ones(1), a2])
        a3 = sigmoid(np.dot(weights_h_2, a2))
        a3 = np.concatenate([np.ones(1), a3])
        a4 = sigmoid(np.dot(weights_o, a3))

        d4 = a4 - Y1[i, :]
        d3 = np.dot(weights_o.T, d4) * sigmoid_derivative(a3)
        d2 = np.dot(weights_h_2.T, d3[1:]) * sigmoid_derivative(a2)

        weights_h_1_grad += np.dot(d2[1:].reshape(-1, 1), a1.reshape(1, -1))
        weights_h_2_grad += np.dot(d3[1:].reshape(-1, 1), a2.reshape(1, -1))
        weights_o_grad += np.dot(d4.reshape(-1, 1), a3.reshape(1, -1))

    weights_h_1_grad += lambda_ * np.concatenate([np.zeros((hidden_layer_1_size, 1)), weights_h_1[:, 1:]], axis=1)
    weights_h_2_grad += lambda_ * np.concatenate([np.zeros((hidden_layer_2_size, 1)), weights_h_2[:, 1:]], axis=1)
    weights_o_grad += lambda_ * np.concatenate([np.zeros((output_layer_size, 1)), weights_o[:, 1:]], axis=1)
    weights_h_1_grad /= m
    weights_h_2_grad /= m
    weights_o_grad /= m

    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([weights_h_1_grad.ravel(order=order), weights_h_2_grad.ravel(order=order)])
    grad = np.concatenate([weights_h_1_grad.ravel(), weights_h_2_grad.ravel()])
    grad = np.concatenate([grad, weights_o_grad.ravel()])
    return J, grad


def predict(inputs, w1, w2, w3):
    inputs = np.concatenate([np.ones(1), inputs])
    h1 = sigmoid(np.dot(w1, inputs))
    h1 = np.concatenate([np.ones(1), h1])
    h2 = sigmoid(np.dot(w2, h1))
    h2 = np.concatenate([np.ones(1), h2])
    o = sigmoid(np.dot(w3, h2))
    return np.argmax(o)


def generate_vector(path):
    data = []
    image = Image.open(path)
    pix = image.load()
    for i in range(20):
        for j in range(20):
            data.append(1 if int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3) < 128 else 0)
    data = np.array(data)
    return data
