import numpy as np
from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nnCostFunction(nn_params, layers, X, Y, lambda_=0.0):
	print(1)
	
	count = 0
	Theta = []	# Theta[0], Theta[1], Theta[2] - соответствующие матрицы весов 
	for i in range(len(layers)-1):
		Theta.append(np.reshape(nn_params[count: count + layers[i+1] * (layers[i] + 1)], (layers[i+1], (layers[i] + 1))))
		count += layers[i+1] * (layers[i] + 1)

	Theta_grad = []	# Theta_grad[0], Theta_grad[1], Theta_grad[2] - соответствующие матрицы частных производных функции ошибки по весам
	for i in range(len(layers)-1):
		Theta_grad.append(np.zeros(Theta[i].shape))

	m = Y.size	# Размер обучающей выборки
	J = 0	# Значение функции ошибки

	X = np.concatenate([np.ones((m, 1)), X], axis=1)
	Y1 = np.zeros((m, layers[-1]))
	for i in range(m):
		Y1[i, Y[i]] = 1


	J = (-1 / m) * np.sum(np.array(
                [np.sum(np.log(sigmoid(np.dot(Theta[2], np.concatenate([np.ones(1), sigmoid(np.dot(Theta[1], np.concatenate([np.ones(1), sigmoid(np.dot(Theta[0], X[i, :]))])))])))) * Y1[i, :] +
                        np.log(1 - sigmoid(np.dot(Theta[2], np.concatenate([np.ones(1), sigmoid(np.dot(Theta[1], np.concatenate([np.ones(1), sigmoid(np.dot(Theta[0], X[i, :]))])))])))) * (1 - Y1[i, :]))
                 for i in range(m)])) + (lambda_ / (2 * m)) * (np.sum(Theta[0][:, 1:] ** 2) + np.sum(Theta[1][:, 1:] ** 2) + np.sum(Theta[2][:, 1:] ** 2))


	for i in range(m):
		a1 = X[i, :]

		a2 = sigmoid(np.dot(Theta[0], a1))
		a2 = np.concatenate([np.ones(1), a2])

		a3 = sigmoid(np.dot(Theta[1], a2))
		a3 = np.concatenate([np.ones(1), a3])

		a4 = sigmoid(np.dot(Theta[2], a3))

		d4 = a4 - Y1[i, :]
		d3 = np.dot(Theta[2].T, d4) * a3 * (1 - a3)
		d2 = np.dot(Theta[1].T, d3[1:]) * a2 * (1 - a2)

		Theta_grad[0] += np.dot(d2[1:].reshape(-1, 1), a1.reshape(1, -1))
		Theta_grad[1] += np.dot(d3[1:].reshape(-1, 1), a2.reshape(1, -1))
		Theta_grad[2] += np.dot(d4.reshape(-1, 1), a3.reshape(1, -1))


	Theta_grad[0] += lambda_ * np.concatenate([np.zeros((layers[1], 1)), Theta[0][:, 1:]], axis=1)
	Theta_grad[1] += lambda_ * np.concatenate([np.zeros((layers[2], 1)), Theta[1][:, 1:]], axis=1)
	Theta_grad[2] += lambda_ * np.concatenate([np.zeros((layers[3], 1)), Theta[2][:, 1:]], axis=1)
	Theta_grad[0] /= m
	Theta_grad[1] /= m
	Theta_grad[2] /= m


	grad = np.concatenate([Theta_grad[0].ravel(), Theta_grad[1].ravel()])
	grad = np.concatenate([grad, Theta_grad[2].ravel()])

	return J, grad


def randInitialize(In, Out, epsilon_init=0.12):
	W = np.zeros((In, Out))
	W = np.random.rand(In, Out) * 2 * epsilon_init - epsilon_init
	return W


def computeNumericalGradient(J, theta, e=1e-4):
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad

def checkNNGradients(nnCostFunction, lambda_=0):

	# Небольшие рармеры д
	input_layer_size = 3
	hidden_layer_1_size = 4
	hidden_layer_2_size = 3
	num_labels = 3
	m = 5

	Theta1 = debugInitializeWeights(hidden_layer_1_size, input_layer_size)
	Theta2 = debugInitializeWeights(hidden_layer_2_size, hidden_layer_1_size)
	Theta3 = debugInitializeWeights(num_labels, hidden_layer_2_size)

		# Reusing debugInitializeWeights to generate X
	X = debugInitializeWeights(m, input_layer_size - 1)
	y = np.arange(1, 1+m) % num_labels
		# print(y)
		# Unroll parameters
	nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
	nn_params = np.concatenate([nn_params, Theta3.ravel()])

		# short hand for cost function
	costFunc = lambda p: nnCostFunction(p, (input_layer_size, hidden_layer_1_size, hidden_layer_2_size,
		                                    num_labels), X, y, lambda_)
	cost, grad = costFunc(nn_params)
	numgrad = computeNumericalGradient(costFunc, nn_params)

		# Visually examine the two gradient computations.The two columns you get should be very similar.
	print(np.stack([numgrad, grad], axis=1))
	print('The above two columns you get should be very similar.')
	print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

		# Evaluate the norm of the difference between two the solutions. If you have a correct
		# implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
		# should be less than 1e-9.
	diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

	print('If your backpropagation implementation is correct, then \n'
		      'the relative difference will be small (less than 1e-9). \n'
		      'Relative Difference: %g' % diff)

def debugInitializeWeights(fan_out, fan_in):

    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def gradientDescent(costFunction, initial_nn_params, alpha, num_iters):
	theta = initial_nn_params.copy()
	J_history = [] 
    
	for i in range(num_iters):
		q=0
		_, q = costFunction(theta)
		print(q)
		theta -= 10*q
		J_history.append(costFunction(theta)[0])
    
	return theta, J_history


def predict(inputs, Theta):
	inputs = np.concatenate([np.ones(1), inputs])
	h1 = sigmoid(np.dot(Theta[0], inputs))
	h1 = np.concatenate([np.ones(1), h1])
	h2 = sigmoid(np.dot(Theta[1], h1))
	h2 = np.concatenate([np.ones(1), h2])
	o = sigmoid(np.dot(Theta[2], h2))
	print(o)
	return np.argmax(o)


def generate_vector(path):
	data = []
	image = Image.open(path)
	image.thumbnail((20, 20))
	pix = image.load()
	for i in range(20):
		for j in range(20):
			data.append(1 if int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3) < 128 else 0)
	data = np.array(data)
	return data
