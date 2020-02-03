import numpy as np
from scipy import optimize
import functions

dictionary = {0: 'Circle', 1: 'Triangle', 2: 'Cross'}

Y = []
X = []
file = open('data_set.txt', 'r')
data = file.readlines()
for i in data:
    Y.append(i.split(':')[1])
    X.append(list(map(int, i.split(':')[2].split(','))))

X = np.array(X)  # labels
Y = np.array(Y)  # data vectors
m = Y.size  # size of learning set

input_layer_size = 400
hidden_layer_1_size = 32
hidden_layer_2_size = 16
output_layer_size = 3

weghts_to_hidden_1 = np.random.random((32, 401))
weights_to_hidden_2 = np.random.random((16, 33))
weights_to_output = np.random.random((3, 17))

initial_nn_params = np.concatenate([weghts_to_hidden_1.ravel(), weights_to_hidden_2.ravel()], axis=0)
initial_nn_params = np.concatenate([initial_nn_params, weights_to_output.ravel()], axis=0)

options = {'maxiter': 500}

lambda_ = 0

costFunction = lambda p: functions.nnCostFunction(p, input_layer_size,
                                                  hidden_layer_1_size,
                                                  hidden_layer_2_size,
                                                  output_layer_size, X, Y, lambda_)

res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

nn_params = res.x

weights_h_1 = np.reshape(nn_params[:hidden_layer_1_size * (input_layer_size + 1)],
                         (hidden_layer_1_size, (input_layer_size + 1)))

weights_h_2 = np.reshape(nn_params[(hidden_layer_1_size * (input_layer_size + 1)):(hidden_layer_1_size * (
        input_layer_size + 1)) + hidden_layer_2_size * (hidden_layer_1_size + 1)],
                         (hidden_layer_2_size, (hidden_layer_1_size + 1)))

weights_o = np.reshape(
    nn_params[(hidden_layer_1_size * (input_layer_size + 1)) + hidden_layer_2_size * (hidden_layer_1_size + 1):],
    (output_layer_size, (hidden_layer_2_size + 1)))

while 1:
    path = input('Введите путь к изображению (20*20): ')
    print('Изображение: ' + dictionary[functions.predict(functions.generate_vector(path), weights_h_1, weights_h_2, weights_o)])