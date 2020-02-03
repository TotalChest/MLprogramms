import numpy as np
from scipy import optimize
import functions


src_file = 'data_set.txt'	# Файл с подготовленными векторами
input_layer_size = 400	# Длина входного вектора изображения
hidden_layer_1_size = 32	# Количество нейронов первого скрытого слоя
hidden_layer_2_size = 16	# Количество нейронов второго скрытого слоя
num_labels = 3	# Количество меток
options = {'maxiter': 100}	# Настройки оптимизатора
lambda_ = 0	# Параметр регуляризации


dictionary = {0: 'Circle', 1: 'Triangle', 2: 'Cross'}
Y = []
X = []
file = open(src_file, 'r')
data = file.readlines()
for i in data:
    Y.append(int(i.split(':')[1]))
    X.append(list(map(int, i.split(':')[2].split(','))))
X = np.array(X)	# Векторы по строкам (m, 400)
Y = np.array(Y)	# Метки векторов (m)
m = Y.size	# Размер обучающей выборки
layers = (input_layer_size, hidden_layer_1_size, hidden_layer_2_size, num_labels)	# Корреж архитектуры неросети


initial_Theta1 = functions.randInitialize(hidden_layer_1_size, input_layer_size + 1)	# Theta1 (32, 401)
initial_Theta2 = functions.randInitialize(hidden_layer_2_size, hidden_layer_1_size + 1)	# Theta2 (16, 33)
initial_Theta3 = functions.randInitialize(num_labels, hidden_layer_2_size + 1)	# Theta3 (3, 17)

initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
initial_nn_params = np.concatenate([initial_nn_params, initial_Theta3.ravel()], axis=0)

'''
J,_=functions.nnCostFunction(initial_nn_params, (input_layer_size,
                                                  hidden_layer_1_size,
                                                  hidden_layer_2_size,
                                                  num_labels), X, Y, 0)

print(J,_)
'''
costFunction = lambda p: functions.nnCostFunction(initial_nn_params, layers, X, Y, lambda_)
nn_params, J_history = functions.gradientDescent(costFunction, initial_nn_params, 0.1, 100)
print(J_history)
'''
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

nn_params = res.x
print(costFunction(nn_params))
'''

count = 0
Theta = []	# Theta[0], Theta[1], Theta[2] - соответствующие матрицы весов 
for i in range(len(layers)-1):
	Theta.append(np.reshape(nn_params[count: count + layers[i+1] * (layers[i] + 1)], (layers[i+1], (layers[i] + 1))))
	count += layers[i+1] * (layers[i] + 1)

while 1:
    path = input('Введите путь к изображению: ')
    print('Изображение: ' + dictionary[functions.predict(functions.generate_vector(path), Theta)])
