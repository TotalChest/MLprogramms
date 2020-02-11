'''
Скрипт настройки нейронной сети по распознаванию геометрических фигур на изображении.

Обучающие данные должны быть заранее сгенерированы и записаны в src_file (смотри Data_Generation.py).

Настройка нейросети производится изменением переменных в начале файла (эти переменные закомментированы).
'''

import numpy as np
from scipy import optimize
import functions

save = 1  # Сохранить веса в файл?
save_file = 'save_weight.txt'   # Файл для сохранения весов
src_file = 'data_set.txt'  # Файл с подготовленными векторами изображений
size = (20, 20)  # Размер изображения
layers = (400, 32, 16, 3)  # Архитектура нейросети
options = {'maxiter': 500}  # Настройки оптимизатора
lambda_ = 1  # Параметр регуляризации
dictionary = {0: 'Circle', 1: 'Triangle', 2: 'Cross'}  # Обазначения меток

Y = []
X = []
file = open(src_file, 'r')
data = file.readlines()
for i in data:
    Y.append(int(i.split(':')[1]))
    X.append(list(map(int, i.split(':')[2].split(','))))
file.close()
X = np.array(X)  # Векторы изображений по строкам (m, 400)
Y = np.array(Y)  # Метки векторов (m)
m = Y.size  # Размер обучающей выборки

# Случайная инициализация весов
initTheta = []
for i in range(len(layers) - 1):
    initTheta.append(functions.randInitialize(layers[i + 1], layers[i] + 1))

# Преобразование всех весов в один вектор
init_nn_params = np.array([])
for i in range(len(layers) - 1):
    init_nn_params = np.concatenate([init_nn_params, initTheta[i].ravel()], axis=0)

# Функция стоимости
costFunction = lambda p: functions.nnCostFunction(p, layers, X, Y, lambda_)

# Минимизация функции стоимости
res = optimize.minimize(costFunction, init_nn_params, jac=True, method='TNC', options=options)
nn_params = res.x

# Сохранение весов в файл
if save:
    save_file = open(save_file, 'w')
    for i in range(len(nn_params)):
        save_file.write(('%.10f' % nn_params[i]) + ('' if i == len(nn_params)-1 else ','))
    save_file.close()

# Восстановление матриц весов
count = 0
Theta = []
for i in range(len(layers) - 1):
    Theta.append(
        np.reshape(nn_params[count: count + layers[i + 1] * (layers[i] + 1)], (layers[i + 1], (layers[i] + 1))))
    count += layers[i + 1] * (layers[i] + 1)

# Классификация новых данных
while 1:
    path = input('Введите путь к изображению: ')
    if path == '':
        break
    try:
        image = functions.generate_vector(path, size)
        functions.displayData(image, size)
        print('Изображение: ' + dictionary[functions.predict(image, Theta)])
    except:
        print('Файл не найден!')
