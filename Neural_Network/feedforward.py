'''
Предсказание с весами из файла (Прямое распространение)
'''

import numpy as np
from PIL import Image


layers = (400, 32, 16, 3)  # Архитектура нейросети
size = (20, 20)  # Размер изображения
save_file = 'save_weight.txt'   # Файл с весами
dictionary = {0: 'Circle', 1: 'Triangle', 2: 'Cross'}  # Обазначения меток


# Классификация новых данных (прямое распространение)
def predict(inputs, Theta):
	x = inputs.copy()
	for i in range(len(Theta)):
		x = sigmoid(np.dot(Theta[i], np.concatenate([np.ones(1), x])))
	return np.argmax(x)


# Функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

# Загрузка весов
save_file = open(save_file, 'r')
nn_params = np.array(list(map(float, save_file.read().split(','))))
print(nn_params)

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
    	image = generate_vector(path, size)
    	print('Изображение: ' + dictionary[predict(image, Theta)])
    except:
    	print('Файл не найден!')