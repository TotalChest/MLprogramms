'''
Скрипт генерации векторов изображений

Изображения хранятся в папках с названиями из folders. Названия изображений "i.bmp", где i - номера изображений по порядку.

Вектора  записываются в один файл в следующем формате:
Class_name:Class_number:0,0,1,0,1,1,0,1,0,0,0,1, ... ,1,0,0,0,0,0,1,0
'''


from PIL import Image
import os


dst_file = "data_set.txt"	# Файл векторов
size = 20	# Размер стороны квадатного изображения


folders = ['Circle', 'Triangle', 'Cross']
dictionary = dict(zip(folders, range(len(folders))))
handle = open(dst_file, "w")


for dir in folders:
	for image_num in range(len(os.listdir(dir))):
		handle.write(dir + ':' + str(dictionary[dir]) + ':')
		image = Image.open(dir + '/' + str(image_num + 1) + '.bmp')
		image.thumbnail((size, size))
		pix = image.load()
		for i in range(size):
			for j in range(size):
				handle.write(('1' if int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3) < 128 else '0') + (',' if (i + j != 2 * (size - 1)) else ''))
		handle.write('\n')
handle.close()
