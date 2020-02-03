from PIL import Image
import os


folders = ['Circle', 'Triangle', 'Cross']
dictionary = dict(zip(folders, range(len(folders))))
handle = open("data_set.txt", "w")


for dir in folders:
    for image_num in range(len(os.listdir(dir))):
        handle.write(dir + ':' + str(dictionary[dir]) + ':')
        image = Image.open(dir + '/' + str(image_num + 1) + '.bmp')
        pix = image.load()
        for i in range(20):
            for j in range(20):
                handle.write(('1' if int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3) < 128 else '0') + (
                    ',' if (i + j != 38) else ''))
        handle.write('\n')
handle.close()