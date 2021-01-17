from PIL import Image

handle = open("data.txt", "w")
for k in range(100):
    image = Image.open('Learning_numbers/' + str(k // 10 + 1) + '(' + str(k % 10 + 1) +').jpg')
    pix = image.load()
    for i in range(20):
        for j in range(20):
            handle.write(str(int((pix[i, j][0] + pix[i, j][1] + pix[i, j][2]) / 3)) + ",")
    handle.write(str(k // 10 + 1) + "\n")
handle.close()
