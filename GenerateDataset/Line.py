import cv2
import numpy as np
import random
import os


def GenerateLinesDataset(purpose, numData):
    DEST_IMG_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'images')
    DEST_LABEL_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'labels')

    if not os.path.exists(DEST_IMG_FOLDER):
        os.makedirs(DEST_IMG_FOLDER)
    if not os.path.exists(DEST_LABEL_FOLDER):
        os.makedirs(DEST_LABEL_FOLDER)

    for i in range(numData):
        image = np.zeros((128, 128), dtype=np.uint8)
        x1 = random.randint(0, 128)
        y1 = random.randint(0, 128)
        x2 = random.randint(0, 128)
        y2 = random.randint(0, 128)

        cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        imgPath = os.path.join(DEST_IMG_FOLDER, f'Line_{i}.png')
        cv2.imwrite(imgPath, image)

        labelPath = os.path.join(DEST_LABEL_FOLDER, f'Line_{i}.txt')
        with open(labelPath, 'w') as f:
            class_id = 2
            x_center = ((x1 + x2) / 2) / 128.0
            y_center = ((y1 + y2) / 2) / 128.0
            width = abs(x2 - x1) / 128.0
            height = abs(y2 - y1) / 128.0

            f.write(f"{class_id} {x_center} {y_center} {width} {height} ")

            f.write(f"{x1 / 128.0} {y1 / 128.0} {x2 / 128.0} {y2 / 128.0} ")
            f.write("\n")
