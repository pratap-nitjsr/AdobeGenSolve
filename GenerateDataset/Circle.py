import cv2
import numpy as np
import random
import os


def GenerateCirclesDataset(purpose, numData):
    DEST_IMG_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'images')
    DEST_LABEL_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'labels')

    if not os.path.exists(DEST_IMG_FOLDER):
        os.makedirs(DEST_IMG_FOLDER)
    if not os.path.exists(DEST_LABEL_FOLDER):
        os.makedirs(DEST_LABEL_FOLDER)

    for i in range(numData):
        image = np.zeros((128, 128), dtype=np.uint8)
        rad = random.randint(5, 60)
        x = random.randint(rad, 128 - rad)
        y = random.randint(rad, 128 - rad)

        cv2.circle(image, (x, y), rad, 255, 1)
        imgPath = os.path.join(DEST_IMG_FOLDER, f'Circle_{i}.png')
        cv2.imwrite(imgPath, image)

        labelPath = os.path.join(DEST_LABEL_FOLDER, f'Circle_{i}.txt')
        with open(labelPath, 'w') as f:
            class_id = 0
            x_center = x / 128.0
            y_center = y / 128.0
            width = (2 * rad) / 128.0
            height = (2 * rad) / 128.0

            f.write(f"{class_id} {x_center} {y_center} {width} {height} ")

            num_points = 100
            for theta in np.linspace(0, 2 * np.pi, num_points):
                px = (x + rad * np.cos(theta)) / 128.0
                py = (y + rad * np.sin(theta)) / 128.0
                f.write(f"{px} {py} ")
            f.write("\n")
