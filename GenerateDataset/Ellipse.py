import cv2
import numpy as np
import random
import os


def GenerateEllipseDataset(purpose, numData):
    DEST_IMG_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'images')
    DEST_LABEL_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'labels')

    if not os.path.exists(DEST_IMG_FOLDER):
        os.makedirs(DEST_IMG_FOLDER)
    if not os.path.exists(DEST_LABEL_FOLDER):
        os.makedirs(DEST_LABEL_FOLDER)

    for i in range(numData):
        image = np.zeros((128, 128), dtype=np.uint8)
        x = random.randint(30, 98)
        y = random.randint(30, 98)
        axis_length_x = random.randint(10, 30)
        axis_length_y = random.randint(10, 30)
        angle = random.randint(0, 180)

        cv2.ellipse(image, (x, y), (axis_length_x, axis_length_y), angle, 0, 360, 255, 1)
        imgPath = os.path.join(DEST_IMG_FOLDER, f'Ellipse_{i}.png')
        cv2.imwrite(imgPath, image)

        labelPath = os.path.join(DEST_LABEL_FOLDER, f'Ellipse_{i}.txt')
        with open(labelPath, 'w') as f:
            class_id = 1
            x_center = x / 128.0
            y_center = y / 128.0
            width = (2 * axis_length_x) / 128.0
            height = (2 * axis_length_y) / 128.0

            f.write(f"{class_id} {x_center} {y_center} {width} {height} ")

            num_points = 100
            for theta in np.linspace(0, 2 * np.pi, num_points):
                px = (x + axis_length_x * np.cos(theta) * np.cos(np.radians(angle)) - axis_length_y * np.sin(
                    theta) * np.sin(np.radians(angle))) / 128.0
                py = (y + axis_length_x * np.cos(theta) * np.sin(np.radians(angle)) + axis_length_y * np.sin(
                    theta) * np.cos(np.radians(angle))) / 128.0
                f.write(f"{px} {py} ")
            f.write("\n")
