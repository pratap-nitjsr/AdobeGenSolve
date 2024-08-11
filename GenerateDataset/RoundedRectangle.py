import cv2
import numpy as np
import random
import os


def GenerateRoundedRectangleDataset(purpose, numData):
    DEST_IMG_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'images')
    DEST_LABEL_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'labels')

    if not os.path.exists(DEST_IMG_FOLDER):
        os.makedirs(DEST_IMG_FOLDER)
    if not os.path.exists(DEST_LABEL_FOLDER):
        os.makedirs(DEST_LABEL_FOLDER)

    for i in range(numData):
        image = np.zeros((128, 128), dtype=np.uint8)

        x1 = random.randint(10, 70)
        y1 = random.randint(10, 70)
        width = random.randint(20, 50)
        height = random.randint(20, 50)
        corner_radius = random.randint(5, min(width, height) // 2)

        top_left = (x1, y1)
        bottom_right = (x1 + width, y1 + height)

        cv2.line(image, (x1 + corner_radius, y1), (x1 + width - corner_radius, y1), 255, 1)
        cv2.line(image, (x1 + corner_radius, y1 + height), (x1 + width - corner_radius, y1 + height), 255, 1)
        cv2.line(image, (x1, y1 + corner_radius), (x1, y1 + height - corner_radius), 255, 1)
        cv2.line(image, (x1 + width, y1 + corner_radius), (x1 + width, y1 + height - corner_radius), 255, 1)

        cv2.ellipse(image, (x1 + corner_radius, y1 + corner_radius), (corner_radius, corner_radius), 180, 0, 90, 255,
                    1)
        cv2.ellipse(image, (x1 + width - corner_radius, y1 + corner_radius), (corner_radius, corner_radius), 270, 0, 90,
                    255, 1)
        cv2.ellipse(image, (x1 + corner_radius, y1 + height - corner_radius), (corner_radius, corner_radius), 90, 0, 90,
                    255, 1)
        cv2.ellipse(image, (x1 + width - corner_radius, y1 + height - corner_radius), (corner_radius, corner_radius), 0,
                    0, 90, 255, 1)

        imgPath = os.path.join(DEST_IMG_FOLDER, f'RoundRectangle_{i}.png')
        cv2.imwrite(imgPath, image)

        labelPath = os.path.join(DEST_LABEL_FOLDER, f'RoundRectangle_{i}.txt')
        with open(labelPath, 'w') as f:
            class_id = 6
            x_center = (x1 + x1 + width) / 2 / 128.0
            y_center = (y1 + y1 + height) / 2 / 128.0
            width_norm = width / 128.0
            height_norm = height / 128.0

            f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm} ")

            f.write(f"{(x1 + corner_radius) / 128.0} {y1 / 128.0} {(x1 + width - corner_radius) / 128.0} {y1 / 128.0} ")
            f.write(
                f"{x1 / 128.0} {(y1 + corner_radius) / 128.0} {x1 / 128.0} {(y1 + height - corner_radius) / 128.0} ")
            f.write(
                f"{(x1 + width) / 128.0} {(y1 + corner_radius) / 128.0} {(x1 + width) / 128.0} {(y1 + height - corner_radius) / 128.0} ")
            f.write(
                f"{(x1 + corner_radius) / 128.0} {(y1 + height) / 128.0} {(x1 + width - corner_radius) / 128.0} {(y1 + height) / 128.0} ")

            for theta in np.linspace(0, np.pi / 2, 25):
                f.write(
                    f"{(x1 + corner_radius - corner_radius * np.cos(theta)) / 128.0} {(y1 + corner_radius - corner_radius * np.sin(theta)) / 128.0} ")
                f.write(
                    f"{(x1 + width - corner_radius + corner_radius * np.cos(theta)) / 128.0} {(y1 + corner_radius - corner_radius * np.sin(theta)) / 128.0} ")
                f.write(
                    f"{(x1 + corner_radius - corner_radius * np.cos(theta)) / 128.0} {(y1 + height - corner_radius + corner_radius * np.sin(theta)) / 128.0} ")
                f.write(
                    f"{(x1 + width - corner_radius + corner_radius * np.cos(theta)) / 128.0} {(y1 + height - corner_radius + corner_radius * np.sin(theta)) / 128.0} ")
            f.write("\n")
