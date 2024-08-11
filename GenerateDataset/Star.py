import cv2
import numpy as np
import random
import os


def GenerateStarDataset(purpose, numData):
    DEST_IMG_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'images')
    DEST_LABEL_FOLDER = os.path.join(f'../CustomDataset/{purpose}', 'labels')

    if not os.path.exists(DEST_IMG_FOLDER):
        os.makedirs(DEST_IMG_FOLDER)
    if not os.path.exists(DEST_LABEL_FOLDER):
        os.makedirs(DEST_LABEL_FOLDER)

    for i in range(numData):
        image = np.zeros((128, 128), dtype=np.uint8)

        x_center = random.randint(30, 98)
        y_center = random.randint(30, 98)
        radius = random.randint(10, 30)

        points = 5
        star_points = []
        for j in range(points * 2):
            angle = j * np.pi / points
            if j % 2 == 0:
                x = int(x_center + radius * np.cos(angle))
                y = int(y_center + radius * np.sin(angle))
            else:
                x = int(x_center + (radius / 2) * np.cos(angle))
                y = int(y_center + (radius / 2) * np.sin(angle))
            star_points.append((x, y))

        for j in range(len(star_points)):
            start_point = star_points[j]
            end_point = star_points[(j + 1) % len(star_points)]
            cv2.line(image, start_point, end_point, 255, 1)

        imgPath = os.path.join(DEST_IMG_FOLDER, f'Star_{i}.png')
        cv2.imwrite(imgPath, image)

        labelPath = os.path.join(DEST_LABEL_FOLDER, f'Star_{i}.txt')
        with open(labelPath, 'w') as f:
            class_id = 5
            star_points = np.array(star_points)
            x_center_norm = x_center / 128.0
            y_center_norm = y_center / 128.0
            width_norm = (max(star_points[:, 0]) - min(star_points[:, 0])) / 128.0
            height_norm = (max(star_points[:, 1]) - min(star_points[:, 1])) / 128.0

            f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm} ")

            for point in star_points:
                f.write(f"{point[0] / 128.0} {point[1] / 128.0} ")
            f.write("\n")