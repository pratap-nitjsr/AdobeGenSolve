import cv2
import numpy as np
import random
import os


def GeneratePolygonDataset(purpose, numData):
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
        sides = random.randint(3, 10)

        polygon_points = []
        for j in range(sides):
            angle = 2 * np.pi * j / sides
            x = int(x_center + radius * np.cos(angle))
            y = int(y_center + radius * np.sin(angle))
            polygon_points.append((x, y))

        polygon_points = np.array(polygon_points, np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.polylines(image, [polygon_points], isClosed=True, color=255, thickness=1)

        imgPath = os.path.join(DEST_IMG_FOLDER, f'Polygon_{i}.png')
        cv2.imwrite(imgPath, image)

        labelPath = os.path.join(DEST_LABEL_FOLDER, f'Polygon_{i}.txt')
        with open(labelPath, 'w') as f:
            class_id = 3

            min_x = min(polygon_points[:, 0, 0])
            max_x = max(polygon_points[:, 0, 0])
            min_y = min(polygon_points[:, 0, 1])
            max_y = max(polygon_points[:, 0, 1])

            x_center_norm = (min_x + max_x) / 2 / 128.0
            y_center_norm = (min_y + max_y) / 2 / 128.0
            width_norm = (max_x - min_x) / 128.0
            height_norm = (max_y - min_y) / 128.0

            f.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm} ")

            for point in polygon_points:
                f.write(f"{point[0][0] / 128.0} {point[0][1] / 128.0} ")
            f.write("\n")
