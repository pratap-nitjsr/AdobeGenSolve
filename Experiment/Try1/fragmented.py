import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def hierarchical_clustering(XY, distance_threshold=50):
    dist_matrix = pdist(XY)
    linkage_matrix = linkage(dist_matrix, method='single')
    labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    unique_labels = np.unique(labels)
    clusters = [XY[labels == label] for label in unique_labels]
    return clusters

def identify_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5

    if len(approx) == 3:
        return "Triangle", approx, (x, y)
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if 0.95 <= aspectRatio < 1.05:
            return "Square", approx, (x, y)
        else:
            return "Rectangle", approx, (x, y)
    elif len(approx) == 5:
        return "Pentagon", approx, (x, y)
    elif len(approx) == 10:
        return "Star", approx, (x, y)
    else:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes)
            aspect_ratio = minor_axis_length / major_axis_length
            if aspect_ratio > 0.9:
                return "Circle", ellipse, (int(center[0]), int(center[1]))
            else:
                return "Ellipse", ellipse, (int(center[0]), int(center[1]))
        else:
            return "Irregular", approx, (x, y)

def identifyAndRegularize(csvPath):
    contours = read_csv(csvPath)
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255  # Create a larger white image

    shapes = []

    all_points = np.vstack([np.vstack(cont) for cont in contours])  # Combine all points from fragments
    clusters = hierarchical_clustering(all_points)  # Cluster points

    for cluster in clusters:
        cluster = np.array(cluster, dtype=np.int32)
        hull = cv2.convexHull(cluster)
        mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Edge detection
        edges = cv2.Canny(mask, 50, 150)
        detected_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in detected_contours:
            shape, shape_contour, text_pos = identify_shape(contour)

            if shape != "Irregular":
                shapes.append(shape)
                if shape in ["Circle", "Ellipse"]:
                    if shape == "Circle":
                        center = (int(shape_contour[0][0]), int(shape_contour[0][1]))
                        radius = int(shape_contour[1][0] / 2)
                        cv2.circle(img, center, radius, (0, 255, 0), 2)
                    else:
                        cv2.ellipse(img, shape_contour, (0, 255, 0), 2)
                else:
                    cv2.drawContours(img, [shape_contour], 0, (0, 255, 0), 2)
                cv2.putText(img, shape, text_pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            else:
                shapes.append("Irregular")
                cv2.drawContours(img, [shape_contour], 0, (0, 0, 255), 2)
                cv2.putText(img, "Irregular", text_pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    print(shapes)
    cv2.imshow("Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

csvPath = r'problems/frag2.csv'
identifyAndRegularize(csvPath)
