import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def plot(paths_XYs):
    colours = ['black']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()

def identifyAndRegularize(csvPath):
    contours = read_csv(csvPath)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # Create a white image

    shapes = []

    for cont in contours:
        for sub_cont in cont:
            contour = np.array(sub_cont, dtype=np.float32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            approx = np.array(approx, dtype=np.int32)
            print(approx)
            # cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if len(approx) == 3:
                shapes.append("Triangle")
                cv2.putText(img, "Triangle", (x, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                print(aspectRatio)
                if 0.95 <= aspectRatio < 1.05:
                    shapes.append("Square")
                    cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    img = cv2.polylines(img, [approx], isClosed=True, color=(255, 255, 0), thickness=2)
                else:
                    shapes.append("Rectangle")
                    cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    img = cv2.polylines(img, [approx], isClosed=True, color=(255, 255, 0), thickness=2)
            elif len(approx) == 5:
                shapes.append("Pentagon")
                cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                img = cv2.polylines(img, [approx], isClosed=True, color=(255, 255, 0), thickness=2)
            elif len(approx) == 10:
                shapes.append("Star")
                cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                img = cv2.polylines(img, [approx], isClosed=True, color=(255, 255, 0), thickness=2)
            else:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, orientation) = ellipse
                    major_axis_length = max(axes)
                    minor_axis_length = min(axes)
                    aspect_ratio = minor_axis_length / major_axis_length
                    if aspect_ratio > 0.9:
                        shapes.append("Circle")
                        cv2.putText(img, "Circle", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 0, 0))
                        cv2.circle(img, (int(center[0]), int(center[1])), int(major_axis_length / 2), (255, 255, 0), 2)
                    else:
                        shapes.append("Ellipse")
                        cv2.putText(img, "Ellipse", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (0, 0, 0))
                        cv2.ellipse(img, ellipse, (255, 255, 0), 2)
                else:
                    shapes.append("Irregular")
                    cv2.putText(img, "Irregular", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    print(shapes)
    cv2.imshow("Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

csvPath = r'problems/frag1.csv'
identifyAndRegularize(csvPath)
