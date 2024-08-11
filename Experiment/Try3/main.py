import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# import svgwrite
# import cairosvg

# Function to read CSV files
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

# Function to plot polylines
def plot(path_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(path_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()


# def polylines2svg(paths_XYs, svg_path):
#     W, H = 0, 0
#     for path_XYs in paths_XYs:
#         for XY in path_XYs:
#             W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
#     padding = 0.1
#     W, H = int(W + padding * W), int(H + padding * H)
#
#     dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
#     group = dwg.g()
#     colors = ['#ff0000', '#00ff00', '#0000ff', '#00ffff', '#ff00ff', '#ffff00', '#000000']
#     for i, path in enumerate(paths_XYs):
#         path_data = []
#         c = colors[i % len(colors)]
#         for XY in path:
#             path_data.append(("M", (XY[0, 0], XY[0, 1])))
#             for j in range(1, len(XY)):
#                 path_data.append(("L", (XY[j, 0], XY[j, 1])))
#             if not np.allclose(XY[0], XY[-1]):
#                 path_data.append(("Z", None))
#         group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
#     dwg.add(group)
#     dwg.save()
#     png_path = svg_path.replace('.svg', '.png')
#     fact = max(1, 1024 // min(H, W))
#     cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')
#     return png_path

# Example usage
# polylines2svg(path_XYs, 'output.svg')


import numpy as np

# Function to regularize curves
def regularize_curves(path_XYs):
    regularized_paths = []
    for XYs in path_XYs:
        for XY in XYs:
            # Identify straight lines
            if is_straight_line(XY):
                regularized_paths.append([XY])
            # Identify circles
            elif is_circle(XY):
                regularized_paths.append([fit_circle(XY)])
            # Identify ellipses
            elif is_ellipse(XY):
                regularized_paths.append([fit_ellipse(XY)])
            # Identify rectangles
            elif is_rectangle(XY):
                regularized_paths.append([fit_rectangle(XY)])
            # Identify regular polygons
            elif is_regular_polygon(XY):
                regularized_paths.append([fit_regular_polygon(XY)])
            else:
                regularized_paths.append([XY])
    return regularized_paths

# Helper functions to identify and fit shapes
def is_straight_line(XY):
    model = LinearRegression()
    model.fit(XY[:, 0].reshape(-1, 1), XY[:, 1])
    r2 = r2_score(XY[:, 1], model.predict(XY[:, 0].reshape(-1, 1)))
    return r2 > 0.99

def is_circle(XY):
    # Calculate the centroid of the points
    center = np.mean(XY, axis=0)
    # Calculate the distance from each point to the centroid
    distances = np.linalg.norm(XY - center, axis=1)
    # Calculate the standard deviation of these distances
    std_dev = np.std(distances)
    # If the standard deviation is small, the points form a circle
    return std_dev < 1e-2

def fit_circle(XY):
    # Calculate the centroid of the points
    center = np.mean(XY, axis=0)
    # Calculate the radius as the mean distance from the center
    radius = np.mean(np.linalg.norm(XY - center, axis=1))
    # Create points for the fitted circle
    angles = np.linspace(0, 2 * np.pi, 100)
    circle_XY = np.array([center + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
    return circle_XY

def is_ellipse(XY):
    # Placeholder implementation, you can use more sophisticated methods
    return False

def fit_ellipse(XY):
    # Placeholder implementation
    return XY

def is_rectangle(XY):
    # Check for right angles and equal opposite sides
    if len(XY) != 4:
        return False
    d1 = np.linalg.norm(XY[0] - XY[1])
    d2 = np.linalg.norm(XY[1] - XY[2])
    d3 = np.linalg.norm(XY[2] - XY[3])
    d4 = np.linalg.norm(XY[3] - XY[0])
    diag1 = np.linalg.norm(XY[0] - XY[2])
    diag2 = np.linalg.norm(XY[1] - XY[3])
    return np.isclose(d1, d3) and np.isclose(d2, d4) and np.isclose(diag1, diag2)

def fit_rectangle(XY):
    # Directly return the points if they form a rectangle
    return XY

def is_regular_polygon(XY):
    # Check for equal side lengths and equal angles
    num_points = len(XY)
    if num_points < 3:
        return False
    side_lengths = [np.linalg.norm(XY[i] - XY[(i + 1) % num_points]) for i in range(num_points)]
    angles = []
    for i in range(num_points):
        v1 = XY[i] - XY[i - 1]
        v2 = XY[(i + 1) % num_points] - XY[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    return np.allclose(side_lengths, side_lengths[0]) and np.allclose(angles, angles[0])

def fit_regular_polygon(XY):
    # Calculate centroid
    center = np.mean(XY, axis=0)
    num_points = len(XY)
    radius = np.mean([np.linalg.norm(point - center) for point in XY])
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    polygon_XY = np.array([center + radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
    return polygon_XY

# Example usage
# path_XYs = read_csv('examples/isolated.csv')
# regularized_paths = regularize_curves(path_XYs)
# plot(regularized_paths)

import numpy as np
from scipy.spatial import distance

def detect_symmetry(path_XYs):
    symmetric_paths = []
    for XYs in path_XYs:
        for XY in XYs:
            if has_reflection_symmetry(XY):
                symmetric_paths.append([XY])
            elif has_rotational_symmetry(XY):
                symmetric_paths.append([XY])
            else:
                symmetric_paths.append([XY])
    return symmetric_paths

def has_reflection_symmetry(XY):
    center = np.mean(XY, axis=0)
    reflected_XY = XY.copy()
    for axis in [0, 1]:
        reflected_XY[:, axis] = 2 * center[axis] - reflected_XY[:, axis]
        if np.allclose(np.sort(XY, axis=0), np.sort(reflected_XY, axis=0)):
            return True
    return False

def has_rotational_symmetry(XY):
    center = np.mean(XY, axis=0)
    num_points = len(XY)
    for k in range(2, 9):  # Check for 2-fold to 8-fold symmetry
        rotated_XY = XY.copy()
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        for angle in angles:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
            rotated_points = np.dot(XY - center, rotation_matrix) + center
            if np.allclose(np.sort(XY, axis=0), np.sort(rotated_points, axis=0)):
                return True
    return False

# Example usage
# symmetric_paths = detect_symmetry(path_XYs)
# plot(symmetric_paths)

def complete_curves(path_XYs):
    completed_paths = []
    for XYs in path_XYs:
        for XY in XYs:
            if is_disconnected_occlusion(XY):
                completed_paths.append(complete_disconnected_occlusion(XY))
            elif is_connected_occlusion(XY):
                completed_paths.append(complete_connected_occlusion(XY))
            else:
                completed_paths.append([XY])
    return completed_paths

def is_disconnected_occlusion(XY):
    # Check if there are multiple disjoint segments in the curve
    # A simple heuristic can be checking large gaps in consecutive points
    distances = np.linalg.norm(np.diff(XY, axis=0), axis=1)
    return np.any(distances > np.mean(distances) * 2)

def complete_disconnected_occlusion(XY):
    # Placeholder for completing disconnected occlusions
    # This can be implemented using interpolation or fitting techniques
    return XY

def is_connected_occlusion(XY):
    # Check if there are segments partially occluded but still connected
    # A simple heuristic can be checking for sharp angles
    angles = []
    for i in range(1, len(XY) - 1):
        v1 = XY[i] - XY[i - 1]
        v2 = XY[i + 1] - XY[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    return np.any(np.array(angles) < np.pi / 4)

def complete_connected_occlusion(XY):
    # Placeholder for completing connected occlusions
    # This can be implemented using curve fitting techniques
    return XY

# Example usage
# completed_paths = complete_curves(symmetric_paths)
# plot(completed_paths)


# Main function to process the input CSV and generate the output
def process_curves(input_csv, output_svg):
    path_XYs = read_csv(input_csv)
    plot(path_XYs)

    regularized_paths = regularize_curves(path_XYs)
    plot(regularized_paths)

    symmetric_paths = detect_symmetry(regularized_paths)
    plot(symmetric_paths)

    completed_paths = complete_curves(symmetric_paths)
    plot([completed_paths])

    # polylines2svg(completed_paths, output_svg)

# Example usage
# process_curves('examples/isolated.csv', 'output.svg')

process_curves('..\..\problems\occlusion1.csv', '2')