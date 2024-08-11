import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLO model for shape detection

model_path = r'model/Shapes_segmentation_YOLO.pt'
# model_path = os.getenv('MODEL_PATH', 'model/Shapes_segmentation_YOLO.pt')
model = YOLO(model_path)


# Function to convert base64 string to image
def base64ToImage(b64str):
    image_data = base64.b64decode(b64str)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


# Function to detect shapes in the image using the YOLO model
def detectImage(image):
    results = model(image, conf=0.07, iou=0.6)  # Get the results from the YOLO model
    return results

# Function to create an image with the detected shapes using masks
def makeImage(image, detections):
    # Create a copy of the original image to draw on
    output_image = np.zeros_like(image)

    for detection in detections:
        class_id = detection['class']
        mask = detection['mask']  # Segmentation mask

        # Convert the mask to a boolean mask and apply it on the original image
        mask = np.array(mask, dtype=np.int32).reshape((-1, 2))
        mask = np.clip(mask, 0, min(image.shape[0], image.shape[1]) - 1)
        mask = mask.astype(np.int32)
        mask_img = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw the mask onto the mask image
        cv2.fillPoly(mask_img, [mask], 255)

        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if class_id == 0:  # Circle
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output_image, center, radius, (0, 255, 0), 2)

        elif class_id == 1:  # Ellipse
            for contour in contours:
                if len(contour) >= 5:  # Check if there are enough points to fit an ellipse
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)

        elif class_id == 2:  # Line
            for contour in contours:
                if len(contour) >= 2:
                    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((output_image.shape[1] - x) * vy / vx) + y)
                    cv2.line(output_image, (output_image.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)

        elif class_id == 3:  # Polygon
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.polylines(output_image, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

        elif class_id == 4:  # Rectangle
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        elif class_id == 5:  # Star

            for contour in contours:

                # Get the minimum enclosing circle of the contour

                (x, y), radius = cv2.minEnclosingCircle(contour)

                center = (int(x), int(y))

                radius = int(radius)

                # Generate points for a star

                num_points = 5

                outer_radius = radius

                inner_radius = radius * 0.5

                angle_offset = np.pi / num_points

                star_points = []

                for i in range(num_points * 2):

                    angle = i * np.pi / num_points + angle_offset

                    if i % 2 == 0:

                        r = outer_radius

                    else:

                        r = inner_radius

                    point = (int(center[0] + r * np.cos(angle)), int(center[1] + r * np.sin(angle)))

                    star_points.append(point)

                # Convert the star points to a contour-like array

                star_points = np.array(star_points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw the star

                cv2.polylines(output_image, [star_points], isClosed=True, color=(0, 255, 0), thickness=2)


        elif class_id == 6:  # Rounded Rectangle
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                corner_radius = min(w, h) // 5  # Approximate the corner radius
                cv2.rectangle(output_image, (x + corner_radius, y), (x + w - corner_radius, y + h), (0, 255, 0), 2)
                cv2.rectangle(output_image, (x, y + corner_radius), (x + w, y + h - corner_radius), (0, 255, 0), 2)
                cv2.circle(output_image, (x + corner_radius, y + corner_radius), corner_radius, (0, 255, 0), 2)
                cv2.circle(output_image, (x + w - corner_radius, y + corner_radius), corner_radius, (0, 255, 0), 2)
                cv2.circle(output_image, (x + corner_radius, y + h - corner_radius), corner_radius, (0, 255, 0), 2)
                cv2.circle(output_image, (x + w - corner_radius, y + h - corner_radius), corner_radius, (0, 255, 0), 2)

    return output_image




# Function to convert image to base64 string
def ImageTobase64(image):
    _, buffer = cv2.imencode('.png', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    return image_b64


# Flask app initialization
app = Flask(__name__)
CORS(app)


# Route for generating the shapes on the image
@app.route('/generate', methods=['POST'])
def generate():
    print("Check1")
    data = request.json
    image_b64 = data.get('image', '')

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    # Convert the base64 string to an image
    image = base64ToImage(image_b64)

    # Detect shapes in the image
    results = detectImage(image)

    detections = []
    for result in results:
        masks = result.masks.xy  # Extract masks
        classes = result.boxes.cls  # Corresponding class IDs
        for mask, class_id in zip(masks, classes):
            detections.append({'class': int(class_id), 'mask': mask})

    # Draw the shapes on the image using the masks
    result_image = makeImage(image, detections)

    # cv2.imshow("a",result_image)
    # cv2.waitKey(0)

    # Convert the result image back to base64
    result_image_b64 = ImageTobase64(result_image)

    return jsonify({"image": result_image_b64})


# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
