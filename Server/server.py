import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLO model for shape detection
model_path = os.getenv('MODEL_PATH', 'model/Shapes_segmentation_YOLO.pt')
model = YOLO(model_path)


# Function to convert base64 string to image
def base64ToImage(b64str):
    image_data = base64.b64decode(b64str)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


# Function to detect shapes in the image using the YOLO model
def detectImage(image):
    results = model(image)  # Get the results from the YOLO model
    return results


# Function to create an image with the detected shapes using masks
def makeImage(image, detections):
    for detection in detections:
        class_id = detection['class']
        mask = detection['mask']  # Segmentation mask

        # Convert the mask to a boolean mask and apply it on the original image
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if class_id == 0:  # Circle
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(image, center, radius, (0, 255, 0), 2)

        elif class_id == 1:  # Ellipse
            for contour in contours:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)

        elif class_id == 2:  # Line
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        elif class_id == 3:  # Polygon
            for contour in contours:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        elif class_id == 4:  # Rectangle
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        elif class_id == 5:  # Star
            for contour in contours:
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        elif class_id == 6:  # Rounded Rectangle
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Rounded effect can be added, but is approximated here

    return image


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

    # Convert the result image back to base64
    result_image_b64 = ImageTobase64(result_image)

    return jsonify({"image": result_image_b64})


# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
