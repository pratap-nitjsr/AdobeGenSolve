# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir ultralytics flask flask-cors opencv-python-headless

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MODEL_PATH="/app/model/Shapes_segmentation_YOLO.pt"

# Run server.py when the container launches
CMD ["python", "server.py"]
