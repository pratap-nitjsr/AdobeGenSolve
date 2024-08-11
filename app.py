import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Flask server URL
server_url = "http://localhost:5000/generate"

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def base64_to_image(b64str: str) -> Image.Image:
    image_data = base64.b64decode(b64str)
    image = Image.open(BytesIO(image_data))
    return image

def main():
    st.title("Shape Detection with YOLO")
    st.write("Upload an image, and the server will detect shapes and return the processed image.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        # Convert image to base64
        img_base64 = image_to_base64(original_image)

        # Send image to the Flask server
        with st.spinner('Processing...'):
            response = requests.post(server_url, json={"image": img_base64})

        if response.status_code == 200:
            # Convert the base64 string back to an image
            result_image_b64 = response.json().get("image")
            result_image = base64_to_image(result_image_b64)

            # Display the result image
            st.image(result_image, caption="Processed Image", use_column_width=True)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()