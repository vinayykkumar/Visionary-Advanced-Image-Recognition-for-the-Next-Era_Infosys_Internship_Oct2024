import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configure the page layout to wide
st.set_page_config(layout="wide", page_title="Advanced Image Recognition")

# Load your trained model
model = load_model("finetuned_model.keras")  # Replace with your model's filename
class_labels = ['Akshay', 'Amitabh', 'Prabhas', 'Vijay']  # Update with your classes

# Define class images with proper paths
class_images = {
    'Akshay': 'Data/My dataset/Akshay Kumar/Image_15.jpg',
    'Amitabh': 'Data/My dataset/Amitabh Bachchan/2.jpg',
    'Prabhas': 'Data/My dataset/Prabhas/4.jpg',
    'Vijay': 'Data/My dataset/vijay/0c7b1bfad7156c419e9aefb385995fa9.jpg',
}


# Function to preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (300, 300))  # Resize to a standard size to improve quality
    equalized_image = cv2.equalizeHist(resized_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    normalized_image = blurred_image / 255.0
    mean, std = cv2.meanStdDev(normalized_image)
    standardized_image = (normalized_image - mean[0][0]) / (std[0][0] + 1e-8)
    final_image = cv2.resize(standardized_image, target_size)  # Resize to model's expected input size
    final_image = np.expand_dims(final_image, axis=-1)
    final_image = np.repeat(final_image, 3, axis=-1)  # Convert back to 3 channels (RGB)
    return np.expand_dims(final_image, axis=0)

# Add custom styles
st.markdown(
    """
    <style>
        .header {
            background-color: #2E86C1;
            padding: 10px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            border-radius: 10px;
        }
        .class-images {
            display: flex;
            justify-content: space-evenly;
            padding: 10px;
            gap: 20px;
        }
        .class-images img {
            width: 250px;
            height: 250px;
            object-fit: cover;
            border-radius: 10px;
        }
        .output-section {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }
        .output-image img {
            width: 250px;
            height: 250px;
            object-fit: cover;
        }
        .results-box {
            background-color: #FFFFFF;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="header">ADVANCED IMAGE RECOGNITION FOR NEXT ERA</div>', unsafe_allow_html=True)

# Split the screen into two halves for uploading image and displaying result
left_column, right_column = st.columns([1, 1])

# Left column: Upload image, "Predict" button, and results
with left_column:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    predict_button = st.button("Predict")

    if uploaded_file is not None:
        st.write("Image upload is successful. Please predict.")

    if predict_button and uploaded_file is not None:
        # Convert the uploaded file to a byte array and decode only once
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Corrected decoding

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Predict the class
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        prediction_prob = prediction[0][predicted_class] * 100

        # Display results and the image in the right column
        with right_column:
            st.markdown('<div class="output-section">', unsafe_allow_html=True)
            # Display the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.markdown('<div class="output-image">', unsafe_allow_html=True)
            st.image(image_rgb, caption="Uploaded Image (Prediction)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display the results
            st.markdown('<div class="results-box">', unsafe_allow_html=True)
            st.subheader("Results")
            st.write(f"**Predicted Class:** {class_labels[predicted_class]}")
            st.write(f"**Confidence:** {prediction_prob:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Class References Section
st.subheader("Class References")
class_images_container = st.container()
with class_images_container:
    cols = st.columns(4)
    for idx, (class_name, image_path) in enumerate(class_images.items()):
        with cols[idx]:
            class_image = cv2.imread(image_path)
            if class_image is not None:
                class_image_rgb = cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB)
                resized_class_image = cv2.resize(class_image_rgb, (250, 250))  # Ensure consistent size
                st.image(resized_class_image, caption=class_name, use_container_width=True)  # Display fixed size class images
