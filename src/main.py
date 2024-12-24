import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configure the page layout to wide
st.set_page_config(layout="wide", page_title="Advanced Image Recognition")

# Load your trained model
@st.cache_resource
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_path = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\models\image_model.keras"
model = load_trained_model(model_path)


class_labels = ['Akshay Kumar', 'Amitabh Bachchan', 'Prabhas', 'Vijay']
# Confidence threshold
CONFIDENCE_THRESHOLD = 30.0

# Function to preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, target_size)
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Add custom styles
st.markdown(
    """
    <style>
        body {
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            background: linear-gradient(to right, #4CAF50, #45a049);
            color: white;
            padding: 15px;
            font-size: 32px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .card:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .upload-card {
            text-align: center;
            border: 2px dashed #4CAF50;
            padding: 20px;
            cursor: pointer;
        }
        .upload-card:hover {
            background-color: #e8f5e9;
        }
        .predict-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 18px;
            width: 100%;
            cursor: pointer;
            border: none;
        }
        .predict-button:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="header">Advanced Image Recognition for the Next Era</div>', unsafe_allow_html=True)

# Main Layout
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Upload and Predict")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process and predict
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is not None:
            prediction = model.predict(preprocessed_image)
            predicted_class_idx = np.argmax(prediction)
            prediction_prob = prediction[0][predicted_class_idx] * 100

            # Display prediction results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Results")
            if prediction_prob >= CONFIDENCE_THRESHOLD:
                st.success(f"Predicted Class: {class_labels[predicted_class_idx]}")
                st.info(f"Confidence: {prediction_prob:.2f}%")
            else:
                st.warning("The image does not match any known class.")
                st.error(f"Confidence: {prediction_prob:.2f}% (Below Threshold)")
            st.markdown('</div>', unsafe_allow_html=True)
