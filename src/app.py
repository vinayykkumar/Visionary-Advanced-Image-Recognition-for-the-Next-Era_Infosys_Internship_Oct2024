import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 
from PIL import Image

st.header('Image Classification Model')
model = load_model('/Users/AakritiGupta/Downloads/actor_model_finetuned.keras')
# Define class labels
class_labels = ['Akshay Kumar', 'Prabhas', 'Amitabh Bachchan', 'Vijay']

img_height = 180
img_width = 180

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check if the image was loaded correctly
    if img is None:
        st.error("Could not read the image. Please upload again.")
    else:
        # Resize and normalize the image
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = np.expand_dims(img_resized / 255.0, axis=0)

        # Make a prediction
        predictions = model.predict(img_normalized)
        predicted_label = class_labels[np.argmax(predictions)]
        predicted_score = np.max(predictions)

        # Display the image and prediction
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write(f"Prediction: {predicted_label} with confidence {predicted_score * 100:.2f}%")