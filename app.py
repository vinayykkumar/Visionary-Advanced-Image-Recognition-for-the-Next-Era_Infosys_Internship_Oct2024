import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the saved model
model = tf.keras.models.load_model("tl_model.h5")

class_names = [
    'shikhar_dhawan', 'shimron_hetmyer', 'shoaib_malik', 'steve_smith', 'tabraiz_shamsi','thisara_perera', 'tom_curran','trent_boult','vijay_shankar', 'virat_kohli'
]


def preprocess_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, size)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = image / 255.0
        return np.expand_dims(image, axis=0)  # Add batch dimension
    else:
        print(f"Could not read the image from {image_path}")
        return None
    
# Streamlit UI
st.title("Visionary: Advanced Image Recognition for the Next Era")
st.write("Upload an image of a cricketer, and the model will predict who it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        temp_path = "temp_uploaded_image.jpg"  # Temporary path
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        image = Image.open(temp_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Processing...")

        # Preprocess the image using the file path
        preprocessed_image = preprocess_image(temp_path) 

        # Make predictions
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Show the result
        st.write(f"**Prediction:** {class_names[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")