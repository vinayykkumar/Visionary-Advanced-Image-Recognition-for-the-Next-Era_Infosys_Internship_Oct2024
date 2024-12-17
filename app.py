import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 
from PIL import Image

st.header('Image Classification Model')
model = load_model('D:/Data Science/Infosys Project/Image_classify1.keras')
data_cat = ['Amitabh Bachchan', 'Narendra Modi', 'Rohit Sharma', 'Virat Kohli']
img_height = 180
img_width = 180

# Use file uploader for image selection.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image_load = Image.open(uploaded_file)
    image_load = image_load.resize((img_width, img_height))  # Resize image to expected size

    img_arr = tf.keras.preprocessing.image.img_to_array(image_load)  # Convert image to array
    img_bat = np.expand_dims(img_arr, axis=0)  # Prepare batch

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display the image
    st.image(image_load, caption='Uploaded Image', width=200)

    # Determine the predicted class
    predicted_index = np.argmax(score)
    predicted_score = np.max(score)

    # Set a threshold for classification
    threshold = 0.5
    
    if predicted_score < threshold:
        st.markdown("<span style='color: red; font-size: 20px; font-weight: bold;'>Unknown Person! Please upload another image.</span>", unsafe_allow_html=True)
    else:
        # Display the predicted name in bold and red color
        st.markdown(f"The Person is:<span style='color: red; font-size: 20px; font-weight: bold;'> {data_cat[predicted_index]}</span>", unsafe_allow_html=True)
        # Display the accuracy of model
        st.write(f'With accuracy of {predicted_score * 100:.2f}%')