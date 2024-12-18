import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

# Setup paths
UPLOAD_FOLDER = 'static/uploaded_images'  # Save images in static/uploaded_images
MODEL_DIR = os.path.join(os.getcwd(), 'models')  # Folder to store model and label files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Allowed image file extensions
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Redirect to /recognize when visiting the root URL."""
    return redirect(url_for('recognize'))

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    result = None
    image_path = None
    if request.method == 'POST':
        try:
            # Check for existing model and label files
            model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("improved_face_recognition_model") and f.endswith(".h5")]
            labels_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_labels.npy")]

            if model_files and labels_files:
                # Load the most recent model and labels
                model_files.sort(reverse=True)  # Get the most recent model file
                latest_model_path = os.path.join(MODEL_DIR, model_files[0])

                labels_files.sort(reverse=True)  # Get the most recent labels file
                latest_labels_path = os.path.join(MODEL_DIR, labels_files[0])

                model = tf.keras.models.load_model(latest_model_path)
                label_classes = np.load(latest_labels_path)
            else:
                return "No existing model found. Please train a model first."

            # Handle the uploaded image
            uploaded_file = request.files['image']
            if uploaded_file and allowed_file(uploaded_file.filename):
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)

                # Process the image
                img = cv2.imread(file_path)
                img_resized = cv2.resize(img, (128, 128))  # Resize to match model input
                img_normalized = img_resized / 255.0
                img_input = np.expand_dims(img_normalized, axis=0)

                # Predict using the model
                predictions = model.predict(img_input)
                predicted_class = label_classes[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

                result = f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%"
                image_path = filename  # Just save the filename to be used in the template

        except Exception as e:
            return f"An error occurred during recognition: {e}"

        # Render the result section to send back to the client
        return render_template('result.html', result=result, image_path=image_path)

    return render_template('recognize.html')


if __name__ == '__main__':
    app.run(debug=True)
