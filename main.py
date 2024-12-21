import os
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from google.cloud import storage
from tensorflow.keras.models import load_model
from PIL import Image
import io
from google.auth import credentials

# Setup paths
MODEL_DIR = 'models'                        # Directory for models
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Allowed image file extensions
BUCKET_NAME = 'image_recognition_models'     # Replace with your actual bucket name

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'

# Ensure the upload folder exists (for local use)
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up Google Cloud credentials
credentials_path = "prompt-441717-4ce1dfc5b489.json"  # Update with the path to your JSON key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Google Cloud Storage Client
storage_client = storage.Client()

# Global variables for model and label encoder
model = None
label_encoder = None

# Setup logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_labels(model_dir):
    """
    Load the trained model and label encoder only once during app startup.
    """
    global model, label_encoder
    if model is None or label_encoder is None:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("improved_face_recognition_model") and f.endswith(".h5")]
        labels_files = [f for f in os.listdir(model_dir) if f.endswith("_labels.npy")]

        if not model_files or not labels_files:
            raise FileNotFoundError("Model or labels file not found in the specified directory.")

        model_files.sort(reverse=True)
        labels_files.sort(reverse=True)

        model_path = os.path.join(model_dir, model_files[0])
        labels_path = os.path.join(model_dir, labels_files[0])

        model = load_model(model_path)
        label_encoder = np.load(labels_path, allow_pickle=True)
    
    return model, label_encoder

def upload_to_cloud_storage(file, filename):
    """
    Upload the image to Google Cloud Storage.
    """
    blob = storage_client.bucket(BUCKET_NAME).blob(filename)
    blob.upload_from_file(file)
    blob.make_public()  # Make the file public if needed
    return blob.public_url

def preprocess_image(image_stream, img_size=(160, 160)):
    """
    Preprocess the image: resize, normalize, and reshape.
    """
    # Open the image with Pillow
    img = Image.open(image_stream)
    img = img.resize(img_size)
    
    # Normalize the image
    img = np.array(img) / 255.0
    
    # Expand dimensions to match model's input shape (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, label_encoder, img):
    """
    Predict the label of the given image using the trained model.
    """
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_label = label_encoder[predicted_class_index][0]
    confidence = np.max(predictions) * 100
    return predicted_class_label, confidence

@app.route('/')
def index():
    """Redirect to /recognize when visiting the root URL."""
    return redirect(url_for('recognize'))

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    result = None
    image_url = None
    if request.method == 'POST':
        try:
            # Load model and label encoder (only once on first request)
            model, label_encoder = load_model_and_labels(MODEL_DIR)

            # Handle the uploaded image
            uploaded_file = request.files['image']
            if uploaded_file and allowed_file(uploaded_file.filename):
                filename = secure_filename(uploaded_file.filename)
                app.logger.info(f"Processing image: {filename}")

                # Upload image to Google Cloud Storage
                image_url = upload_to_cloud_storage(uploaded_file, filename)
                app.logger.info(f"Image uploaded to Cloud Storage: {image_url}")

                # Preprocess the image
                img = preprocess_image(uploaded_file.stream, img_size=(160, 160))

                # Predict the label of the image
                predicted_class, confidence = predict_image(model, label_encoder, img)

                result = f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%"
                app.logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}%")

        except Exception as e:
            app.logger.error(f"Error during recognition: {str(e)}")
            return render_template('recognize.html', result=f"An error occurred: {e}")

        return render_template('result.html', result=result, image_url=image_url)

    return render_template('recognize.html')

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=8080, debug=True)
