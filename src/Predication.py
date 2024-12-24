import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load Model
model_path = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\models\image_model.keras"
model = load_model(model_path)

# Class labels (defined in the same order as during training)
CLASS_LABELS = ['Akshay Kumar', 'Amitabh Bachchan', 'Prabhas', 'Vijay']

# Image Preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Predict
image_path = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\My DataSet\My dataset\Prabhas\50.jpg"
image = preprocess_image(image_path)
predictions = model.predict(image)
predicted_class_index = np.argmax(predictions)

# Map the predicted index to class name
predicted_class_name = CLASS_LABELS[predicted_class_index]

print(f"Predicted Class: {predicted_class_name}")
