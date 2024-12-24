import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# Paths
data_dir = r'C:\Users\Shyam\OneDrive\Desktop\project\Visionary\My DataSet\My dataset'  # Dataset path
output_dir = r'C:\Users\Shyam\OneDrive\Desktop\project\Visionary\Processed Dataset'  # Output path for preprocessed images

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 1. Visualize the Dataset
print("Loading and visualizing dataset...")

# Load the dataset and organize images by class (subfolder names as class labels)
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# Convert the dataset into a NumPy iterator
data_iterator = data.as_numpy_iterator()

# Fetch a batch of images and labels
batch = data_iterator.next()

# Visualize the first 4 images in the batch
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):  # Show first 4 images
    ax[idx].imshow(img.astype(int))  # Convert to int for proper display
    ax[idx].title.set_text(batch[1][idx])  # Display class label as title
plt.show()

# 2. Preprocessing Function
def preprocess_image(image, size=(224, 224)):
    """Resize, normalize, and apply preprocessing to an image."""
    image = cv2.resize(image, size)  # Resize the image
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = image.astype(np.uint8)  # Ensure the image is 8-bit unsigned before applying CLAHE
    image = clahe.apply(image)  # Apply CLAHE for contrast enhancement
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(image, axis=-1)  # Add channel dimension

# 3. Apply Preprocessing to the Dataset and Save Images
def preprocess_and_save_dataset(dataset, size=(224, 224)):
    """Preprocess an entire dataset and save preprocessed images."""
    preprocessed_images = []
    labels = []
    
    for images, lbls in dataset:
        for idx, (img, lbl) in enumerate(zip(images, lbls)):
            # Preprocess each image
            preprocessed_img = preprocess_image(img.numpy(), size)
            
            # Save preprocessed image
            class_dir = os.path.join(output_dir, str(lbl.numpy()))
            os.makedirs(class_dir, exist_ok=True)
            file_path = os.path.join(class_dir, f"image_{idx}.png")
            cv2.imwrite(file_path, (preprocessed_img.squeeze() * 255).astype(np.uint8))  # Save as grayscale
            
            # Append to list
            preprocessed_images.append(preprocessed_img)
            labels.append(lbl.numpy())
    
    return np.array(preprocessed_images), np.array(labels)

# Preprocess and save dataset
print("Preprocessing and saving dataset...")
train_images, train_labels = preprocess_and_save_dataset(data)

# Display dataset shape after preprocessing
print(f"Training data shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

# Optional: Visualize a few preprocessed images
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(train_images[:4]):  # Show first 4 preprocessed images
    ax[idx].imshow(img.squeeze(), cmap='gray')  # Show as grayscale
    ax[idx].title.set_text(train_labels[idx])  # Display class label as title
plt.show()
