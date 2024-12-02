import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load dataset
def load_dataset(dataset_dir, img_size=(128, 128)):
    """
    Load images and labels from the specified dataset directory.

    Args:
        dataset_dir (str): Path to dataset directory.
        img_size (tuple): Target size for resizing images.

    Returns:
        tuple: Tuple containing image data (X), labels (y), and LabelEncoder instance.
    """
    X, y = [], []
    label_names = []

    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_names.append(person_name)
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(person_name)

    X = np.array(X, dtype='float32') / 255.0  # Normalize images
    y = np.array(y)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

# Function to build CNN model (Updated)
def build_model(input_shape, num_classes, learning_rate=0.001):
    """
    Build and compile a CNN model.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model: Compiled CNN model.
    """
    # Updated CNN model architecture
    cnn_model = Sequential([
        Conv2D(filters=36, kernel_size=7, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=54, kernel_size=5, activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(2024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Number of outputs is dynamic based on num_classes
    ])

    optimizer = Adam(learning_rate=learning_rate)
    cnn_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return cnn_model

# Function to plot training history
def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.

    Args:
        history: Training history object from Keras model.
    """
    # Accuracy Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'], label='Training Accuracy', color='blue')
    sns.lineplot(x=range(1, len(history.history['val_accuracy']) + 1), y=history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'], label='Training Loss', color='blue')
    sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main function to train the model
def train_model(dataset_dir, output_model_path, img_size=(128, 128), learning_rate=0.001):
    """
    Train and save a face recognition model.

    Args:
        dataset_dir (str): Path to dataset directory.
        output_model_path (str): Path to save the trained model.
        img_size (tuple): Target size for resizing images.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Load dataset
    X, y, label_encoder = load_dataset(dataset_dir, img_size)
    num_classes = len(np.unique(y))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes, learning_rate=learning_rate)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=38)

     # Extract final accuracy values
    final_train_accuracy = history.history['accuracy'][-1] * 100  # Convert to percentage
    final_val_accuracy = history.history['val_accuracy'][-1] * 100  # Convert to percentage

    # Print training and validation accuracy as percentages
    print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")

    # Save the model and label encoder
    model.save(output_model_path)
    np.save(output_model_path + "_labels.npy", label_encoder.classes_)
    print(f"Model saved at {output_model_path}")

    # Plot training history
    plot_training_history(history)

# Example Usage
dataset_directory = "augmented_dataset"  # Path to dataset
output_model_file = "face_recognition_model.h5"  # Path to save model

train_model(dataset_directory, output_model_file)
