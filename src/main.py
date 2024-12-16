import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.image_augmentation import preprocess_and_augment_with_pillow
from src.model_training import build_improved_model, load_dataset, train_model
from model_evaluation import evaluate_model
from src.model_plotting import plot_training_history  # Import the plotting function

def main():
    # Directories
    input_dataset_dir = "data\My dataset"  # Input directory where original images are stored
    output_dataset_dir = "data\augmented_dataset"  # Output directory where augmented images will be saved
    output_model_file = "improved_face_recognition_model.h5"  # Path to save the model

    # Step 1: Preprocess and Augment Images
    print("Preprocessing and augmenting images...")
    preprocess_and_augment_with_pillow(input_dataset_dir, output_dataset_dir, img_size=(128, 128), augmentations=4)

    # Step 2: Train the Model
    print("Training the model...")
    history = train_model(output_dataset_dir, output_model_file, img_size=(128, 128), learning_rate=0.001, weight_decay=1e-4)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)

    # Step 3: Load the model for evaluation
    print("Loading the trained model...")
    model = tf.keras.models.load_model(output_model_file)
    
    # Load the validation set for evaluation
    X, y, label_encoder = load_dataset(output_dataset_dir, img_size=(128, 128))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    class_names = label_encoder.classes_

    # Step 4: Evaluate the Model
    print("Evaluating the model...")
    evaluate_model(model, label_encoder, X_val, y_val, class_names)

if __name__ == "__main__":
    main()
