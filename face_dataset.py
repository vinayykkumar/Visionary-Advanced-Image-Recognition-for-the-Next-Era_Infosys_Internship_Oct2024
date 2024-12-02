import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_and_augment_with_keras(input_dir, output_dir, img_size=(128, 128), augmentations=2):
    """
    Preprocesses and augments images using OpenCV and Keras ImageDataGenerator.
    
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the output directory for preprocessed and augmented images.
        img_size (tuple): Target size for resizing the images (width, height).
        augmentations (int): Number of augmented images to generate per input image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Iterate through each person's folder in the dataset
    for person_name in os.listdir(input_dir):
        person_folder = os.path.join(input_dir, person_name)
        output_person_folder = os.path.join(output_dir, person_name)

        if not os.path.isdir(person_folder):
            continue  # Skip files, only process directories

        # Create corresponding output directory for each person if it doesn't exist
        if not os.path.exists(output_person_folder):
            os.makedirs(output_person_folder)

        # Process each image in the person's folder
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            
            # Read the image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Step 1: Resize the image
            image_resized = cv2.resize(image, img_size)

            # Step 2: Apply Bilateral Filter for noise reduction while preserving edges
            image_bilateral = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)

            # Step 3: Convert to grayscale for thresholding
            gray = cv2.cvtColor(image_bilateral, cv2.COLOR_BGR2GRAY)

            # Step 4: Apply simple thresholding for background removal
            _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Apply the threshold mask to the original image
            foreground = cv2.bitwise_and(image_bilateral, image_bilateral, mask=thresholded)

            # Save the preprocessed image
            preprocessed_path = os.path.join(output_person_folder, f"processed_{img_name}")
            cv2.imwrite(preprocessed_path, foreground)
            print(f"Preprocessed and saved: {preprocessed_path}")

            # Convert to Pillow Image for augmentation
            pil_image = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))

            # Convert PIL image to numpy array for use with ImageDataGenerator
            img_array = np.array(pil_image)
            img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension

            # Step 5: Apply data augmentation with Keras ImageDataGenerator
            i = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_person_folder, 
                                      save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= augmentations:
                    break
            print(f"Generated {augmentations} augmented images for: {img_name}")

# Define input and output directories and call the function
input_directory = "Indian_actors_faces"  # Replace with the path to your dataset folder
output_directory = "augmented_dataset"   # Where augmented images will be saved
preprocess_and_augment_with_keras(input_directory, output_directory, augmentations=2)
