import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random

def augment_image(image):
    """
    Apply random augmentations to an image using Pillow.
    """
    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
        image = image.rotate(angle, expand=True)

    # Random horizontal flip
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # Random brightness adjustment
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.8, 1.2)  # Brightness factor
        image = enhancer.enhance(factor)

    # Random zoom (crop and resize back)
    if random.random() > 0.5:
        width, height = image.size
        zoom_factor = random.uniform(0.8, 1.2)  # Zoom in/out factor
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Use Resampling.LANCZOS
        if zoom_factor > 1:
            # Crop to original size
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        else:
            # Pad to original size
            image = ImageOps.pad(image, (width, height), color=(0, 0, 0))

    return image

def preprocess_and_augment_with_pillow(input_dir, output_dir, img_size=(128, 128), augmentations=2):
    """
    Preprocesses and augments images using OpenCV and Pillow.
    
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the output directory for preprocessed and augmented images.
        img_size (tuple): Target size for resizing the images (width, height).
        augmentations (int): Number of augmented images to generate per input image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

            # Step 5: Apply data augmentation with Pillow
            for i in range(augmentations):
                augmented_image = augment_image(pil_image)  # Apply random augmentations
                aug_img_path = os.path.join(output_person_folder, f"aug_{i}_{img_name}")
                augmented_image.save(aug_img_path)
                print(f"Augmented and saved: {aug_img_path}")

# Define input and output directories and call the function
input_directory = "Indian_actors_faces"          # Replace with the path to your dataset folder
output_directory = "augmented_dataset"           # Where augmented images will be saved
preprocess_and_augment_with_pillow(input_directory, output_directory, augmentations=2)
