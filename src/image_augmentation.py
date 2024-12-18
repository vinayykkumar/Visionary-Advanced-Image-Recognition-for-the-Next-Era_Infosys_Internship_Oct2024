import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

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
        zoom_factor = random.uniform(0.8, 1.2)
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if zoom_factor > 1:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        else:
            image = ImageOps.pad(image, (width, height), color=(0, 0, 0))

    return image

def preprocess_and_augment_with_pillow(input_dir, output_dir, img_size=(128, 128), augmentations=4):
    """
    Preprocesses and augments images using OpenCV and Pillow.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the subfolders (person folders) in the input directory
    for person_name in os.listdir(input_dir):
        person_folder = os.path.join(input_dir, person_name)
        output_person_folder = os.path.join(output_dir, person_name)

        # Skip non-folder items
        if not os.path.isdir(person_folder):
            continue

        # Create the person folder in the output directory (ensure folder structure is maintained)
        if not os.path.exists(output_person_folder):
            os.makedirs(output_person_folder)

        # Iterate over all the images in each person's folder
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            # Read image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Preprocess the image: resize, bilateral filter, convert to grayscale, threshold, and extract foreground
            image_resized = cv2.resize(image, img_size)
            image_bilateral = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(image_bilateral, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            foreground = cv2.bitwise_and(image_bilateral, image_bilateral, mask=thresholded)

            # Save the preprocessed image to the output directory
            preprocessed_path = os.path.join(output_person_folder, f"processed_{img_name}")
            cv2.imwrite(preprocessed_path, foreground)
            print(f"Preprocessed and saved: {preprocessed_path}")

            # Convert to PIL image for augmentation
            pil_image = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))

            # Perform augmentations and save them
            for i in range(augmentations):
                augmented_image = augment_image(pil_image)
                aug_img_path = os.path.join(output_person_folder, f"aug_{i}_{img_name}")
                augmented_image.save(aug_img_path)
                print(f"Augmented and saved: {aug_img_path}")



# # Example usage:
# input_dataset_dir = "data/My dataset"
# output_dataset_dir = "data/augmented_dataset"
# # Call the function to preprocess and augment images
# preprocess_and_augment_with_pillow(input_dataset_dir, output_dataset_dir, img_size=(128, 128), augmentations=4)
