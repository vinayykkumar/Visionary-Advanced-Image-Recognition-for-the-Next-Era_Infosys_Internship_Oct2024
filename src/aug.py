import os
import cv2
from glob import glob

# Paths
input_dataset_path = r'C:\Users\Shyam\OneDrive\Desktop\project\Visionary\My DataSet\My dataset'  # Input dataset folder
output_dataset_path = r'C:\Users\Shyam\OneDrive\Desktop\project\Visionary\src\Augmentdataset'  # Augmented dataset output folder

# Ensure the output directory exists
os.makedirs(output_dataset_path, exist_ok=True)

# Define augmentation functions
def resize_image(image, dimensions=(200, 200)):
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def flip_image_horizontally(image):
    return cv2.flip(image, 1)

def rotate_image_by_angle(image, angle=30):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def increase_brightness(image, value=50):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

# Iterate through class folders
for class_folder in os.listdir(input_dataset_path):
    class_input_path = os.path.join(input_dataset_path, class_folder)

    # Skip files, process only directories
    if not os.path.isdir(class_input_path):
        continue

    # Create corresponding class output directory
    class_output_path = os.path.join(output_dataset_path, class_folder)
    os.makedirs(class_output_path, exist_ok=True)

    # Gather all image paths in the class folder
    image_files = glob(os.path.join(class_input_path, '*.jpg'))

    # Process each image
    for i, img_file in enumerate(image_files):
        # Load the image
        original_image = cv2.imread(img_file)
        if original_image is None:
            print(f"Error: Could not load image {img_file}")
            continue

        # Apply transformations
        resized = resize_image(original_image)
        grayscale = convert_to_grayscale(resized)
        flipped = flip_image_horizontally(resized)
        rotated = rotate_image_by_angle(resized)
        brightened = increase_brightness(resized)

        # Save augmented images
        base_filename = os.path.splitext(os.path.basename(img_file))[0]
        cv2.imwrite(os.path.join(class_output_path, f"{base_filename}_resized.jpg"), resized)
        cv2.imwrite(os.path.join(class_output_path, f"{base_filename}_grayscale.jpg"), grayscale)
        cv2.imwrite(os.path.join(class_output_path, f"{base_filename}_flipped.jpg"), flipped)
        cv2.imwrite(os.path.join(class_output_path, f"{base_filename}_rotated.jpg"), rotated)
        cv2.imwrite(os.path.join(class_output_path, f"{base_filename}_brightened.jpg"), brightened)

        # Log progress
        print(f"[{class_folder}] Processed {i + 1}/{len(image_files)}: {base_filename}")
