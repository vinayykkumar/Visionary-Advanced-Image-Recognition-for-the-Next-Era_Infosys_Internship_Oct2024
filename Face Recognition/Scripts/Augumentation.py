
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
main_folder_path = '/content/drive/MyDrive/Infosys _internship/Output data'  # Original dataset path
augmented_data_path = '/content/drive/MyDrive/Infosys _internship/Augumented data'  # Folder where augmented images will be saved

# Create the augmented data directory if it doesn't exist
os.makedirs(augmented_data_path, exist_ok=True)

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each subfolder (celebrity folder)
celebrity_folders = [f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
for folder in celebrity_folders:
    folder_path = os.path.join(main_folder_path, folder)

    # Create a corresponding folder in the augmented data directory for each celebrity
    augmented_folder_path = os.path.join(augmented_data_path, folder)
    os.makedirs(augmented_folder_path, exist_ok=True)

    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Loop through each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Convert image to RGB for augmentation (Keras works with RGB images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape the image to fit the Keras generator
        image_array = np.expand_dims(image_rgb, axis=0)

        # Apply augmentations and save augmented images
        image_count = 0
        for x_batch in datagen.flow(image_array, batch_size=1, save_to_dir=augmented_folder_path, save_prefix=folder, save_format='jpeg'):
            image_count += 1
            if image_count >= 3:  # Save 3 augmented images for each original image
                break

print("Data augmentation complete and saved to the new folder.")
