import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dataset_dir = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\My DataSet\My dataset"
train_data_dir = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\src\Processed\\train"
val_data_dir = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\src\Processed\\val"

# Create directories if they do not exist
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(val_data_dir, exist_ok=True)

# Split dataset
def split_dataset(original_dir, train_dir, val_dir, test_size=0.2):
    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if os.path.isdir(class_dir):
            # Create class-specific directories in train and val folders
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            # Get all file paths for the current class
            file_paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]

            # Split into train and validation
            train_files, val_files = train_test_split(file_paths, test_size=test_size, random_state=42)

            # Move files to respective directories
            for file_path in train_files:
                shutil.copy(file_path, os.path.join(train_dir, class_name))

            for file_path in val_files:
                shutil.copy(file_path, os.path.join(val_dir, class_name))

    print(f"Dataset successfully split into training and validation sets.")

# Call the function
split_dataset(original_dataset_dir, train_data_dir, val_data_dir, test_size=0.2)
