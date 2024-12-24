
import cv2
import numpy as np
import os

def preprocess_images(folder_path, target_size=(128, 128)):
    X = []
    y = []
    class_labels = sorted(os.listdir(folder_path))

    for idx, label in enumerate(class_labels):
        class_folder = os.path.join(folder_path, label)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                X.append(img)
                y.append(idx)

    X = np.array(X) / 255.0  # Normalize images
    y = np.array(y)
    return X, y, class_labels
