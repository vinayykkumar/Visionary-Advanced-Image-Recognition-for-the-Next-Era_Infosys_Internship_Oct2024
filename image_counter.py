import os

def count_images_in_directory(directory):
    """
    Counts the number of image files in a directory (including subdirectories).
    """
    count = 0
    for root, _, files in os.walk(directory):
        count += len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    return count
