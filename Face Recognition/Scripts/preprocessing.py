for folder in celebrity_folders:
    # Define the path to the current celebrity folder
    folder_path = os.path.join(cropped_face, folder)
    output_folder = os.path.join(preprocessed, folder)  # Create a corresponding output folder for each celebrity

    # Ensure the output folder for the celebrity
    os.makedirs(output_folder, exist_ok=True)

    # ***List all image files in the current folder***
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Track if we've displayed a sample image for this celebrity
    displayed_sample = False

    # ***Loop over each image in the folder***
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Initialize a plot for the sample display
        if not displayed_sample:
            fig, axs = plt.subplots(1, 8, figsize=(24, 5))  # 8 subplots to include the original image
            fig.suptitle(f'Sample Image Processing Steps - {folder.replace("_", " ").title()}')

        # Display the Original Image
        if not displayed_sample:
            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
            axs[0].set_title(f'{image_file} - Original')
            axs[0].axis('off')

        # Step 1: Convert to Grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not displayed_sample:
            axs[1].imshow(gray_image, cmap='gray')
            axs[1].set_title('Grayscale')
            axs[1].axis('off')

        # Step 2: Resize the Image
        resized_image = cv2.resize(gray_image, (300, 300))
        if not displayed_sample:
            axs[2].imshow(resized_image, cmap='gray')
            axs[2].set_title('Resized')
            axs[2].axis('off')

        # Step 3: Histogram Equalization
        equalized_image = cv2.equalizeHist(resized_image)
        if not displayed_sample:
            axs[3].imshow(equalized_image, cmap='gray')
            axs[3].set_title('Equalized')
            axs[3].axis('off')

        # Step 4: Image Smoothing (Blurring)
        blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
        if not displayed_sample:
            axs[4].imshow(blurred_image, cmap='gray')
            axs[4].set_title('Blurred')
            axs[4].axis('off')

        # Step 5: Apply Edge Detection (Optional)
        edges = cv2.Canny(blurred_image, 100, 200)
        if not displayed_sample:
            axs[5].imshow(edges, cmap='gray')
            axs[5].set_title('Edges')
            axs[5].axis('off')

        # Step 7: Normalization
        normalized_image = blurred_image / 255.0
        if not displayed_sample:
            axs[6].imshow((normalized_image * 255).astype(np.uint8), cmap='gray')
            axs[6].set_title('Normalized')
            axs[6].axis('off')

        # Step 8: Image Standardization (Mean Subtraction and Scaling)
        mean, std = cv2.meanStdDev(normalized_image)
        standardized_image = (normalized_image - mean[0][0]) / (std[0][0] + 1e-8)
        if not displayed_sample:
            axs[7].imshow((standardized_image * 255).astype(np.uint8), cmap='gray')
            axs[7].set_title('Standardized')
            axs[7].axis('off')

        # Save the processed image in the output folder
        output_image_path = os.path.join(output_folder, f'processed_{image_file}')
        cv2.imwrite(output_image_path, (standardized_image * 255).astype(np.uint8))

        # Mark that we've displayed a sample for this celebrity
        if not displayed_sample:
            plt.show()
            displayed_sample = True

print("All images in all folders have been processed.")
