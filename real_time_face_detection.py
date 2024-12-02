import cv2
import numpy as np
from tensorflow.keras.models import load_model


def load_labels(label_path):
    """
    Load label encoder classes.

    Args:
    - label_path: str, path to the label encoder file.

    Returns:
    - labels: list, class names.
    """
    return np.load(label_path)



def preprocess_image(image, img_size=(128, 128)):
    """
    Preprocess an image for prediction.

    Args:
    - image: np.ndarray, input image.
    - img_size: tuple, target size for resizing.

    Returns:
    - preprocessed_image: np.ndarray, preprocessed image.
    """
    image = cv2.resize(image, img_size)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)


def recognize_faces(model_path, label_path, img_size=(128, 128)):
    """
    Perform real-time face recognition using a webcam.

    Args:
    - model_path: str, path to the trained model.
    - label_path: str, path to the label encoder file.
    - img_size: tuple, target size for resizing images.
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

    # Load model and labels
    model = load_model(model_path)
    labels = load_labels(label_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            processed_face = preprocess_image(face, img_size)

            # Predict class with suppressed verbosity
            predictions = model.predict(processed_face, verbose=0)  # Suppressed output
            class_id = np.argmax(predictions)
            confidence = predictions[0][class_id]
            label = labels[class_id]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show the video frame
        cv2.imshow("Real-Time Face Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
