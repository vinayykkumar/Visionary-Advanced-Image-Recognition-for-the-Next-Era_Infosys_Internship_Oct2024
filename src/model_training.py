import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def load_dataset(dataset_dir, img_size=(160, 160)):
    """
    Load images and labels from the specified dataset directory.
    """
    X, y = [], []
    label_names = []

    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_names.append(person_name)
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(person_name)

    X = np.array(X, dtype='float32') / 255.0  # Normalize images
    y = np.array(y)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

def build_mobilenetv2_model(input_shape, num_classes, learning_rate=0.001):
    """
    Build and compile a MobileNetV2-based CNN model.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Fine-tune the base model by unfreezing some layers
    base_model.trainable = True
    for layer in base_model.layers[:-4]:  # Freeze all but the last 4 layers
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(dataset_dir, output_model_path, img_size=(160, 160), learning_rate=0.001):
    """
    Train and save a face recognition model.
    """
    X, y, label_encoder = load_dataset(dataset_dir, img_size)
    num_classes = len(np.unique(y))  # Number of unique classes based on labels

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Build the model
    model = build_mobilenetv2_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=20,
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    # Save the model and label encoder
    model.save(output_model_path)
    np.save(output_model_path + "_labels.npy", label_encoder.classes_)
    print(f"Model saved at {output_model_path}")

    return history
