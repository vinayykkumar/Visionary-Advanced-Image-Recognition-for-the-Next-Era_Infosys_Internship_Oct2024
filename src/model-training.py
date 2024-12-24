
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Function to build an improved model
def build_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
   
    # Fine-tuning: Unfreeze the top layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all layers except the last 30
        layer.trainable = False
    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation="relu", kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    # Compile with an advanced optimizer
    model.compile(optimizer=Adam(learning_rate=0.0005),  # Use smaller LR for fine-tuning
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Function to train the model
def train_model(train_data_dir, val_data_dir, output_model_path, img_size=(224, 224), batch_size=16, epochs=30):
    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load the datasets
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    val_generator = val_datagen.flow_from_directory(
        val_data_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )

    num_classes = len(train_generator.class_indices)

    # Build the model
    model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)

    # Callbacks for saving the best model, early stopping, and learning rate scheduling
    callbacks = [
        ModelCheckpoint(output_model_path, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"Model saved at {output_model_path}")
    return history

# Plotting function to visualize training progress
def plot_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# Example usage (adjust paths as necessary)
train_data_dir = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\src\Processed\train"
val_data_dir = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\src\Processed\val"
output_model_path = r"C:\Users\Shyam\OneDrive\Desktop\project\Visionary\models\\image_model.keras"

# Train the model and visualize progress
history = train_model(train_data_dir, val_data_dir, output_model_path, img_size=(224, 224), batch_size=16, epochs=30)
plot_history(history)

