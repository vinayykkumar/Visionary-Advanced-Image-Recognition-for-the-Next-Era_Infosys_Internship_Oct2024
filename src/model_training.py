
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes, train_data):
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Load VGG16 pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Unfreeze the last 4 layers of the base model for fine-tuning
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # Build the model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model with SGD and momentum
    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, class_weights_dict

def train_model(model, train_data, val_data, class_weights_dict, epochs=30):
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weights_dict  # Add if dataset is imbalanced
    )
    return history

# Example of how to use these functions:
# Assuming `train_data` and `val_data` are preloaded datasets for training and validation

# Input shape of images, modify as per your dataset
input_shape = (128, 128, 3)

# Number of classes in your dataset
num_classes = len(train_data.class_indices)

# Create model and class weights
model, class_weights_dict = create_model(input_shape, num_classes, train_data)

# Train the model
history = train_model(model, train_data, val_data, class_weights_dict, epochs=30)

