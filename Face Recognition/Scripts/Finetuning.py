
# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Compile the model again with a lower learning rate for fine-tuning
cnn_model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training (fine-tuning)
history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

