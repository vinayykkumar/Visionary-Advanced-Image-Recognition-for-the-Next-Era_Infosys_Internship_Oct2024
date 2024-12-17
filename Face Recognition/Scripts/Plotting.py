import matplotlib.pyplot as plt

# Assuming history contains the training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])  # Access accuracy directly from the dictionary
plt.plot(history['val_accuracy'])  # Access val_accuracy directly from the dictionary
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['loss'])  # Access loss directly from the dictionary
plt.plot(history['val_loss'])  # Access val_loss directly from the dictionary
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')

plt.show()