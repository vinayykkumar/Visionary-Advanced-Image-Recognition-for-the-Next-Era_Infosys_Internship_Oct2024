import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, label_encoder, X_val, y_val, class_names):
    """
    Evaluate model using validation data.
    """
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Classification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
