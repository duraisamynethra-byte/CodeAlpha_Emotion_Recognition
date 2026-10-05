import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import load_model


# Load data
X = np.load("data/mfcc_features.npy")
y = np.load("data/emotion_labels.npy")
X = (X - np.mean(X)) / np.std(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Load trained model
model = load_model("models/emotion_cnn.keras")

print("Model loaded successfully!")

# Predict emotions
predictions = model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Emotion Recognition - Confusion Matrix")
plt.tight_layout()

# Save image
plt.savefig("confusion_matrix.png", dpi=300)

print("Confusion matrix saved as confusion_matrix.png")