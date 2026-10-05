import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Load MFCC features
X = np.load("data/mfcc_features.npy")
y = np.load("data/emotion_labels.npy")
X = (X - np.mean(X)) / np.std(X)

print("MFCC features loaded successfully!")
print("Feature shape:", X.shape)


# Add channel dimension for 2D CNN
X = X[..., np.newaxis]

print("CNN input shape:", X.shape)


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# Calculate class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)


# Build 2D CNN model
model = Sequential([
    Input(shape=(40, 174, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(y_categorical.shape[1], activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n2D CNN model created successfully!")
model.summary()


# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# Train model
print("\nStarting 2D CNN training...")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
# Save training history
np.save("data/training_history.npy", history.history)

# Evaluate model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel evaluation completed!")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")


# Predictions
y_pred = model.predict(X_test, verbose=0)

y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test_classes,
        y_pred_classes,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# Create models folder
os.makedirs("models", exist_ok=True)


# Save model
model.save("models/emotion_cnn.keras")

# Save emotion labels
np.save(
    "models/emotion_labels.npy",
    label_encoder.classes_
)


print("\nModel saved to: models/emotion_cnn.keras")
print("Emotion labels saved to: models/emotion_labels.npy")

print("\nTraining completed successfully!")