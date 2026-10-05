import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model


# Check audio file argument
if len(sys.argv) < 2:
    print("Please provide a WAV audio file.")
    print("Example: python predict_emotion.py sample.wav")
    sys.exit()


audio_file = sys.argv[1]


# Load trained model
model = load_model("models/emotion_cnn.keras")

# Load emotion labels
emotion_labels = np.load(
    "models/emotion_labels.npy",
    allow_pickle=True
)


# Load audio
audio, sample_rate = librosa.load(
    audio_file,
    sr=16000
)


# Extract MFCC
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)


# Make same length as training data
max_length = 174

if mfcc.shape[1] < max_length:
    pad_width = max_length - mfcc.shape[1]

    mfcc = np.pad(
        mfcc,
        pad_width=((0, 0), (0, pad_width)),
        mode="constant"
    )
else:
    mfcc = mfcc[:, :max_length]
# Normalize MFCC features the same way as training
mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

# Add dimensions for CNN
mfcc = mfcc[np.newaxis, ..., np.newaxis]


# Predict emotion
prediction = model.predict(mfcc, verbose=0)

predicted_index = np.argmax(prediction)
predicted_emotion = emotion_labels[predicted_index]

confidence = prediction[0][predicted_index] * 100


print("\nEmotion Recognition Result")
print("--------------------------")
print("Predicted Emotion:", predicted_emotion)
print("Confidence:", round(confidence, 2), "%")