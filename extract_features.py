import os
import librosa
import numpy as np
import pandas as pd

# Dataset path
DATASET_PATH = "data/ravdess/Audio_Speech_Actors_01-24_16k"

# Emotion labels from RAVDESS
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
labels = []

print("Starting MFCC feature extraction...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in os.listdir(actor_path):

        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, filename)

        # Get emotion code from filename
        emotion_code = filename.split("-")[2]
        emotion = emotion_map.get(emotion_code)

        if emotion is None:
            continue

        # Load audio
        audio, sample_rate = librosa.load(file_path, sr=16000)

        # Extract full MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Make every audio file the same length
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

        features.append(mfcc)
        labels.append(emotion)

print("Feature extraction completed!")

# Convert to NumPy array
X = np.array(features)
y = np.array(labels)

# Save features and labels
np.save("data/mfcc_features.npy", X)
np.save("data/emotion_labels.npy", y)

print("Total audio files processed:", len(X))
print("MFCC shape:", X.shape)
print("MFCC features saved to data/mfcc_features.npy")
print("Emotion labels saved to data/emotion_labels.npy")