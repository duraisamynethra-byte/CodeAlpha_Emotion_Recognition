# CodeAlpha Emotion Recognition from Speech

## Project Overview

This project is an Emotion Recognition from Speech system developed as part of the CodeAlpha Machine Learning Internship.

The system analyzes speech audio and predicts the emotion expressed in the speech using **MFCC (Mel-Frequency Cepstral Coefficients)** and a **2D Convolutional Neural Network (CNN)**.

## Objective

The objective is to recognize human emotions such as:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** speech dataset.

The dataset contains speech recordings representing different emotional states.

## Technologies Used

- Python
- TensorFlow
- Keras
- Librosa
- NumPy
- Scikit-learn
- Matplotlib

## Methodology

The project follows these steps:

1. Load speech audio files.
2. Extract MFCC features using Librosa.
3. Standardize the MFCC features.
4. Encode the emotion labels.
5. Split the dataset into training and testing sets.
6. Train a 2D CNN model.
7. Evaluate the model using test accuracy and a confusion matrix.
8. Predict emotions from new WAV audio files.

## Model Architecture

The 2D CNN contains:

- Convolutional layers
- Max Pooling layers
- Flatten layer
- Dense layer
- Dropout layer
- Softmax output layer

## Performance

The final normalized 2D CNN achieved:

**Test Accuracy: 52.08%**

A confusion matrix is included in the project as:

`confusion_matrix.png`

## How to Run

Install the required libraries:

```bash
pip install numpy librosa tensorflow scikit-learn matplotlib