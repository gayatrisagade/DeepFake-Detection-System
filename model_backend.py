
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model (call this after training and saving model)
model = tf.keras.models.load_model("model.h5")

def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_processed = mfccs.T
    # Pad/truncate for fixed length if needed
    if mfccs_processed.shape[0] < 100:
        pad_width = 100 - mfccs_processed.shape[0]
        mfccs_processed = np.pad(mfccs_processed, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs_processed = mfccs_processed[:100, :]
    return np.expand_dims(mfccs_processed, axis=0)

def predict_audio(audio_path):
    features = extract_mfcc(audio_path)
    prediction = model.predict(features)
    label = "Real" if prediction[0][0] > 0.5 else "Fake"
    return label, float(prediction[0][0])
