import os
import numpy as np
import librosa
import torch
import traceback
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, flash, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import ViTImageProcessor, ViTForImageClassification
import mysql.connector

app = Flask(__name__)

app.secret_key = 'vaishnavi$deepfake_project2025!'


# MySQL Config
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="deepfake_detection"
)
cursor = db.cursor(dictionary=True)
# Load Audio Model
audio_model = tf.keras.models.load_model(r'C:\Users\gayat\OneDrive\Documents\Desktop\Final Year Project\DeepFakeDetection\model\model.h5')
max_length = 56293


# Load Image Model
model_dir =r"C:\Users\gayat\OneDrive\Documents\Desktop\Final Year Project\DeepFakeDetection\Image_model\deepfake_vs_real_image_detection\checkpoint-7142"

image_model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)
processor = ViTImageProcessor.from_pretrained(model_dir)

# --- Preprocessing functions ---

def preprocess_audio(audio_path, max_length):
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T
    padded_mfccs = pad_sequences([mfccs], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    return padded_mfccs

def preprocess_image(image_stream):
    image = Image.open(image_stream).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# --- Auth & Navigation ---

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            cursor.execute("SELECT * FROM users1 WHERE username = %s AND password = %s", (username, password))
            user = cursor.fetchone()
            if user:
                return redirect('/dashboard')
            else:
                flash('Invalid credentials!')
                return redirect('/login')
        except Exception as e:
            flash(f'Login failed: {str(e)}')
            return redirect('/login')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            # Check if user already exists
            cursor.execute("SELECT * FROM users1 WHERE username = %s", (username,))
            user = cursor.fetchone()
            if user:
                flash("Username already taken!")
                return redirect('/register')

            # Insert new user
            cursor.execute("INSERT INTO users1 (username, password) VALUES (%s, %s)", (username, password))
            db.commit()
            flash('Registration successful! Please login.')
            return redirect('/login')
        except Exception as e:
            db.rollback()
            flash(f'Error: {str(e)}')
            return redirect('/register')

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# --- Individual Modalities ---

@app.route('/audio')
def audio_home():
    return render_template('audio.html')

@app.route('/image')
def image_home():
    return render_template('image.html')

# --- Predictions ---

@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio = request.files['audio']
    audio_path = os.path.join('uploads', audio.filename)
    os.makedirs('uploads', exist_ok=True)
    audio.save(audio_path)

    padded_sample = preprocess_audio(audio_path, max_length)
    prediction = audio_model.predict(padded_sample)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])
    result = "Fake" if predicted_class == 1 else "Real"

    return jsonify({'prediction': result, 'confidence': confidence})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    image_input = preprocess_image(image)

    with torch.no_grad():
        output = image_model(**image_input)
    logits = output.logits
    predicted_class_id = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()
    label = image_model.config.id2label.get(predicted_class_id, "Unknown")

    return jsonify({'prediction': label, 'confidence': confidence})

@app.route('/predict_hybrid', methods=['POST'])
def predict_hybrid():
    if 'audio' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'Both audio and image files are required'}), 400

    audio_file = request.files['audio']
    image_file = request.files['image']
    os.makedirs('uploads', exist_ok=True)
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)

    audio_input = preprocess_audio(audio_path, max_length)
    audio_pred = audio_model.predict(audio_input)[0]

    image_input = preprocess_image(image_file)
    with torch.no_grad():
        image_output = image_model(**image_input)
    image_pred = torch.softmax(image_output.logits, dim=-1)[0].numpy()

    combined_pred = (audio_pred + image_pred) / 2
    predicted_class = int(np.argmax(combined_pred))
    confidence = float(combined_pred[predicted_class])
    label = "Fake" if predicted_class == 1 else "Real"

    return jsonify({'prediction': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
