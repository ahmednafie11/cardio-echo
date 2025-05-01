from flask import Flask, request, render_template
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.video import r2plus1d_18
import firebase_admin
from firebase_admin import credentials, firestore, auth
import traceback

# Désactiver explicitement l'émulateur Firestore
if os.environ.get("FIRESTORE_EMULATOR_HOST"):
   del  os.environ["FIRESTORE_EMULATOR_HOST"]

# Initialiser l'application Flask
app = Flask(__name__)

# Initialiser Firebase Admin
try:
    cred = credentials.Certificate("/Users/mac/cardio-echo/firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase Admin: {str(e)}")

# Chemins
weights_path = "/Users/mac/cardio-echo/weights/r2plus1d_18_official.pth"
upload_folder = "/Users/mac/cardio-echo/uploads"
os.makedirs(upload_folder, exist_ok=True)

# Initialisation du modèle
model = None
model_error = None
try:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    model = r2plus1d_18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict.pop('fc.0.weight', None)
    state_dict.pop('fc.0.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    model_error = str(e)
    print(f"Error loading model: {model_error}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return f"Error: Model not initialized. Details: {model_error}", 500
    try:
        # Vérifier le token Firebase
        auth_header = request.headers.get('Authorization')
        user_id = 'anonymous'
        if auth_header and auth_header.startswith('Bearer '):
            id_token = auth_header.split('Bearer ')[1]
            try:
                decoded_token = auth.verify_id_token(id_token)
                user_id = decoded_token['uid']
            except auth.InvalidIdTokenError as e:
                print(f"Invalid token: {str(e)}")

        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        print(f"File saved: {file_path}")

        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))  # [112, 112, 3]
            frame = frame / 255.0  # Normalisation
            frames.append(frame)
        cap.release()
        if not frames:
            return "No frames extracted", 400
        print(f"Extracted {len(frames)} frames")

        # Convertir en array : [num_frames, 112, 112, 3]
        frames = np.array(frames)
        print(f"Frames shape: {frames.shape}")
        # Transposer pour obtenir [num_frames, 3, 112, 112]
        frames = frames.transpose(0, 3, 1, 2)
        # Convertir en tenseur et ajouter batch : [1, 3, num_frames, 112, 112]
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, num_frames, 112, 112]
        print(f"Tensor shape: {frames.shape}")

        with torch.no_grad():
            lvef = model(frames).item() * 100
        print(f"Predicted LVEF: {lvef}")

        # Sauvegarder dans Firestore
        try:
            db.collection('analyses').add({
                'user': user_id,
                'filename': file.filename,
                'lvef': lvef,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            print("Data saved to Firestore")
        except Exception as e:
            print(f"Error saving to Firestore: {str(e)}")

        return f"Echocardiography video processed, predicted LVEF: {lvef:.1f}%"
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        print(traceback.format_exc())
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)
