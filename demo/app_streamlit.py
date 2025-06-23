import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
import sys
# Ajout du dossier racine au PYTHONPATH pour Streamlit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.landmark_utils import extract_landmark_vector
from src.data_processing.extract_landmarks import LandmarkExtractor

# --- Config ---
MODEL_TYPES = {
    'Siamese (Few-shot)': {
        'encoder_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/fewshot/siamese/siamese_encoder.keras')),
        'support_dir': os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed')),
        'type': 'siamese'
    },
    'Classic (LSTM)': {
        'model_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/classic/lsf_recognition_model.keras')),
        'type': 'classic'
    }
    # Ajoute d'autres modèles ici si besoin
}
MAX_SEQ_LEN = 200
THRESHOLD = st.sidebar.slider("Seuil d'inconnu (distance, few-shot)", 0.0, 10.0, 2.5, 0.1)

st.set_page_config(page_title="Reconnaissance LSF - Démo", layout="centered")
st.title("Reconnaissance LSF - Démo Universelle")
st.write("""
- Uploadez une vidéo de vous en train de signer (format mp4, webm, avi)
- Sélectionnez le modèle à utiliser (few-shot, classique, etc.)
- Les landmarks seront extraits et affichés
- Le modèle recherchera le signe le plus proche ou prédira la classe
""")

# --- Sélection du modèle ---
model_choice = st.sidebar.selectbox("Choisir le modèle à utiliser", list(MODEL_TYPES.keys()))
model_info = MODEL_TYPES[model_choice]

# --- Chargement du modèle ---
@st.cache_resource
def load_encoder(path):
    return load_model(path)

@st.cache_resource
def load_classic_model(path):
    return load_model(path)

@st.cache_resource
def load_support_set(support_dir):
    support_X, support_labels, support_files = [], [], []
    for sign in sorted(os.listdir(support_dir)):
        sign_dir = os.path.join(support_dir, sign)
        if not os.path.isdir(sign_dir):
            continue
        npy_files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        if not npy_files:
            continue
        arr = np.load(os.path.join(sign_dir, npy_files[0]))
        if arr.shape[0] < MAX_SEQ_LEN:
            arr = np.pad(arr, ((0, MAX_SEQ_LEN - arr.shape[0]), (0, 0)), mode='constant')
        else:
            arr = arr[:MAX_SEQ_LEN]
        support_X.append(arr)
        support_labels.append(sign)
        support_files.append(npy_files[0])
    return np.array(support_X), support_labels, support_files

# --- Extraction des landmarks ---
def extract_landmarks_from_video(video_path):
    extractor = LandmarkExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = extractor.holistic.process(image)
        keypoints = extract_landmark_vector(results)
        keypoints_list.append(keypoints)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        frames.append(frame.copy())
    cap.release()
    extractor.holistic.close()
    keypoints_arr = np.array(keypoints_list)
    if keypoints_arr.shape[0] < MAX_SEQ_LEN:
        keypoints_arr = np.pad(keypoints_arr, ((0, MAX_SEQ_LEN - keypoints_arr.shape[0]), (0,0)), mode='constant')
    else:
        keypoints_arr = keypoints_arr[:MAX_SEQ_LEN]
    return keypoints_arr, frames

# --- Inference Few-shot ---
def predict_sign_fewshot(encoder, support_X, support_labels, query_X, threshold=2.5):
    emb_supports = encoder.predict(support_X, verbose=0)
    emb_query = encoder.predict(np.expand_dims(query_X, 0), verbose=0)[0]
    dists = np.linalg.norm(emb_supports - emb_query, axis=1)
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    if min_dist > threshold:
        return None, min_dist, dists
    return support_labels[min_idx], min_dist, dists

# --- Inference Classic ---
def predict_sign_classic(model, query_X):
    # Suppose que le modèle classique attend (1, seq_len, features)
    X = np.expand_dims(query_X, 0)
    y_pred = model.predict(X, verbose=0)
    pred_idx = np.argmax(y_pred)
    confidence = np.max(y_pred)
    return pred_idx, confidence, y_pred

# --- UI ---
if model_info['type'] == 'siamese':
    encoder = load_encoder(model_info['encoder_path'])
    support_X, support_labels, support_files = load_support_set(model_info['support_dir'])
else:
    model = load_classic_model(model_info['model_path'])
    # Charger le mapping label->classe si besoin

uploaded_file = st.file_uploader("Uploader une vidéo de signe (mp4, webm, avi)", type=["mp4", "webm", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    st.video(video_path)
    st.info("Extraction des landmarks en cours...")
    keypoints_arr, frames = extract_landmarks_from_video(video_path)
    st.success(f"{keypoints_arr.shape[0]} frames extraites.")
    st.write("Aperçu des landmarks extraits :")
    for i in range(min(5, len(frames))):
        st.image(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), caption=f"Frame {i}")
    st.info("Recherche du signe...")
    if model_info['type'] == 'siamese':
        pred, min_dist, dists = predict_sign_fewshot(encoder, support_X, support_labels, keypoints_arr, threshold=THRESHOLD)
        if pred is not None:
            st.success(f"Signe reconnu : **{pred}** (distance = {min_dist:.2f})")
        else:
            st.warning(f"Aucun signe connu trouvé (distance minimale = {min_dist:.2f} > seuil)")
        top5_idx = np.argsort(dists)[:5]
        st.write("Top 5 des signes les plus proches :")
        for idx in top5_idx:
            st.write(f"- {support_labels[idx]} (distance = {dists[idx]:.2f})")
    else:
        pred_idx, confidence, y_pred = predict_sign_classic(model, keypoints_arr)
        st.success(f"Classe prédite : **{pred_idx}** (confiance = {confidence:.2f})")
        # Afficher le top 5 si possible
        top5_idx = np.argsort(y_pred[0])[::-1][:5]
        st.write("Top 5 classes :")
        for idx in top5_idx:
            st.write(f"- Classe {idx} (score = {y_pred[0][idx]:.2f})")
    os.unlink(video_path)
else:
    st.info("Veuillez uploader une vidéo pour commencer.") 