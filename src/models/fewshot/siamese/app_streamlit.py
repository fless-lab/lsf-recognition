import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model
import mediapipe as mp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/data_processing')))
import landmark_utils

st.set_page_config(page_title="Reconnaissance LSF One-Shot", layout="centered")
st.title("Reconnaissance LSF One-Shot (Siamese)")
st.write("""
- Uploadez une vidéo de vous en train de signer (format mp4, webm, avi)
- Les landmarks seront extraits et affichés
- Le modèle recherchera le signe le plus proche dans la base
- Si la distance est trop grande, le signe sera considéré comme inconnu
""")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed'))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/siamese_encoder.keras'))
MAX_SEQ_LEN = 200
THRESHOLD = st.sidebar.slider("Seuil d'inconnu (distance)", 0.0, 10.0, 2.5, 0.1)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def load_encoder():
    return load_model(ENCODER_PATH)

@st.cache_resource
def load_support_set():
    support_X, support_labels, support_files = [], [], []
    for sign in sorted(os.listdir(DATA_DIR)):
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(sign_dir):
            continue
        npy_files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        if not npy_files:
            continue
        # On prend le premier exemple par classe
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
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # Extraction des landmarks via la fonction utilitaire
        keypoints = landmark_utils.extract_landmark_vector(results)
        keypoints_list.append(keypoints)
        # Pour affichage
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        frames.append(frame.copy())
    cap.release()
    holistic.close()
    keypoints_arr = np.array(keypoints_list)
    # Padding/troncature
    if keypoints_arr.shape[0] < MAX_SEQ_LEN:
        keypoints_arr = np.pad(keypoints_arr, ((0, MAX_SEQ_LEN - keypoints_arr.shape[0]), (0,0)), mode='constant')
    else:
        keypoints_arr = keypoints_arr[:MAX_SEQ_LEN]
    return keypoints_arr, frames

# --- Inference ---
def predict_sign(encoder, support_X, support_labels, query_X, threshold=2.5):
    emb_supports = encoder.predict(support_X, verbose=0)
    emb_query = encoder.predict(np.expand_dims(query_X, 0), verbose=0)[0]
    dists = np.linalg.norm(emb_supports - emb_query, axis=1)
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    if min_dist > threshold:
        return None, min_dist, dists
    return support_labels[min_idx], min_dist, dists

# --- UI ---
encoder = load_encoder()
support_X, support_labels, support_files = load_support_set()

uploaded_file = st.file_uploader("Uploader une vidéo de signe (mp4, webm, avi)", type=["mp4", "webm", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    st.video(video_path)
    st.info("Extraction des landmarks en cours...")
    keypoints_arr, frames = extract_landmarks_from_video(video_path)
    st.success(f"{keypoints_arr.shape[0]} frames extraites.")
    # Affichage des landmarks sur la vidéo (premières frames)
    st.write("Aperçu des landmarks extraits :")
    for i in range(min(5, len(frames))):
        st.image(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), caption=f"Frame {i}")
    # Prédiction
    st.info("Recherche du signe le plus proche...")
    pred, min_dist, dists = predict_sign(encoder, support_X, support_labels, keypoints_arr, threshold=THRESHOLD)
    if pred is not None:
        st.success(f"Signe reconnu : **{pred}** (distance = {min_dist:.2f})")
    else:
        st.warning(f"Aucun signe connu trouvé (distance minimale = {min_dist:.2f} > seuil)")
    # Affichage des 5 signes les plus proches
    top5_idx = np.argsort(dists)[:5]
    st.write("Top 5 des signes les plus proches :")
    for idx in top5_idx:
        st.write(f"- {support_labels[idx]} (distance = {dists[idx]:.2f})")
    # Nettoyage
    os.unlink(video_path)
else:
    st.info("Veuillez uploader une vidéo pour commencer.") 