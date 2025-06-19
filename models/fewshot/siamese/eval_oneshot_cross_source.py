"""
Évaluation one-shot cross-source pour le modèle siamois LSF

Usage :
    python eval_oneshot_cross_source.py

Ce script :
- Charge l'encoder entraîné (siamese_encoder.keras)
- Identifie les classes avec au moins 2 sources dans data/processed/{sign}/{source}.npy
- Pour chaque classe à 2 sources :
    - Prend un exemple de chaque source (support, query)
    - Encode tous les supports
    - Pour chaque query, calcule la distance à tous les supports, prédit la classe la plus proche
    - Calcule l'accuracy one-shot (retrouve-t-il le bon support ?)
- Affiche le score global et par classe
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed'))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/siamese_encoder.keras'))
MAX_SEQ_LEN = 200

# --- Utilitaires ---
def pad_seq(arr, max_len=MAX_SEQ_LEN):
    if arr.shape[0] < max_len:
        return np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)), mode='constant')
    else:
        return arr[:max_len]

def find_classes_with_2_sources(data_dir):
    classes = {}
    for sign in sorted(os.listdir(data_dir)):
        sign_dir = os.path.join(data_dir, sign)
        if not os.path.isdir(sign_dir):
            continue
        npy_files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        if len(npy_files) >= 2:
            classes[sign] = npy_files
    return classes

def load_examples_for_class(sign, npy_files, data_dir):
    # On prend un exemple par source (si possible)
    examples = []
    for f in npy_files:
        arr = np.load(os.path.join(data_dir, sign, f))
        arr = pad_seq(arr)
        examples.append((f, arr))
    return examples

def compute_accuracy(encoder, classes, data_dir):
    n_total = 0
    n_correct = 0
    per_class = {}
    for sign, npy_files in classes.items():
        if len(npy_files) < 2:
            continue
        # On prend 2 sources différentes
        ex1 = npy_files[0]
        ex2 = npy_files[1]
        arr1 = pad_seq(np.load(os.path.join(data_dir, sign, ex1)))
        arr2 = pad_seq(np.load(os.path.join(data_dir, sign, ex2)))
        # Support set = [arr1], Query = arr2 (et inversement)
        supports = np.stack([arr1, arr2])
        support_labels = [sign, sign]
        queries = [arr2, arr1]
        query_labels = [sign, sign]
        # Encode
        emb_supports = encoder.predict(supports, verbose=0)
        emb_queries = encoder.predict(np.stack(queries), verbose=0)
        # Pour chaque query, trouver le support le plus proche
        for i, emb_q in enumerate(emb_queries):
            dists = np.linalg.norm(emb_supports - emb_q, axis=1)
            pred_idx = np.argmin(dists)
            pred_label = support_labels[pred_idx]
            true_label = query_labels[i]
            n_total += 1
            if pred_label == true_label:
                n_correct += 1
                per_class.setdefault(sign, []).append(1)
            else:
                per_class.setdefault(sign, []).append(0)
    acc = n_correct / n_total if n_total > 0 else 0.0
    return acc, per_class, n_total

# --- Main ---
def main():
    logger.info(f"Chargement de l'encoder depuis {ENCODER_PATH}")
    encoder = load_model(ENCODER_PATH)
    logger.info(f"Recherche des classes avec au moins 2 sources dans {DATA_DIR}")
    classes = find_classes_with_2_sources(DATA_DIR)
    logger.info(f"{len(classes)} classes avec au moins 2 sources trouvées.")
    acc, per_class, n_total = compute_accuracy(encoder, classes, DATA_DIR)
    logger.info(f"One-shot cross-source accuracy globale : {acc*100:.2f}% sur {n_total} essais.")
    # Affichage par classe (optionnel)
    for sign, results in per_class.items():
        logger.info(f"Classe {sign} : {np.mean(results)*100:.1f}% ({len(results)} essais)")

if __name__ == '__main__':
    main() 