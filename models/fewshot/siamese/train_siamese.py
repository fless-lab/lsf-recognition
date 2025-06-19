import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed'))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../siamese/checkpoints'))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MAX_SEQ_LEN = 200
BATCH_SIZE = 16
EPOCHS = 50
EMBED_DIM = 128

# --- Data utils ---
def load_sequences(data_dir, max_per_class=100):
    """Charge les séquences .npy par classe."""
    X, y = [], []
    class_map = {}
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        files = files[:max_per_class]
        for f in files:
            arr = np.load(os.path.join(class_dir, f))
            if arr.shape[0] < MAX_SEQ_LEN:
                arr = np.pad(arr, ((0, MAX_SEQ_LEN - arr.shape[0]), (0, 0)), mode='constant')
            else:
                arr = arr[:MAX_SEQ_LEN]
            X.append(arr)
            y.append(label)
        class_map[label] = len(files)
    n_unique = sum(1 for v in class_map.values() if v == 1)
    n_double = sum(1 for v in class_map.values() if v > 1)
    logger.info(f"Loaded {len(X)} sequences from {len(class_map)} classes. {n_unique} uniques, {n_double} doublons ou +.")
    return np.array(X), np.array(y), class_map

def make_pairs(X, y, n_pairs=500):
    """Génère des paires (ancre, comparée) et labels (1=same, 0=diff)."""
    pairs, labels = [], []
    class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}
    # Séparer classes uniques et doublons
    unique_classes = [c for c, idxs in class_indices.items() if len(idxs) == 1]
    double_classes = [c for c, idxs in class_indices.items() if len(idxs) > 1]
    for _ in range(n_pairs):
        # Positive pair (possible seulement pour les doublons)
        if double_classes:
            c = random.choice(double_classes)
            idx1, idx2 = np.random.choice(class_indices[c], 2, replace=False)
            pairs.append([X[idx1], X[idx2]])
            labels.append(1)
        # Negative pair (possible pour toutes les classes)
        c1, c2 = random.sample(list(class_indices.keys()), 2)
        idx1 = np.random.choice(class_indices[c1])
        idx2 = np.random.choice(class_indices[c2])
        pairs.append([X[idx1], X[idx2]])
        labels.append(0)
    pairs = np.array(pairs)
    labels = np.array(labels)
    return [pairs[:,0], pairs[:,1]], labels

# --- Model ---
def build_siamese(input_shape, embed_dim=EMBED_DIM):
    inp = Input(shape=input_shape)
    x = layers.Masking(mask_value=0.)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    encoder = Model(inp, x, name='encoder')

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    emb_a = encoder(input_a)
    emb_b = encoder(input_b)
    l1_dist = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([emb_a, emb_b])
    out = layers.Dense(1, activation='sigmoid')(l1_dist)
    siamese = Model([input_a, input_b], out)
    return siamese, encoder

# --- Main ---
def main():
    X, y, class_map = load_sequences(DATA_DIR)
    pairs, labels = make_pairs(X, y, n_pairs=500)
    logger.info(f"Generated {len(labels)} pairs.")
    input_shape = (MAX_SEQ_LEN, X.shape[2])
    model, encoder = build_siamese(input_shape)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(pairs, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    model.save(os.path.join(CHECKPOINT_DIR, 'siamese_model.keras'))
    encoder.save(os.path.join(CHECKPOINT_DIR, 'siamese_encoder.keras'))
    logger.info("Models saved.")

if __name__ == '__main__':
    main() 