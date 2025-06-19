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
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../metric/checkpoints'))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MAX_SEQ_LEN = 200
EMBED_DIM = 128
BATCH_SIZE = 16
EPOCHS = 50

# --- Data utils ---
def load_sequences(data_dir, max_per_class=100):
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
    logger.info(f"Loaded {len(X)} sequences from {len(class_map)} classes.")
    return np.array(X), np.array(y), class_map

def make_triplets(X, y, n_triplets=10000):
    triplets = []
    class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}
    for _ in range(n_triplets):
        c = random.choice(list(class_indices.keys()))
        a, p = np.random.choice(class_indices[c], 2, replace=True)
        c_neg = random.choice([k for k in class_indices.keys() if k != c])
        n = np.random.choice(class_indices[c_neg])
        triplets.append([X[a], X[p], X[n]])
    triplets = np.array(triplets)
    return [triplets[:,0], triplets[:,1], triplets[:,2]]

# --- Model ---
def build_encoder(input_shape, embed_dim=EMBED_DIM):
    inp = Input(shape=input_shape)
    x = layers.Masking(mask_value=0.)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    return Model(inp, x, name='encoder')

def triplet_loss(a, p, n, margin=1.0):
    pos_dist = tf.reduce_sum(tf.square(a - p), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(a - n), axis=-1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

# --- Main ---
def main():
    X, y, class_map = load_sequences(DATA_DIR)
    input_shape = (MAX_SEQ_LEN, X.shape[2])
    encoder = build_encoder(input_shape)
    optimizer = Adam(1e-3)
    for epoch in range(EPOCHS):
        triplets = make_triplets(X, y, n_triplets=1000)
        with tf.GradientTape() as tape:
            a_emb = encoder(triplets[0])
            p_emb = encoder(triplets[1])
            n_emb = encoder(triplets[2])
            loss = triplet_loss(a_emb, p_emb, n_emb)
        grads = tape.gradient(loss, encoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, encoder.trainable_weights))
        logger.info(f"Epoch {epoch}: loss={loss.numpy():.4f}")
    encoder.save(os.path.join(CHECKPOINT_DIR, 'metric_encoder.keras'))
    logger.info("Encoder saved.")

if __name__ == '__main__':
    main() 