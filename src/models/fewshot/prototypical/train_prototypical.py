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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prototypical/checkpoints'))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MAX_SEQ_LEN = 200
EMBED_DIM = 128
N_WAY = 10
N_SHOT = 1
N_QUERY = 5
EPISODES = 1000

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

def sample_episode(X, y, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY):
    classes = np.random.choice(np.unique(y), n_way, replace=False)
    support, support_labels, query, query_labels = [], [], [], []
    for i, c in enumerate(classes):
        idx = np.where(y == c)[0]
        chosen = np.random.choice(idx, n_shot + n_query, replace=False)
        support.extend(X[chosen[:n_shot]])
        support_labels.extend([i]*n_shot)
        query.extend(X[chosen[n_shot:]])
        query_labels.extend([i]*n_query)
    return (np.array(support), np.array(support_labels), np.array(query), np.array(query_labels))

# --- Model ---
def build_encoder(input_shape, embed_dim=EMBED_DIM):
    inp = Input(shape=input_shape)
    x = layers.Masking(mask_value=0.)(inp)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    return Model(inp, x, name='encoder')

def prototypical_loss(support, support_labels, query, query_labels, encoder):
    # Encode
    support_emb = encoder(support)
    query_emb = encoder(query)
    n_way = tf.shape(tf.unique(support_labels)[0])[0]
    # Compute prototypes
    prototypes = []
    for i in range(n_way):
        mask = tf.equal(support_labels, i)
        proto = tf.reduce_mean(tf.boolean_mask(support_emb, mask), axis=0)
        prototypes.append(proto)
    prototypes = tf.stack(prototypes)
    # Distances
    dists = tf.norm(tf.expand_dims(query_emb,1) - tf.expand_dims(prototypes,0), axis=-1)
    # Softmax over -distance
    log_p_y = tf.nn.log_softmax(-dists, axis=1)
    loss = -tf.reduce_mean(tf.gather_nd(log_p_y, tf.stack([tf.range(tf.shape(query_labels)[0]), query_labels], axis=1)))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(log_p_y, axis=1), query_labels), tf.float32))
    return loss, acc

# --- Main ---
def main():
    X, y, class_map = load_sequences(DATA_DIR)
    input_shape = (MAX_SEQ_LEN, X.shape[2])
    encoder = build_encoder(input_shape)
    optimizer = Adam(1e-3)
    for episode in range(EPISODES):
        support, support_labels, query, query_labels = sample_episode(X, y)
        with tf.GradientTape() as tape:
            loss, acc = prototypical_loss(support, support_labels, query, query_labels, encoder)
        grads = tape.gradient(loss, encoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, encoder.trainable_weights))
        if episode % 50 == 0:
            logger.info(f"Episode {episode}: loss={loss.numpy():.4f}, acc={acc.numpy():.4f}")
    encoder.save(os.path.join(CHECKPOINT_DIR, 'prototypical_encoder.keras'))
    logger.info("Encoder saved.")

if __name__ == '__main__':
    main() 