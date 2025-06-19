import os
import numpy as np
import gc
import psutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging

# Configuration pour éviter les problèmes de mémoire
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Affiche l'utilisation mémoire actuelle."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def load_data_in_batches(directory, batch_size=100, max_samples_per_class=None):
    """Charge les données par batches pour éviter les problèmes de mémoire."""
    X, y = [], []
    labels = []
    
    # Compter le nombre total de fichiers
    total_files = 0
    for sign_folder in sorted(os.listdir(directory)):
        sign_path = os.path.join(directory, sign_folder)
        if not os.path.isdir(sign_path):
            continue
        npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
        if max_samples_per_class:
            npy_files = npy_files[:max_samples_per_class]
        total_files += len(npy_files)
    
    logger.info(f"Total files to load: {total_files}")
    
    # Charger par batches
    loaded_files = 0
    for sign_folder in sorted(os.listdir(directory)):
        sign_path = os.path.join(directory, sign_folder)
        if not os.path.isdir(sign_path):
            continue
        
        if sign_folder not in labels:
            labels.append(sign_folder)
        
        npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
        if max_samples_per_class:
            npy_files = npy_files[:max_samples_per_class]
        
        for npy_file in sorted(npy_files):
            try:
                data = np.load(os.path.join(sign_path, npy_file))
                X.append(data)
                y.append(sign_folder)
                loaded_files += 1
                
                # Afficher le progrès
                if loaded_files % 50 == 0:
                    memory_mb = get_memory_usage()
                    logger.info(f"Loaded {loaded_files}/{total_files} files. Memory: {memory_mb:.1f} MB")
                
                # Vérifier la mémoire et forcer le garbage collection si nécessaire
                if loaded_files % batch_size == 0:
                    memory_mb = get_memory_usage()
                    if memory_mb > 2000:  # Si plus de 2GB
                        logger.warning(f"High memory usage: {memory_mb:.1f} MB. Forcing garbage collection.")
                        gc.collect()
                        
            except Exception as e:
                logger.error(f"Error loading {npy_file}: {e}")
                continue
    
    logger.info(f"Successfully loaded {len(X)} samples")
    return X, y, labels

def create_data_generator(X, y, batch_size=32):
    """Générateur de données pour éviter de charger tout en mémoire."""
    num_samples = len(X)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = [X[j] for j in batch_indices]
            batch_y = [y[j] for j in batch_indices]
            yield batch_X, batch_y

def pad_sequences_batch(sequences, max_len=None):
    """Pad une batch de séquences."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad avec des zéros
            padded = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant')
        else:
            # Tronquer si trop long
            padded = seq[:max_len]
        padded_sequences.append(padded)
    
    return np.array(padded_sequences, dtype=np.float32)

def main():
    """Main function to train the LSTM model with memory optimization."""
    try:
        script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
        data_path = os.path.join(project_root, 'data')
        models_path = os.path.join(project_root, 'models', 'trained')
        log_path = os.path.join(project_root, 'logs')

        logger.info(f"Project root: {project_root}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Models path: {models_path}")
        logger.info(f"Log path: {log_path}")

        os.makedirs(models_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # Vérifier la mémoire disponible
        memory_mb = get_memory_usage()
        logger.info(f"Initial memory usage: {memory_mb:.1f} MB")

        # Charger les données d'entraînement avec limitation
        logger.info("Loading training data with memory optimization...")
        train_dir = os.path.join(data_path, 'train')
        
        if not os.path.exists(train_dir):
            logger.error(f"Train directory not found: {train_dir}")
            return
        
        # Limiter le nombre d'échantillons par classe pour éviter les problèmes de mémoire
        max_samples_per_class = 50  # Réduire si nécessaire
        X_train_raw, y_train_raw, _ = load_data_in_batches(
            train_dir, 
            batch_size=50, 
            max_samples_per_class=max_samples_per_class
        )
        
        logger.info(f"Loaded {len(X_train_raw)} training samples")
        
        if len(X_train_raw) == 0:
            logger.error("No training data found!")
            return

        # Créer un ensemble de validation
        logger.info("Creating validation set...")
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_raw, y_train_raw, test_size=0.15, random_state=42, stratify=y_train_raw
        )
        logger.info(f"Train samples: {len(X_train_raw)}, Validation samples: {len(X_val_raw)}")

        # Encoder les labels
        corpus_path = os.path.join(project_root, 'data', 'nlp', 'corpus.txt')
        
        if not os.path.exists(corpus_path):
            logger.error(f"Corpus file not found: {corpus_path}")
            return
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            all_labels = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(all_labels)} labels from corpus")

        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        num_classes = len(label_encoder.classes_)
        logger.info(f"Number of classes: {num_classes}")

        # Calculer la longueur maximale des séquences
        logger.info("Calculating max sequence length...")
        max_len_train = max(len(seq) for seq in X_train_raw) if X_train_raw else 0
        max_len_val = max(len(seq) for seq in X_val_raw) if X_val_raw else 0
        max_len = max(max_len_train, max_len_val)
        logger.info(f"Max sequence length: {max_len}")

        # Encoder les labels
        y_train_encoded = label_encoder.transform(y_train_raw)
        y_val_encoded = label_encoder.transform(y_val_raw)

        # Préparer les données d'entraînement
        logger.info("Preparing training data...")
        X_train = pad_sequences_batch(X_train_raw, max_len)
        y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)

        # Préparer les données de validation
        logger.info("Preparing validation data...")
        X_val = pad_sequences_batch(X_val_raw, max_len)
        y_val_categorical = to_categorical(y_val_encoded, num_classes=num_classes)

        # Nettoyer la mémoire
        del X_train_raw, X_val_raw, y_train_raw, y_val_raw, y_train_encoded, y_val_encoded
        gc.collect()

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_train_categorical shape: {y_train_categorical.shape}")
        logger.info(f"y_val_categorical shape: {y_val_categorical.shape}")

        # Définir le modèle avec une architecture plus simple
        logger.info("Building model...")
        input_shape = (max_len, X_train.shape[2])
        logger.info(f"Input shape: {input_shape}")
        
        model = Sequential([
            Masking(mask_value=0., input_shape=input_shape),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='Adam', 
            loss='categorical_crossentropy', 
            metrics=['categorical_accuracy']
        )
        model.summary()

        # Callbacks
        tensorboard_callback = TensorBoard(log_dir=log_path)
        checkpoint_path = os.path.join(models_path, 'lsf_recognition_model.keras')
        model_checkpoint = ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_categorical_accuracy', 
            mode='max'
        )
        early_stop = EarlyStopping(
            monitor='val_categorical_accuracy', 
            mode='max', 
            patience=10, 
            restore_best_weights=True
        )

        # Entraîner le modèle
        logger.info("Training model...")
        memory_mb = get_memory_usage()
        logger.info(f"Memory before training: {memory_mb:.1f} MB")

        history = model.fit(
            X_train, y_train_categorical,
            epochs=50,  # Réduit de 100 à 50
            batch_size=16,  # Réduit de 32 à 16
            validation_data=(X_val, y_val_categorical),
            callbacks=[tensorboard_callback, model_checkpoint, early_stop],
            verbose=1
        )

        logger.info(f"Model training complete. Best model saved to {checkpoint_path}")
        
        # Afficher les résultats finaux
        final_accuracy = max(history.history['val_categorical_accuracy'])
        logger.info(f"Best validation accuracy: {final_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"ERROR in train_model.py: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
