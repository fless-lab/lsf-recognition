import os
import numpy as np
import gc
import psutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging
import json

# Configuration pour éviter les problèmes de mémoire
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Affiche l'utilisation mémoire actuelle."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def load_data_efficiently(directory, split_type='train', max_memory_gb=8):
    """Charge les données de manière efficace avec gestion mémoire intelligente."""
    X, y = [], []
    labels = []
    
    max_memory_mb = max_memory_gb * 1024
    logger.info(f"Loading {split_type} data with memory limit: {max_memory_gb} GB")
    
    # Lister tous les fichiers d'abord
    all_files = []
    for sign_folder in sorted(os.listdir(directory)):
        sign_path = os.path.join(directory, sign_folder)
        if os.path.isdir(sign_path):
            npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            for npy_file in npy_files:
                all_files.append((sign_folder, npy_file))
    
    logger.info(f"Found {len(all_files)} files to load")
    
    # Charger par batches pour contrôler la mémoire
    batch_size = 100
    loaded_files = 0
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        
        for sign_folder, npy_file in batch_files:
            try:
                file_path = os.path.join(directory, sign_folder, npy_file)
                data = np.load(file_path)
                
                X.append(data)
                y.append(sign_folder)
                
                if sign_folder not in labels:
                    labels.append(sign_folder)
                
                loaded_files += 1
                
                # Vérifier la mémoire toutes les 50 charges
                if loaded_files % 50 == 0:
                    memory_mb = get_memory_usage()
                    logger.info(f"Loaded {loaded_files}/{len(all_files)} files, memory: {memory_mb:.1f} MB")
                    
                    # Si trop de mémoire, forcer le garbage collection
                    if memory_mb > max_memory_mb * 0.8:  # 80% de la limite
                        logger.warning(f"High memory usage ({memory_mb:.1f} MB). Forcing garbage collection.")
                        gc.collect()
                        
            except Exception as e:
                logger.error(f"Error loading {npy_file}: {e}")
                continue
        
        # Garbage collection après chaque batch
        gc.collect()
        
        # Vérifier la mémoire après le batch
        memory_mb = get_memory_usage()
        if memory_mb > max_memory_mb:
            logger.warning(f"Memory limit reached ({memory_mb:.1f} MB). Stopping data loading.")
            break
    
    logger.info(f"Successfully loaded {len(X)} samples from {len(set(y))} classes")
    return X, y, labels

def pad_sequences_efficient(sequences, max_len=None):
    """Pad les séquences de manière efficace."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences) if sequences else 200
    
    # Limiter la longueur maximale pour éviter les problèmes mémoire
    max_len = min(max_len, 1000)  # Max 1000 frames
    
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

def create_enhanced_model(input_shape, num_classes):
    """Crée un modèle amélioré avec une architecture plus sophistiquée."""
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.),
        
        # Première couche LSTM bidirectionnelle
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        # Deuxième couche LSTM bidirectionnelle
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        # Troisième couche LSTM
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Couches denses
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_simple_split(X, y, val_ratio=0.15):
    """Crée un split simple sans stratification pour éviter les erreurs."""
    logger.info(f"Creating simple split with val_ratio={val_ratio}")
    
    # Calculer le nombre d'échantillons pour la validation
    n_val = int(len(X) * val_ratio)
    n_train = len(X) - n_val
    
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Train samples: {n_train}")
    logger.info(f"Val samples: {n_val}")
    
    # Split simple (sans stratification)
    indices = np.random.permutation(len(X))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    
    return X_train, X_val, y_train, y_val

def main():
    """Main function to train the enhanced LSTM model."""
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

        # Vérifier la mémoire initiale
        memory_mb = get_memory_usage()
        logger.info(f"Initial memory usage: {memory_mb:.1f} MB")

        # Charger les données d'entraînement
        logger.info("Loading training data...")
        train_dir = os.path.join(data_path, 'train')
        
        if not os.path.exists(train_dir):
            logger.error(f"Train directory not found: {train_dir}")
            return
        
        X_train_raw, y_train_raw, _ = load_data_efficiently(train_dir, 'train', max_memory_gb=8)
        
        logger.info(f"Loaded {len(X_train_raw)} training samples")
        
        if len(X_train_raw) == 0:
            logger.error("No training data found!")
            return

        # Créer un split simple (sans stratification)
        logger.info("Creating validation set...")
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = create_simple_split(
            X_train_raw, y_train_raw, val_ratio=0.15
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
        max_len_train = max(len(seq) for seq in X_train_raw) if X_train_raw else 200
        max_len_val = max(len(seq) for seq in X_val_raw) if X_val_raw else 200
        max_len = min(max(max_len_train, max_len_val), 1000)  # Limiter à 1000
        logger.info(f"Max sequence length: {max_len}")

        # Encoder les labels
        y_train_encoded = label_encoder.transform(y_train_raw)
        y_val_encoded = label_encoder.transform(y_val_raw)

        # Préparer les données d'entraînement
        logger.info("Preparing training data...")
        X_train = pad_sequences_efficient(X_train_raw, max_len)
        y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)

        # Préparer les données de validation
        logger.info("Preparing validation data...")
        X_val = pad_sequences_efficient(X_val_raw, max_len)
        y_val_categorical = to_categorical(y_val_encoded, num_classes=num_classes)

        # Nettoyer la mémoire
        del X_train_raw, X_val_raw, y_train_raw, y_val_raw, y_train_encoded, y_val_encoded
        gc.collect()

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_train_categorical shape: {y_train_categorical.shape}")
        logger.info(f"y_val_categorical shape: {y_val_categorical.shape}")

        # Définir le modèle amélioré
        logger.info("Building enhanced model...")
        input_shape = (max_len, X_train.shape[2])
        logger.info(f"Input shape: {input_shape}")
        
        model = create_enhanced_model(input_shape, num_classes)

        model.compile(
            optimizer='Adam', 
            loss='categorical_crossentropy', 
            metrics=['categorical_accuracy']
        )
        model.summary()

        # Callbacks améliorés
        tensorboard_callback = TensorBoard(log_dir=log_path)
        checkpoint_path = os.path.join(models_path, 'lsf_recognition_model_fixed.keras')
        model_checkpoint = ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_categorical_accuracy', 
            mode='max'
        )
        early_stop = EarlyStopping(
            monitor='val_categorical_accuracy', 
            mode='max', 
            patience=20,  # Plus patient
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7
        )

        # Entraîner le modèle
        logger.info("Training enhanced model...")
        memory_mb = get_memory_usage()
        logger.info(f"Memory before training: {memory_mb:.1f} MB")

        history = model.fit(
            X_train, y_train_categorical,
            epochs=200,  # Plus d'époques
            batch_size=16,  # Batch size plus petit
            validation_data=(X_val, y_val_categorical),
            callbacks=[tensorboard_callback, model_checkpoint, early_stop, reduce_lr],
            verbose=1
        )

        logger.info(f"Model training complete. Best model saved to {checkpoint_path}")
        
        # Afficher les résultats finaux
        final_accuracy = max(history.history['val_categorical_accuracy'])
        logger.info(f"Best validation accuracy: {final_accuracy:.4f}")
        
        # Sauvegarder l'historique
        history_path = os.path.join(models_path, 'training_history_fixed.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        logger.info(f"Training history saved to {history_path}")
        
        # Afficher l'utilisation mémoire finale
        memory_mb = get_memory_usage()
        logger.info(f"Final memory usage: {memory_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f"ERROR in train_model_fixed.py: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main() 