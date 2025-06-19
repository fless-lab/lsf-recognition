import os
import numpy as np
import gc
import psutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging

# Configuration pour éviter les problèmes de mémoire
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

def load_data_safely(directory, max_samples_per_class=20, max_classes=50):
    """Charge les données de manière très sécurisée avec des limites strictes."""
    X, y = [], []
    labels = []
    
    logger.info(f"Loading data with limits: max {max_samples_per_class} samples per class, max {max_classes} classes")
    
    # Lister les classes disponibles
    available_classes = []
    for sign_folder in sorted(os.listdir(directory)):
        sign_path = os.path.join(directory, sign_folder)
        if os.path.isdir(sign_path):
            npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            if len(npy_files) >= 5:  # Au moins 5 échantillons par classe
                available_classes.append(sign_folder)
    
    # Limiter le nombre de classes
    if len(available_classes) > max_classes:
        available_classes = available_classes[:max_classes]
        logger.info(f"Limited to {max_classes} classes for memory safety")
    
    logger.info(f"Processing {len(available_classes)} classes")
    
    # Charger les données classe par classe
    for i, sign_folder in enumerate(available_classes):
        sign_path = os.path.join(directory, sign_folder)
        npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
        
        # Limiter le nombre d'échantillons par classe
        if len(npy_files) > max_samples_per_class:
            npy_files = npy_files[:max_samples_per_class]
        
        labels.append(sign_folder)
        
        for npy_file in npy_files:
            try:
                data = np.load(os.path.join(sign_path, npy_file))
                X.append(data)
                y.append(sign_folder)
                
                # Vérifier la mémoire toutes les 10 charges
                if len(X) % 10 == 0:
                    memory_mb = get_memory_usage()
                    logger.info(f"Loaded {len(X)} samples, memory: {memory_mb:.1f} MB")
                    
                    # Si trop de mémoire utilisée, arrêter
                    if memory_mb > 1500:  # 1.5GB max
                        logger.warning(f"Memory limit reached ({memory_mb:.1f} MB). Stopping data loading.")
                        break
                        
            except Exception as e:
                logger.error(f"Error loading {npy_file}: {e}")
                continue
        
        # Vérifier la mémoire après chaque classe
        memory_mb = get_memory_usage()
        logger.info(f"Class {i+1}/{len(available_classes)} ({sign_folder}): {len([s for s in y if s == sign_folder])} samples, memory: {memory_mb:.1f} MB")
        
        # Forcer le garbage collection
        gc.collect()
        
        # Si trop de mémoire, arrêter
        if memory_mb > 1500:
            logger.warning("Memory limit reached. Stopping data loading.")
            break
    
    logger.info(f"Successfully loaded {len(X)} samples from {len(set(y))} classes")
    return X, y, labels

def pad_sequences_safe(sequences, max_len=None):
    """Pad les séquences de manière sécurisée."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences) if sequences else 100
    
    # Limiter la longueur maximale pour éviter les problèmes mémoire
    max_len = min(max_len, 500)  # Max 500 frames
    
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
    """Main function to train the LSTM model with maximum safety."""
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

        # Charger les données d'entraînement de manière sécurisée
        logger.info("Loading training data safely...")
        train_dir = os.path.join(data_path, 'train')
        
        if not os.path.exists(train_dir):
            logger.error(f"Train directory not found: {train_dir}")
            return
        
        # Charger avec des limites strictes
        X_train_raw, y_train_raw, _ = load_data_safely(
            train_dir, 
            max_samples_per_class=20,  # Très limité
            max_classes=30  # Très limité
        )
        
        logger.info(f"Loaded {len(X_train_raw)} training samples")
        
        if len(X_train_raw) == 0:
            logger.error("No training data found!")
            return

        # Créer un ensemble de validation
        logger.info("Creating validation set...")
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
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
        max_len_train = max(len(seq) for seq in X_train_raw) if X_train_raw else 100
        max_len_val = max(len(seq) for seq in X_val_raw) if X_val_raw else 100
        max_len = min(max(max_len_train, max_len_val), 500)  # Limiter à 500
        logger.info(f"Max sequence length: {max_len}")

        # Encoder les labels
        y_train_encoded = label_encoder.transform(y_train_raw)
        y_val_encoded = label_encoder.transform(y_val_raw)

        # Préparer les données d'entraînement
        logger.info("Preparing training data...")
        X_train = pad_sequences_safe(X_train_raw, max_len)
        y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)

        # Préparer les données de validation
        logger.info("Preparing validation data...")
        X_val = pad_sequences_safe(X_val_raw, max_len)
        y_val_categorical = to_categorical(y_val_encoded, num_classes=num_classes)

        # Nettoyer la mémoire
        del X_train_raw, X_val_raw, y_train_raw, y_val_raw, y_train_encoded, y_val_encoded
        gc.collect()

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_train_categorical shape: {y_train_categorical.shape}")
        logger.info(f"y_val_categorical shape: {y_val_categorical.shape}")

        # Définir le modèle avec une architecture très simple
        logger.info("Building model...")
        input_shape = (max_len, X_train.shape[2])
        logger.info(f"Input shape: {input_shape}")
        
        model = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.),
            LSTM(16, return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
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
        checkpoint_path = os.path.join(models_path, 'lsf_recognition_model_safe.keras')
        model_checkpoint = ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_categorical_accuracy', 
            mode='max'
        )
        early_stop = EarlyStopping(
            monitor='val_categorical_accuracy', 
            mode='max', 
            patience=5,  # Très court
            restore_best_weights=True
        )

        # Entraîner le modèle
        logger.info("Training model safely...")
        memory_mb = get_memory_usage()
        logger.info(f"Memory before training: {memory_mb:.1f} MB")

        history = model.fit(
            X_train, y_train_categorical,
            epochs=20,  # Très court
            batch_size=8,  # Très petit
            validation_data=(X_val, y_val_categorical),
            callbacks=[tensorboard_callback, model_checkpoint, early_stop],
            verbose=1
        )

        logger.info(f"Model training complete. Best model saved to {checkpoint_path}")
        
        # Afficher les résultats finaux
        final_accuracy = max(history.history['val_categorical_accuracy'])
        logger.info(f"Best validation accuracy: {final_accuracy:.4f}")
        
        # Afficher l'utilisation mémoire finale
        memory_mb = get_memory_usage()
        logger.info(f"Final memory usage: {memory_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f"ERROR in train_model_safe.py: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main() 