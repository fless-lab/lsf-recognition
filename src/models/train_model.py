import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data_from_directory(directory):
    """Loads all .npy files from a directory and its subdirectories."""
    X, y = [], []
    labels = []
    for sign_folder in sorted(os.listdir(directory)):
        sign_path = os.path.join(directory, sign_folder)
        if not os.path.isdir(sign_path):
            continue
        labels.append(sign_folder)
        for npy_file in sorted(os.listdir(sign_path)):
            if npy_file.endswith('.npy'):
                data = np.load(os.path.join(sign_path, npy_file))
                X.append(data)
                y.append(sign_folder)
    return X, y, labels




def main():
    """Main function to train the LSTM model."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    models_path = os.path.join(project_root, 'models', 'trained')
    log_path = os.path.join(project_root, 'logs')

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Load augmented training data
    print("Loading augmented training data...")
    X_train_raw, y_train_raw, _ = load_data_from_directory(os.path.join(data_path, 'train'))

    # Create a validation set from the augmented training data
    print("Creating validation set...")
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_raw, y_train_raw, test_size=0.15, random_state=42, stratify=y_train_raw
    )

    # Encode labels
    corpus_path = os.path.join(project_root, 'data', 'nlp', 'corpus.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        all_labels = [line.strip() for line in f.readlines()]

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    # Pad sequences
    print("Padding and encoding data...")
    max_len_train = max(len(seq) for seq in X_train_raw) if X_train_raw else 0
    max_len_val = max(len(seq) for seq in X_val_raw) if X_val_raw else 0
    max_len = max(max_len_train, max_len_val)

    X_train = pad_sequences(X_train_raw, maxlen=max_len, padding='post', truncating='post', dtype='float32')
    y_train_encoded = label_encoder.transform(y_train_raw)
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes).astype(int)

    X_val = pad_sequences(X_val_raw, maxlen=max_len, padding='post', truncating='post', dtype='float32')
    y_val_encoded = label_encoder.transform(y_val_raw)
    y_val_categorical = to_categorical(y_val_encoded, num_classes=num_classes).astype(int)

    # Define model
    print("Building model...")
    input_shape = (max_len, X_train.shape[2])
    
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(64, return_sequences=True), # Removed relu activation
        Dropout(0.3),
        LSTM(64, return_sequences=False), # Removed relu activation
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=log_path)
    checkpoint_path = os.path.join(models_path, 'lsf_recognition_model.keras')
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_categorical_accuracy', mode='max')

    # Train model
    print("Training model on augmented data...")
    early_stop = EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=15, restore_best_weights=True)

    model.fit(X_train, y_train_categorical,
              epochs=100,
              batch_size=32,
              validation_data=(X_val, y_val_categorical),
              callbacks=[tensorboard_callback, model_checkpoint, early_stop])

    print(f"Model training complete. Best model saved to {checkpoint_path}")

if __name__ == '__main__':
    main()
