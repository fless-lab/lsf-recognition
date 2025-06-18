import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
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

    # Load data
    print("Loading data...")
    X_train_raw, y_train_raw, train_labels = load_data_from_directory(os.path.join(data_path, 'train'))
    X_val_raw, y_val_raw, _ = load_data_from_directory(os.path.join(data_path, 'val'))

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    y_train_encoded = label_encoder.transform(y_train_raw)
    y_val_encoded = label_encoder.transform(y_val_raw)
    y_train_categorical = to_categorical(y_train_encoded).astype(int)
    y_val_categorical = to_categorical(y_val_encoded).astype(int)

    # Pad sequences
    print("Padding sequences...")
    max_len = max(len(seq) for seq in X_train_raw + X_val_raw)
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train_raw, maxlen=max_len, padding='post', dtype='float32')
    X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val_raw, maxlen=max_len, padding='post', dtype='float32')
    
    # Define model
    print("Building model...")
    model = Sequential([
        Masking(mask_value=0., input_shape=(max_len, X_train.shape[2])),
        LSTM(64, return_sequences=True, activation='relu'),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(train_labels), activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=log_path)
    checkpoint_path = os.path.join(models_path, 'lsf_recognition_model.keras')
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_categorical_accuracy', mode='max')

    # Train model
    print("Training model...")
    model.fit(X_train, y_train_categorical, epochs=100, 
              validation_data=(X_val, y_val_categorical), 
              callbacks=[tensorboard_callback, model_checkpoint])

    print(f"Model training complete. Best model saved to {checkpoint_path}")

if __name__ == '__main__':
    main()
