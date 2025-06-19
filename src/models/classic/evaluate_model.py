import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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
    """Main function to evaluate the LSTM model."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    models_path = os.path.join(project_root, 'models', 'trained')
    model_path = os.path.join(models_path, 'lsf_recognition_model.keras')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    # Load data
    print("Loading data...")
    X_train_raw, _, train_labels = load_data_from_directory(os.path.join(data_path, 'train'))
    X_val_raw, _, _ = load_data_from_directory(os.path.join(data_path, 'val'))
    X_test_raw, y_test_raw, _ = load_data_from_directory(os.path.join(data_path, 'test'))

    # Encode labels
    corpus_path = os.path.join(project_root, 'data', 'nlp', 'corpus.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        all_labels = [line.strip() for line in f.readlines()]

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels) # Fit on all possible labels
    y_test_encoded = label_encoder.transform(y_test_raw)

    # Pad sequences to the same length as during training
    print("Padding sequences...")
    all_sequences = X_train_raw + X_val_raw + X_test_raw
    max_len = max(len(seq) for seq in all_sequences) if all_sequences else 0
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test_raw, maxlen=max_len, padding='post', dtype='float32')

    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    # Make predictions
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    print("\nClassification Report:")
    num_classes = len(label_encoder.classes_)
    report = classification_report(y_test_encoded, y_pred_classes, labels=np.arange(num_classes), target_names=label_encoder.classes_, zero_division=0)
    print(report)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    confusion_matrix_path = os.path.join(project_root, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

if __name__ == '__main__':
    main()
