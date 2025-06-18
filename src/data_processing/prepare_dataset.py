import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_dataset(project_root):
    """
    Splits the processed landmark data into training, validation, and test sets.
    Also creates a text corpus of sign names.
    """
    processed_path = os.path.join(project_root, 'data', 'processed')
    train_path = os.path.join(project_root, 'data', 'train')
    val_path = os.path.join(project_root, 'data', 'val')
    test_path = os.path.join(project_root, 'data', 'test')
    nlp_path = os.path.join(project_root, 'data', 'nlp')

    # Clean up previous splits
    for path in [train_path, val_path, test_path, nlp_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    all_files = []
    labels = []
    sign_names = set()

    print("Collecting processed data...")
    for dataset_folder in os.listdir(processed_path):
        dataset_full_path = os.path.join(processed_path, dataset_folder)
        if not os.path.isdir(dataset_full_path): continue

        for sign_folder in os.listdir(dataset_full_path):
            sign_full_path = os.path.join(dataset_full_path, sign_folder)
            if not os.path.isdir(sign_full_path): continue
            
            sign_names.add(sign_folder)
            for file_name in os.listdir(sign_full_path):
                if file_name.endswith('.npy'):
                    all_files.append(os.path.join(sign_full_path, file_name))
                    labels.append(sign_folder)

    if not all_files:
        print("No processed data found to split. Exiting.")
        return

    print(f"Found {len(all_files)} samples across {len(sign_names)} signs.")

    # Split data (80% train, 20% temp -> 10% val, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Function to copy files to their destination
    def copy_files(files, labels, dest_path):
        for file_path, label in zip(files, labels):
            dest_folder = os.path.join(dest_path, label)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(file_path, dest_folder)

    print("Copying files to train, val, and test directories...")
    copy_files(X_train, y_train, train_path)
    copy_files(X_val, y_val, val_path)
    copy_files(X_test, y_test, test_path)

    # Create NLP corpus
    corpus_path = os.path.join(nlp_path, 'corpus.txt')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for sign in sorted(list(sign_names)):
            f.write(f"{sign}\n")
    print(f"Created NLP corpus at {corpus_path}")

    print("Dataset preparation complete.")

if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    prepare_dataset(project_root)
