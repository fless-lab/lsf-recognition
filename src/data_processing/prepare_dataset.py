import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

def main():
    """Main function to prepare the dataset."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    processed_path = os.path.join(data_path, 'processed')
    nlp_path = os.path.join(data_path, 'nlp')

    # Define paths for train, val, test sets
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    test_path = os.path.join(data_path, 'test')

    # Clean up previous splits
    print("Cleaning up previous data splits...")
    for path in [train_path, val_path, test_path, nlp_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # Collect all landmark files and their actual sign labels
    print("Collecting landmark files and labels...")
    all_files = []
    labels = []
    for source_folder in sorted(os.listdir(processed_path)):
        source_path = os.path.join(processed_path, source_folder)
        if not os.path.isdir(source_path):
            continue
        for sub_source_folder in sorted(os.listdir(source_path)):
            sub_source_path = os.path.join(source_path, sub_source_folder)
            if not os.path.isdir(sub_source_path):
                continue
            for npy_file in sorted(os.listdir(sub_source_path)):
                if npy_file.endswith('.npy'):
                    all_files.append(os.path.join(sub_source_path, npy_file))
                    label = os.path.splitext(npy_file)[0]
                    labels.append(label)

    if not all_files:
        print("No processed data found. Exiting.")
        return

    # Create NLP corpus from labels
    unique_labels = sorted(list(set(labels)))
    corpus_path = os.path.join(nlp_path, 'corpus.txt')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for label in unique_labels:
            f.write(f"{label}\n")
    print(f"Corpus created at {corpus_path}")

    # Group files by label
    print("Grouping files by label...")
    files_by_label = defaultdict(list)
    for file_path, label in zip(all_files, labels):
        files_by_label[label].append(file_path)

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("Splitting data based on instance count...")
    for label, files in files_by_label.items():
        if len(files) == 1:
            # Add single-instance signs directly to training set
            X_train.append(files[0])
            y_train.append(label)
        else:
            # For multi-instance signs, put one in test and the rest in train
            X_test.append(files[0])
            y_test.append(label)
            for i in range(1, len(files)):
                X_train.append(files[i])
                y_train.append(label)

    print(f"Total training samples before validation split: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    
    # Create a validation set from the training set
    print("Creating validation set from training data...")
    if len(X_train) > 1:
        # We use a small test_size for validation, and cannot stratify
        # as many classes will have only one sample in the training set.
        val_size = 0.1
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, shuffle=True
        )
        print(f"Created validation set with {len(X_val)} samples.")
    else:
        print("Not enough training data to create a validation set.")
        X_val, y_val = [], []

    # Function to copy files
    def copy_files(files, labels, destination_folder):
        for file_path, label in zip(files, labels):
            dest_dir = os.path.join(destination_folder, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)

    # Copy files to respective directories
    print("Copying files to train, val, and test directories...")
    copy_files(X_train, y_train, train_path)
    copy_files(X_val, y_val, val_path)
    copy_files(X_test, y_test, test_path)

    print("Dataset preparation complete.")


if __name__ == '__main__':
    main()
