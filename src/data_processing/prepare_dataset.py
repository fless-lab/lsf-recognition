import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

def main():
    """Main function to prepare the dataset."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    processed_path = os.path.join(data_path, 'consolidated')
    nlp_path = os.path.join(data_path, 'nlp')

    # Define paths
    train_originals_path = os.path.join(data_path, 'train_originals')
    test_path = os.path.join(data_path, 'test')
    nlp_path = os.path.join(data_path, 'nlp')

    # Clean up previous experimental data
    print("Cleaning up previous experimental directories...")
    for path in [train_originals_path, test_path, nlp_path, os.path.join(data_path, 'train'), os.path.join(data_path, 'val')]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Collect all landmark files and their actual sign labels
    print("Collecting landmark files and labels...")
    all_files = []
    labels = []
    for sign_folder in sorted(os.listdir(processed_path)):
        sign_path = os.path.join(processed_path, sign_folder)
        if not os.path.isdir(sign_path):
            continue
        # The label is the name of the folder
        label = sign_folder
        for npy_file in sorted(os.listdir(sign_path)):
            if npy_file.endswith('.npy'):
                all_files.append(os.path.join(sign_path, npy_file))
                labels.append(label)

    if not all_files:
        print("No processed data found. Exiting.")
        return

    # Analyze the distribution and select the top 100 signs
    label_counts = Counter(labels)
    top_100_labels = {label for label, count in label_counts.most_common(100)}
    print(f"Selected top {len(top_100_labels)} signs for the experiment.")

    # Create NLP corpus from the selected labels
    corpus_path = os.path.join(nlp_path, 'corpus.txt')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for label in sorted(list(top_100_labels)):
            f.write(f"{label}\n")
    print(f"Corpus created for {len(top_100_labels)} signs at {corpus_path}")

    # Group all files by label
    print("Grouping all files by label...")
    files_by_label = defaultdict(list)
    for file_path, label in zip(all_files, labels):
        files_by_label[label].append(file_path)

    X_train_originals, y_train_originals = [], []
    X_test, y_test = [], []

    print("Splitting data into a clean test set and a set for augmentation...")
    for label, files in files_by_label.items():
        if label not in top_100_labels:
            continue

        if len(files) > 1:
            # Put one file in test, the rest in train_originals
            X_test.append(files[0])
            y_test.append(label)
            X_train_originals.extend(files[1:])
            y_train_originals.extend([label] * (len(files) - 1))
        else:
            # If only one file, use it for training augmentation (no test sample for this one)
            X_train_originals.extend(files)
            y_train_originals.extend([label] * len(files))

    print(f"Created a clean test set with {len(X_test)} samples.")
    print(f"Created a set of {len(X_train_originals)} original samples for training augmentation.")

    # Function to copy files
    def copy_files(files, labels, destination_folder):
        for file_path, label in zip(files, labels):
            dest_dir = os.path.join(destination_folder, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)

    # Copy files to respective directories
    print("Copying files to 'train_originals' and 'test' directories...")
    copy_files(X_train_originals, y_train_originals, train_originals_path)
    copy_files(X_test, y_test, test_path)

    print("Dataset preparation complete.")


if __name__ == '__main__':
    main()
