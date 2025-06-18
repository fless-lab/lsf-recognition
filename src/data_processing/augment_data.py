import os
import numpy as np
import shutil

# --- Augmentation Functions ---

def add_noise(data, noise_level=0.005):
    """Adds Gaussian noise to the landmark data."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_landmarks(data, scale_range=0.1):
    """Scales the landmark data."""
    scale_factor = 1.0 + np.random.uniform(-scale_range, scale_range)
    return data * scale_factor

def translate_landmarks(data, translate_range=0.05):
    """Translates the landmark data."""
    translation_vector = np.random.uniform(-translate_range, translate_range, size=data.shape[1])
    return data + translation_vector

def rotate_landmarks(data):
    """Rotates landmarks on the XY plane."""
    angle = np.random.uniform(-np.pi / 18, np.pi / 18)  # Rotate by up to +/- 10 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_data = data.copy()
    # Assuming landmarks are in (x, y, z, visibility, ...) format or similar
    # We will rotate the first two coordinates (x, y) of each landmark
    num_landmarks = data.shape[1] // 2 # Simple assumption, adjust if format is different
    for i in range(num_landmarks):
        xy = data[:, i*2:(i*2)+2]
        rotated_data[:, i*2:(i*2)+2] = np.dot(xy, rotation_matrix.T)
    return rotated_data

# --- Main Augmentation Script ---

def main():
    """Generates augmented data from original samples."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')

    source_dir = os.path.join(data_path, 'train_originals')
    target_dir = os.path.join(data_path, 'train')

    if os.path.exists(target_dir):
        print(f"Cleaning up target directory: {target_dir}")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    print(f"Starting data augmentation from '{source_dir}' to '{target_dir}'...")
    augmentations_per_file = 20

    for sign_folder in sorted(os.listdir(source_dir)):
        source_sign_path = os.path.join(source_dir, sign_folder)
        target_sign_path = os.path.join(target_dir, sign_folder)
        os.makedirs(target_sign_path, exist_ok=True)

        if not os.path.isdir(source_sign_path):
            continue

        for filename in sorted(os.listdir(source_sign_path)):
            if not filename.endswith('.npy'):
                continue

            # 1. Copy the original file
            shutil.copy(os.path.join(source_sign_path, filename), target_sign_path)

            # 2. Create augmented versions
            original_data = np.load(os.path.join(source_sign_path, filename))
            base_name = filename.replace('.npy', '')

            for i in range(augmentations_per_file):
                augmented_data = original_data.copy()
                
                # Apply a random combination of augmentations
                if np.random.rand() > 0.5: augmented_data = add_noise(augmented_data)
                if np.random.rand() > 0.5: augmented_data = scale_landmarks(augmented_data)
                if np.random.rand() > 0.5: augmented_data = translate_landmarks(augmented_data)
                # Rotation can be more disruptive, apply it less often
                if np.random.rand() > 0.7: augmented_data = rotate_landmarks(augmented_data)

                new_filename = f"{base_name}_aug_{i+1}.npy"
                np.save(os.path.join(target_sign_path, new_filename), augmented_data)
        
        print(f"  - Augmented sign: {sign_folder}")

    print("\nData augmentation complete.")

if __name__ == '__main__':
    main()
