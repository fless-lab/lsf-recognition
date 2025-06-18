import os
import shutil
import re

def organize_custom_dataset(project_root):
    """
    Organizes custom videos from a 'custom_videos' directory into 'data/raw/custom',
    sorted into subfolders by sign name.
    Handles the case where the 'custom_videos' directory does not exist.
    """
    custom_videos_path = os.path.join(project_root, 'custom_videos')
    custom_data_path = os.path.join(project_root, 'data', 'raw', 'custom')

    if not os.path.exists(custom_videos_path) or not os.listdir(custom_videos_path):
        print("No custom videos found in 'custom_videos' directory. Skipping organization.")
        return

    print("Organizing custom dataset...")
    os.makedirs(custom_data_path, exist_ok=True)

    for filename in os.listdir(custom_videos_path):
        # Extract sign name from filename (e.g., 'bonjour_001.mp4' -> 'bonjour')
        match = re.match(r'([a-zA-Z_]+)_\d+\..*', filename)
        if match:
            sign_name = match.group(1)
            sign_dir = os.path.join(custom_data_path, sign_name)
            os.makedirs(sign_dir, exist_ok=True)

            source_file = os.path.join(custom_videos_path, filename)
            destination_file = os.path.join(sign_dir, filename)

            shutil.move(source_file, destination_file)
            print(f"Moved {filename} to {sign_dir}")
        else:
            print(f"Could not extract sign name from {filename}. Skipping.")

    print("Custom dataset organization complete.")

if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    organize_custom_dataset(project_root)
