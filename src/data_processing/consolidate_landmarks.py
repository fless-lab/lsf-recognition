import os
import shutil

def main():
    """Consolidates all landmark files into a single directory structure."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    processed_path = os.path.join(project_root, 'data', 'processed')
    consolidated_path = os.path.join(project_root, 'data', 'consolidated')

    # Clean up previous consolidated data
    print(f"Cleaning up previous consolidated data in {consolidated_path}...")
    if os.path.exists(consolidated_path):
        shutil.rmtree(consolidated_path)
    os.makedirs(consolidated_path)

    print(f"Scanning {processed_path} for landmark files...")
    file_count = 0
    # Walk through the complex directory structure
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
                    # The label is the filename without extension
                    label = os.path.splitext(npy_file)[0]
                    
                    # Create a dedicated folder for this label in the consolidated directory
                    destination_dir = os.path.join(consolidated_path, label)
                    os.makedirs(destination_dir, exist_ok=True)
                    
                    source_file_path = os.path.join(sub_source_path, npy_file)
                    destination_file_path = os.path.join(destination_dir, npy_file)

                    # Handle potential filename collisions to avoid overwriting data
                    if os.path.exists(destination_file_path):
                        base, ext = os.path.splitext(npy_file)
                        i = 1
                        while True:
                            new_name = f"{base}_{i}{ext}"
                            new_destination_path = os.path.join(destination_dir, new_name)
                            if not os.path.exists(new_destination_path):
                                destination_file_path = new_destination_path
                                break
                            i += 1
                    
                    shutil.copy(source_file_path, destination_file_path)
                    file_count += 1

    print(f"Consolidation complete. Copied {file_count} files into {consolidated_path}.")

if __name__ == '__main__':
    main()
