import os
import subprocess
import shutil

def check_git_lfs():
    """Checks if git-lfs is installed."""
    try:
        subprocess.run(['git', 'lfs', '--version'], check=True, capture_output=True, text=True)
        print("git-lfs is installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("git-lfs is not installed. Please install it to proceed.")
        print("On Debian/Ubuntu, you can use: sudo apt-get install git-lfs")
        return False

def download_and_organize_parlr_dataset(project_root):
    """
    Downloads the ParlR LSF dataset, organizes it, and cleans up.
    """
    parlr_data_path = os.path.join(project_root, 'data', 'raw', 'parlr')
    temp_clone_path = os.path.join(project_root, 'lsf-data-temp')
    repo_url = 'https://github.com/parlr/lsf-data.git'

    if os.path.exists(parlr_data_path) and os.listdir(parlr_data_path):
        print("ParlR dataset already seems to be downloaded and organized.")
        return

    print("Starting download of ParlR dataset...")

    if not check_git_lfs():
        return

    print(f"Cloning {repo_url} into {temp_clone_path}...")
    clone_command = ['git', 'clone', repo_url, temp_clone_path]
    subprocess.run(clone_command, check=True)

    print("Pulling large files with git-lfs...")
    lfs_pull_command = ['git', 'lfs', 'pull']
    subprocess.run(lfs_pull_command, check=True, cwd=temp_clone_path)

    print("Organizing the dataset...")
    source_videos_path = os.path.join(temp_clone_path, 'videos')
    if os.path.exists(source_videos_path):
        # Ensure the destination directory exists
        os.makedirs(parlr_data_path, exist_ok=True)
        for item in os.listdir(source_videos_path):
            s = os.path.join(source_videos_path, item)
            d = os.path.join(parlr_data_path, item)
            shutil.move(s, d)
        print(f"Moved video data to {parlr_data_path}")
    else:
        print(f"Warning: 'videos' directory not found in the cloned repository.")

    print(f"Cleaning up temporary directory: {temp_clone_path}")
    shutil.rmtree(temp_clone_path)

    print("ParlR dataset download and organization complete.")

if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    download_and_organize_parlr_dataset(project_root)
