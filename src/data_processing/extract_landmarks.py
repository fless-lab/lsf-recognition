import cv2
import mediapipe as mp
import numpy as np
import os

# Setup MediaPipe instances
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extracts landmark data into a single numpy array."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_video(video_path, holistic):
    """Processes a single video file to extract landmarks for each frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    keypoints_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = holistic.process(image)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        keypoints_list.append(keypoints)

    cap.release()
    return np.array(keypoints_list)

def main():
    """Main function to iterate through datasets and extract landmarks."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    raw_path = os.path.join(data_path, 'raw')
    processed_path = os.path.join(data_path, 'processed')

    # Datasets to process
    datasets = ['parlr', 'custom']

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for dataset in datasets:
            dataset_path = os.path.join(raw_path, dataset)
            if not os.path.exists(dataset_path):
                print(f"Dataset path {dataset_path} does not exist. Skipping.")
                continue

            print(f"Processing dataset: {dataset}")
            for sign_folder in os.listdir(dataset_path):
                sign_path = os.path.join(dataset_path, sign_folder)
                if not os.path.isdir(sign_path):
                    continue

                print(f"  Processing sign: {sign_folder}")
                for video_file in os.listdir(sign_path):
                    video_path = os.path.join(sign_path, video_file)
                    
                    # Define output path
                    output_folder = os.path.join(processed_path, dataset, sign_folder)
                    os.makedirs(output_folder, exist_ok=True)
                    output_file_path = os.path.join(output_folder, os.path.splitext(video_file)[0] + '.npy')

                    if os.path.exists(output_file_path):
                        print(f"    Skipping {video_file}, already processed.")
                        continue

                    print(f"    Processing video: {video_file}")
                    keypoints_data = process_video(video_path, holistic)
                    if keypoints_data is not None:
                        np.save(output_file_path, keypoints_data)
                        print(f"    Saved landmarks to {output_file_path}")

    print("Landmark extraction complete.")

if __name__ == '__main__':
    main()
