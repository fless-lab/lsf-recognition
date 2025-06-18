import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup MediaPipe instances
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class LandmarkExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize the landmark extractor with MediaPipe Holistic."""
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  # 0, 1, or 2
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            refine_face_landmarks=True
        )
        
    def extract_keypoints(self, results):
        """Extracts landmark data into a structured format with confidence scores."""
        # Pose landmarks (33 points)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        # Face landmarks (468 points) - only x, y, z (no visibility)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        
        # Left hand landmarks (21 points)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        
        # Right hand landmarks (21 points)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        # Calculate confidence scores
        pose_confidence = np.mean([res.visibility for res in results.pose_landmarks.landmark]) if results.pose_landmarks else 0.0
        face_confidence = 1.0 if results.face_landmarks else 0.0
        lh_confidence = 1.0 if results.left_hand_landmarks else 0.0
        rh_confidence = 1.0 if results.right_hand_landmarks else 0.0
        
        # Combine all landmarks
        landmarks = np.concatenate([pose, face, lh, rh])
        
        # Create metadata
        metadata = {
            'pose_confidence': float(pose_confidence),
            'face_confidence': float(face_confidence),
            'left_hand_confidence': float(lh_confidence),
            'right_hand_confidence': float(rh_confidence),
            'total_frames': len(landmarks) // (33*4 + 468*3 + 21*3 + 21*3),
            'landmark_dimensions': {
                'pose': 33*4,
                'face': 468*3,
                'left_hand': 21*3,
                'right_hand': 21*3
            }
        }
        
        return landmarks, metadata

    def process_video(self, video_path):
        """Processes a single video file to extract landmarks for each frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return None, None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {os.path.basename(video_path)} - {frame_count} frames, {fps} FPS, {width}x{height}")

        keypoints_list = []
        frame_metadata = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = self.holistic.process(image)

            # Extract keypoints and metadata for this frame
            keypoints, frame_meta = self.extract_keypoints(results)
            keypoints_list.append(keypoints)
            
            # Add frame-specific metadata
            frame_meta['frame_index'] = frame_idx
            frame_metadata.append(frame_meta)
            
            frame_idx += 1

        cap.release()
        
        if not keypoints_list:
            logger.warning(f"No landmarks extracted from {video_path}")
            return None, None
            
        # Create video-level metadata
        video_metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'extracted_frames': len(keypoints_list),
            'frame_metadata': frame_metadata,
            'average_pose_confidence': np.mean([meta['pose_confidence'] for meta in frame_metadata]),
            'average_face_confidence': np.mean([meta['face_confidence'] for meta in frame_metadata]),
            'average_left_hand_confidence': np.mean([meta['left_hand_confidence'] for meta in frame_metadata]),
            'average_right_hand_confidence': np.mean([meta['right_hand_confidence'] for meta in frame_metadata])
        }
        
        return np.array(keypoints_list), video_metadata

def main():
    """Main function to extract landmarks from all video sources with provenance tracking."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    raw_path = os.path.join(data_path, 'raw')
    processed_path = os.path.join(data_path, 'processed')

    # Create processed directory structure
    os.makedirs(processed_path, exist_ok=True)

    # Initialize extractor
    extractor = LandmarkExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Process all sources
    sources = ['parlr/jauvert', 'parlr/elix', 'parlr/education-nationale', 'custom']
    
    total_processed = 0
    total_errors = 0
    
    for source in sources:
        source_path = os.path.join(raw_path, source)
        if not os.path.exists(source_path):
            logger.warning(f"Source path {source_path} does not exist. Skipping.")
            continue

        logger.info(f"Processing source: {source}")
        
        # Get all video files in this source
        video_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(('.webm', '.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(video_files)} video files in {source}")
        
        for video_path in video_files:
            # Extract sign name from path
            sign_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Create output directory preserving source information
            output_dir = os.path.join(processed_path, source.replace('/', '_'), sign_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output filenames
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            landmarks_file = os.path.join(output_dir, f"{base_name}_landmarks.npy")
            metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
            
            # Skip if already processed
            if os.path.exists(landmarks_file) and os.path.exists(metadata_file):
                logger.info(f"Skipping {base_name}, already processed.")
                continue
            
            try:
                logger.info(f"Processing: {base_name}")
                
                # Extract landmarks and metadata
                landmarks, metadata = extractor.process_video(video_path)
                
                if landmarks is not None and metadata is not None:
                    # Save landmarks
                    np.save(landmarks_file, landmarks)
                    
                    # Save metadata
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    total_processed += 1
                    logger.info(f"Successfully processed {base_name} - {landmarks.shape[0]} frames")
                else:
                    total_errors += 1
                    logger.error(f"Failed to extract landmarks from {base_name}")
                    
            except Exception as e:
                total_errors += 1
                logger.error(f"Error processing {base_name}: {str(e)}")
                continue

    logger.info(f"Extraction complete. Processed: {total_processed}, Errors: {total_errors}")

if __name__ == '__main__':
    main() 