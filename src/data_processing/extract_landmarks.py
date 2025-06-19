import cv2
import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
import numpy as np
import os
import json
from pathlib import Path
import logging
import sys
from src.utils.landmark_utils import extract_landmark_vector

# Setup logging : console uniquement pour affichage en temps réel via subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LandmarkExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        logger.info("Initialisation de MediaPipe Holistic...")
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,  # 0, 1, or 2
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            refine_face_landmarks=True
        )
        logger.info("MediaPipe Holistic initialisé.")
        
    def extract_keypoints(self, results):
        """Extracts landmark data into a structured format with confidence scores."""
        # Utilise la fonction utilitaire unique
        landmarks = extract_landmark_vector(results)
        # Calculate confidence scores
        pose_confidence = np.mean([res.visibility for res in results.pose_landmarks.landmark]) if results.pose_landmarks else 0.0
        face_confidence = 1.0 if results.face_landmarks else 0.0
        lh_confidence = 1.0 if results.left_hand_landmarks else 0.0
        rh_confidence = 1.0 if results.right_hand_landmarks else 0.0
        metadata = {
            'pose_confidence': float(pose_confidence),
            'face_confidence': float(face_confidence),
            'left_hand_confidence': float(lh_confidence),
            'right_hand_confidence': float(rh_confidence),
            'total_frames': 1,
            'landmark_dimensions': {
                'pose': 33*4,
                'face': 468*3,
                'left_hand': 21*3,
                'right_hand': 21*3
            }
        }
        return landmarks, metadata

    def process_video(self, video_path):
        logger.info(f"Début du traitement vidéo : {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Impossible d'ouvrir la vidéo {video_path}")
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
            if frame_idx % 10 == 0:
                logger.info(f"Frame {frame_idx}/{frame_count}...")

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
        
        logger.info(f"Extraction terminée pour {video_path} : {len(keypoints_list)} frames extraites.")
        return np.array(keypoints_list), video_metadata

def main():
    logger.info("=== DÉBUT Extraction des landmarks ===")
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
    
    # Dictionary to track sign sources
    sign_sources = {}
    
    # Compter le nombre total de vidéos à traiter (toutes sources confondues)
    all_video_files = []
    for source in sources:
        source_path = os.path.join(raw_path, source)
        if not os.path.exists(source_path):
            continue
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith(('.webm', '.mp4', '.avi', '.mov')):
                    all_video_files.append((source, os.path.join(root, file)))
    total_videos = len(all_video_files)
    logger.info(f"Nombre total de vidéos à traiter : {total_videos}")

    current_idx = 0
    for source in sources:
        logger.info(f"--- Source : {source} ---")
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
            current_idx += 1
            sign_name = os.path.splitext(os.path.basename(video_path))[0]
            source_name = source.split('/')[-1]
            logger.info(f"Traitement de la vidéo {current_idx}/{total_videos} : {sign_name} ({source_name})")

            # Create output directory: processed/{sign_name}/
            output_dir = os.path.join(processed_path, sign_name)
            os.makedirs(output_dir, exist_ok=True)

            # Create output filenames: {source_name}.npy
            landmarks_file = os.path.join(output_dir, f"{source_name}.npy")
            metadata_file = os.path.join(output_dir, f"{source_name}_metadata.json")

            # Track sign sources for later analysis
            if sign_name not in sign_sources:
                sign_sources[sign_name] = []
            if source_name not in sign_sources[sign_name]:
                sign_sources[sign_name].append(source_name)

            # Skip if already processed
            if os.path.exists(landmarks_file) and os.path.exists(metadata_file):
                logger.info(f"Skipping {sign_name} from {source_name}, already processed.")
                continue

            try:
                logger.info(f"Extraction : {sign_name} depuis {source_name}")
                # Extract landmarks and metadata
                landmarks, metadata = extractor.process_video(video_path)

                if landmarks is not None and metadata is not None:
                    logger.info(f"Sauvegarde des landmarks et métadonnées pour {sign_name} ({source_name})")
                    # Save landmarks
                    np.save(landmarks_file, landmarks)

                    # Save metadata
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    total_processed += 1
                    logger.info(f"✅ Vidéo {current_idx}/{total_videos} traitée : {sign_name} ({source_name}) - {landmarks.shape[0]} frames")
                else:
                    total_errors += 1
                    logger.error(f"❌ Extraction échouée pour {sign_name} ({source_name})")

            except Exception as e:
                total_errors += 1
                logger.error(f"Erreur lors du traitement de {sign_name} ({source_name}) : {str(e)}")
                continue

    # Save sign sources analysis
    sources_analysis_file = os.path.join(data_path, 'sign_sources_analysis.json')
    with open(sources_analysis_file, 'w', encoding='utf-8') as f:
        json.dump(sign_sources, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info(f"=== FIN Extraction landmarks : {total_processed} vidéos traitées, {total_errors} erreurs ===")
    logger.info(f"Found {len(sign_sources)} unique signs")
    
    # Count signs by number of sources
    source_counts = {}
    for sign, sources in sign_sources.items():
        num_sources = len(sources)
        if num_sources not in source_counts:
            source_counts[num_sources] = []
        source_counts[num_sources].append(sign)
    
    logger.info("Sign distribution by number of sources:")
    for num_sources, signs in sorted(source_counts.items()):
        logger.info(f"  {num_sources} source(s): {len(signs)} signs")
        if num_sources <= 3:  # Show details for signs with few sources
            logger.info(f"    Examples: {', '.join(signs[:5])}")

if __name__ == '__main__':
    main() 