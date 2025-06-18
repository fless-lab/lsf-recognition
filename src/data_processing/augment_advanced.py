import os
import numpy as np
import json
import logging
from pathlib import Path
import random
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import copy
from typing import List, Tuple, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAugmenter:
    def __init__(self, augmentation_factor=5):
        """Initialize the advanced augmenter with multiple augmentation techniques."""
        logger.info(f"Initialisation de l'augmentateur avancé (factor={augmentation_factor})")
        self.augmentation_factor = augmentation_factor
        
    def augment_landmarks(self, landmarks: np.ndarray, metadata: Dict[str, Any]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Apply multiple augmentation techniques to create diverse training samples."""
        logger.info(f"Augmentation d'un échantillon ({landmarks.shape[0]} frames)")
        augmented_samples = []
        
        # Original sample (no augmentation)
        augmented_samples.append((landmarks.copy(), metadata.copy()))
        
        # Generate augmented versions
        for i in range(self.augmentation_factor - 1):
            # Choose augmentation technique
            technique = random.choice(['spatial', 'temporal', 'occlusion', 'perspective', 'mixup'])
            
            if technique == 'spatial':
                aug_landmarks, aug_metadata = self._spatial_augmentation(landmarks, metadata)
            elif technique == 'temporal':
                aug_landmarks, aug_metadata = self._temporal_augmentation(landmarks, metadata)
            elif technique == 'occlusion':
                aug_landmarks, aug_metadata = self._occlusion_augmentation(landmarks, metadata)
            elif technique == 'perspective':
                aug_landmarks, aug_metadata = self._perspective_augmentation(landmarks, metadata)
            elif technique == 'mixup':
                aug_landmarks, aug_metadata = self._mixup_augmentation(landmarks, metadata)
            
            # Add augmentation info to metadata
            aug_metadata['augmentation'] = {
                'technique': technique,
                'version': i + 1,
                'original_file': metadata.get('video_path', 'unknown')
            }
            
            augmented_samples.append((aug_landmarks, aug_metadata))
        
        return augmented_samples
    
    def _spatial_augmentation(self, landmarks: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply spatial transformations (rotation, scaling, translation)."""
        aug_landmarks = landmarks.copy()
        
        # Random rotation (small angles)
        angle = random.uniform(-15, 15)  # degrees
        angle_rad = np.radians(angle)
        
        # Random scaling
        scale_x = random.uniform(0.9, 1.1)
        scale_y = random.uniform(0.9, 1.1)
        
        # Random translation
        tx = random.uniform(-0.05, 0.05)
        ty = random.uniform(-0.05, 0.05)
        
        # Apply transformations to pose landmarks (first 33*4 values)
        pose_start = 0
        pose_end = 33 * 4
        
        for frame_idx in range(landmarks.shape[0]):
            frame_landmarks = landmarks[frame_idx]
            pose_landmarks = frame_landmarks[pose_start:pose_end].reshape(-1, 4)
            
            # Apply transformations to x, y coordinates (keep z and visibility)
            x_coords = pose_landmarks[:, 0]
            y_coords = pose_landmarks[:, 1]
            
            # Rotation
            x_rot = x_coords * np.cos(angle_rad) - y_coords * np.sin(angle_rad)
            y_rot = x_coords * np.sin(angle_rad) + y_coords * np.cos(angle_rad)
            
            # Scaling
            x_scaled = x_rot * scale_x
            y_scaled = y_rot * scale_y
            
            # Translation
            x_final = x_scaled + tx
            y_final = y_scaled + ty
            
            # Update landmarks
            pose_landmarks[:, 0] = x_final
            pose_landmarks[:, 1] = y_final
            
            # Apply same transformations to face landmarks (next 468*3 values)
            face_start = pose_end
            face_end = face_start + 468 * 3
            
            face_landmarks = frame_landmarks[face_start:face_end].reshape(-1, 3)
            x_coords = face_landmarks[:, 0]
            y_coords = face_landmarks[:, 1]
            
            x_rot = x_coords * np.cos(angle_rad) - y_coords * np.sin(angle_rad)
            y_rot = x_coords * np.sin(angle_rad) + y_coords * np.cos(angle_rad)
            x_scaled = x_rot * scale_x
            y_scaled = y_rot * scale_y
            x_final = x_scaled + tx
            y_final = y_scaled + ty
            
            face_landmarks[:, 0] = x_final
            face_landmarks[:, 1] = y_final
            
            # Apply to hand landmarks
            lh_start = face_end
            lh_end = lh_start + 21 * 3
            rh_start = lh_end
            rh_end = rh_start + 21 * 3
            
            for hand_start, hand_end in [(lh_start, lh_end), (rh_start, rh_end)]:
                hand_landmarks = frame_landmarks[hand_start:hand_end].reshape(-1, 3)
                x_coords = hand_landmarks[:, 0]
                y_coords = hand_landmarks[:, 1]
                
                x_rot = x_coords * np.cos(angle_rad) - y_coords * np.sin(angle_rad)
                y_rot = x_coords * np.sin(angle_rad) + y_coords * np.cos(angle_rad)
                x_scaled = x_rot * scale_x
                y_scaled = y_rot * scale_y
                x_final = x_scaled + tx
                y_final = y_scaled + ty
                
                hand_landmarks[:, 0] = x_final
                hand_landmarks[:, 1] = y_final
            
            # Reconstruct frame
            aug_landmarks[frame_idx] = frame_landmarks
        
        aug_metadata = metadata.copy()
        aug_metadata['spatial_transform'] = {
            'rotation_angle': angle,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'translation_x': tx,
            'translation_y': ty
        }
        
        return aug_landmarks, aug_metadata
    
    def _temporal_augmentation(self, landmarks: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply temporal transformations (speed variation, frame dropping)."""
        num_frames = landmarks.shape[0]
        
        if num_frames < 10:  # Too short for temporal augmentation
            return landmarks.copy(), metadata.copy()
        
        # Random speed variation
        speed_factor = random.uniform(0.8, 1.2)
        
        # Calculate new frame indices
        new_num_frames = int(num_frames * speed_factor)
        new_indices = np.linspace(0, num_frames - 1, new_num_frames, dtype=int)
        
        # Interpolate landmarks
        aug_landmarks = landmarks[new_indices]
        
        # Random frame dropping (drop 0-20% of frames)
        drop_ratio = random.uniform(0, 0.2)
        num_to_drop = int(new_num_frames * drop_ratio)
        
        if num_to_drop > 0:
            drop_indices = random.sample(range(new_num_frames), num_to_drop)
            keep_indices = [i for i in range(new_num_frames) if i not in drop_indices]
            aug_landmarks = aug_landmarks[keep_indices]
        
        aug_metadata = metadata.copy()
        aug_metadata['temporal_transform'] = {
            'speed_factor': speed_factor,
            'original_frames': num_frames,
            'new_frames': aug_landmarks.shape[0],
            'dropped_frames': num_to_drop
        }
        
        return aug_landmarks, aug_metadata
    
    def _occlusion_augmentation(self, landmarks: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply occlusion simulation by zeroing out random landmarks."""
        aug_landmarks = landmarks.copy()
        
        # Define landmark groups
        pose_start, pose_end = 0, 33 * 4
        face_start, face_end = pose_end, pose_end + 468 * 3
        lh_start, lh_end = face_end, face_end + 21 * 3
        rh_start, rh_end = lh_end, lh_end + 21 * 3
        
        # Randomly choose which group to occlude
        groups = ['pose', 'face', 'left_hand', 'right_hand']
        occlude_group = random.choice(groups)
        
        # Random occlusion duration (frames)
        occlusion_start = random.randint(0, max(0, landmarks.shape[0] - 5))
        occlusion_duration = random.randint(1, min(5, landmarks.shape[0] - occlusion_start))
        
        # Apply occlusion
        if occlude_group == 'pose':
            start_idx, end_idx = pose_start, pose_end
        elif occlude_group == 'face':
            start_idx, end_idx = face_start, face_end
        elif occlude_group == 'left_hand':
            start_idx, end_idx = lh_start, lh_end
        else:  # right_hand
            start_idx, end_idx = rh_start, rh_end
        
        for frame_idx in range(occlusion_start, occlusion_start + occlusion_duration):
            if frame_idx < landmarks.shape[0]:
                aug_landmarks[frame_idx, start_idx:end_idx] = 0
        
        aug_metadata = metadata.copy()
        aug_metadata['occlusion'] = {
            'occluded_group': occlude_group,
            'start_frame': occlusion_start,
            'duration': occlusion_duration
        }
        
        return aug_landmarks, aug_metadata
    
    def _perspective_augmentation(self, landmarks: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply perspective transformations (viewpoint changes)."""
        aug_landmarks = landmarks.copy()
        
        # Random perspective parameters
        perspective_strength = random.uniform(0.1, 0.3)
        
        # Apply perspective transformation to x coordinates
        for frame_idx in range(landmarks.shape[0]):
            frame_landmarks = landmarks[frame_idx]
            
            # Apply to all landmark groups
            for group_start, group_end in [(0, 33*4), (33*4, 33*4+468*3), (33*4+468*3, 33*4+468*3+21*3), (33*4+468*3+21*3, 33*4+468*3+21*3+21*3)]:
                group_landmarks = frame_landmarks[group_start:group_end]
                
                # Reshape based on group
                if group_start == 0:  # pose
                    landmarks_reshaped = group_landmarks.reshape(-1, 4)
                    coords = landmarks_reshaped[:, :3]  # x, y, z
                else:  # face, hands
                    landmarks_reshaped = group_landmarks.reshape(-1, 3)
                    coords = landmarks_reshaped
                
                # Apply perspective transformation
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                
                # Simple perspective effect
                perspective_factor = 1 + perspective_strength * (y_coords - 0.5)
                x_perspective = x_coords * perspective_factor
                
                # Update coordinates
                coords[:, 0] = x_perspective
                
                # Reconstruct group
                if group_start == 0:  # pose
                    landmarks_reshaped[:, :3] = coords
                    group_landmarks = landmarks_reshaped.flatten()
                else:  # face, hands
                    landmarks_reshaped = coords
                    group_landmarks = landmarks_reshaped.flatten()
                
                aug_landmarks[frame_idx, group_start:group_end] = group_landmarks
        
        aug_metadata = metadata.copy()
        aug_metadata['perspective'] = {
            'perspective_strength': perspective_strength
        }
        
        return aug_landmarks, aug_metadata
    
    def _mixup_augmentation(self, landmarks: np.ndarray, metadata: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply mixup augmentation by blending with a slightly modified version."""
        # Create a slightly modified version
        modified = landmarks.copy()
        
        # Add small random noise
        noise_std = 0.01
        noise = np.random.normal(0, noise_std, landmarks.shape)
        modified += noise
        
        # Mixup factor
        alpha = random.uniform(0.3, 0.7)
        
        # Blend the original and modified versions
        aug_landmarks = alpha * landmarks + (1 - alpha) * modified
        
        aug_metadata = metadata.copy()
        aug_metadata['mixup'] = {
            'alpha': alpha,
            'noise_std': noise_std
        }
        
        return aug_landmarks, aug_metadata

def augment_train_dataset(train_path: str, augmentation_factor: int = 5):
    """Augment the entire train dataset."""
    logger.info(f"=== DÉBUT Augmentation du dataset train ({train_path}) ===")
    
    augmenter = AdvancedAugmenter(augmentation_factor=augmentation_factor)
    
    # Walk through train directory
    total_original = 0
    total_augmented = 0
    
    for sign_folder in os.listdir(train_path):
        sign_path = os.path.join(train_path, sign_folder)
        if not os.path.isdir(sign_path):
            continue
        
        logger.info(f"--- Signe : {sign_folder} ---")
        
        # Find all landmark files for this sign
        landmark_files = [f for f in os.listdir(sign_path) if f.endswith('.npy') and not f.endswith('_metadata.json')]
        
        for landmark_file in landmark_files:
            logger.info(f"Augmentation du fichier : {landmark_file}")
            source_name = landmark_file.replace('.npy', '')
            landmark_path = os.path.join(sign_path, landmark_file)
            metadata_path = os.path.join(sign_path, f"{source_name}_metadata.json")
            
            # Load landmarks and metadata
            landmarks = np.load(landmark_path)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {'video_path': 'unknown'}
            
            # Generate augmented versions
            augmented_samples = augmenter.augment_landmarks(landmarks, metadata)
            
            # Save augmented versions
            for i, (aug_landmarks, aug_metadata) in enumerate(augmented_samples):
                if i == 0:  # Original sample
                    continue
                
                # Create augmented filename
                aug_filename = f"{source_name}_aug_{i:03d}.npy"
                aug_metadata_filename = f"{source_name}_aug_{i:03d}_metadata.json"
                
                aug_landmark_path = os.path.join(sign_path, aug_filename)
                aug_metadata_path = os.path.join(sign_path, aug_metadata_filename)
                
                # Save augmented data
                np.save(aug_landmark_path, aug_landmarks)
                with open(aug_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(aug_metadata, f, indent=2, ensure_ascii=False)
                
                total_augmented += 1
            
            total_original += 1
    
    logger.info(f"=== FIN Augmentation : {total_original} originaux, {total_augmented} augmentés ===")

def main():
    """Main function to augment the train dataset."""
    logger.info("=== Lancement de l'augmentation avancée ===")
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    train_path = os.path.join(data_path, 'train')
    
    # Check if train directory exists
    if not os.path.exists(train_path):
        logger.error(f"Train directory not found: {train_path}")
        logger.error("Please run the consolidation script first.")
        return
    
    # Augment train dataset
    augment_train_dataset(train_path, augmentation_factor=5)
    logger.info("=== Fin de l'augmentation avancée ===")

if __name__ == '__main__':
    main() 