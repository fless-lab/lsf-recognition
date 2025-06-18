import os
import numpy as np
import json
import logging
from pathlib import Path
import random
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import copy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAugmenter:
    def __init__(self, augmentation_factor=10):
        """
        Initialize the advanced augmenter.
        
        Args:
            augmentation_factor: Number of augmented versions to create per original sample
        """
        self.augmentation_factor = augmentation_factor
        self.rng = np.random.RandomState(42)  # For reproducibility
        
    def add_gaussian_noise(self, data, noise_level=0.01):
        """Add Gaussian noise to landmarks."""
        noise = self.rng.normal(0, noise_level, data.shape)
        return data + noise
    
    def add_temporal_noise(self, data, noise_level=0.005):
        """Add temporal noise (smooth variations over time)."""
        # Create smooth noise that varies over time
        time_steps = data.shape[0]
        temporal_noise = self.rng.normal(0, noise_level, (time_steps, 1))
        
        # Smooth the noise using a simple moving average
        window_size = max(1, time_steps // 10)
        smoothed_noise = np.convolve(temporal_noise.flatten(), 
                                   np.ones(window_size)/window_size, 
                                   mode='same').reshape(-1, 1)
        
        # Apply to all landmark dimensions
        return data + smoothed_noise * np.ones_like(data)
    
    def scale_landmarks(self, data, scale_range=0.1):
        """Scale landmarks with realistic bounds."""
        scale_factor = 1.0 + self.rng.uniform(-scale_range, scale_range)
        return data * scale_factor
    
    def translate_landmarks(self, data, translate_range=0.05):
        """Translate landmarks in 3D space."""
        # Different translation for each dimension
        translation = self.rng.uniform(-translate_range, translate_range, 3)
        
        # Apply translation to x, y, z coordinates
        # Assuming landmarks are in format: [x1, y1, z1, v1, x2, y2, z2, v2, ...]
        translated_data = data.copy()
        
        # For pose landmarks (33 points, 4 values each: x, y, z, visibility)
        pose_end = 33 * 4
        for i in range(0, pose_end, 4):
            if i + 2 < pose_end:
                translated_data[:, i] += translation[0]  # x
                translated_data[:, i + 1] += translation[1]  # y
                translated_data[:, i + 2] += translation[2]  # z
        
        # For face landmarks (468 points, 3 values each: x, y, z)
        face_start = pose_end
        face_end = face_start + 468 * 3
        for i in range(face_start, face_end, 3):
            if i + 2 < face_end:
                translated_data[:, i] += translation[0]  # x
                translated_data[:, i + 1] += translation[1]  # y
                translated_data[:, i + 2] += translation[2]  # z
        
        # For hand landmarks (21 points each, 3 values each: x, y, z)
        hand_start = face_end
        for hand_end in [hand_start + 21 * 3, hand_start + 42 * 3]:  # Left and right hands
            for i in range(hand_start, hand_end, 3):
                if i + 2 < hand_end:
                    translated_data[:, i] += translation[0]  # x
                    translated_data[:, i + 1] += translation[1]  # y
                    translated_data[:, i + 2] += translation[2]  # z
            hand_start = hand_end
        
        return translated_data
    
    def rotate_landmarks(self, data, max_angle=15):
        """Rotate landmarks around Y-axis (vertical rotation)."""
        angle_deg = self.rng.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle_deg)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, 0, sin_a],
                                  [0, 1, 0],
                                  [-sin_a, 0, cos_a]])
        
        rotated_data = data.copy()
        
        # Apply rotation to pose landmarks
        pose_end = 33 * 4
        for i in range(0, pose_end, 4):
            if i + 2 < pose_end:
                xyz = data[:, i:i+3]
                rotated_xyz = np.dot(xyz, rotation_matrix.T)
                rotated_data[:, i:i+3] = rotated_xyz
        
        # Apply rotation to face landmarks
        face_start = pose_end
        face_end = face_start + 468 * 3
        for i in range(face_start, face_end, 3):
            if i + 2 < face_end:
                xyz = data[:, i:i+3]
                rotated_xyz = np.dot(xyz, rotation_matrix.T)
                rotated_data[:, i:i+3] = rotated_xyz
        
        # Apply rotation to hand landmarks
        hand_start = face_end
        for hand_end in [hand_start + 21 * 3, hand_start + 42 * 3]:
            for i in range(hand_start, hand_end, 3):
                if i + 2 < hand_end:
                    xyz = data[:, i:i+3]
                    rotated_xyz = np.dot(xyz, rotation_matrix.T)
                    rotated_data[:, i:i+3] = rotated_xyz
            hand_start = hand_end
        
        return rotated_data
    
    def temporal_warping(self, data, warp_factor=0.2):
        """Apply temporal warping to simulate different signing speeds."""
        time_steps = data.shape[0]
        
        # Create warping function
        original_time = np.linspace(0, 1, time_steps)
        warp_amount = self.rng.uniform(-warp_factor, warp_factor)
        
        # Create non-linear warping
        warped_time = original_time + warp_amount * np.sin(np.pi * original_time)
        warped_time = np.clip(warped_time, 0, 1)
        
        # Interpolate data
        warped_data = np.zeros_like(data)
        for dim in range(data.shape[1]):
            interpolator = interp1d(original_time, data[:, dim], 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
            warped_data[:, dim] = interpolator(warped_time)
        
        return warped_data
    
    def partial_occlusion(self, data, occlusion_prob=0.3):
        """Simulate partial occlusion by zeroing out some landmarks."""
        occluded_data = data.copy()
        
        # Randomly occlude some landmarks
        for frame_idx in range(data.shape[0]):
            if self.rng.random() < occlusion_prob:
                # Randomly select landmarks to occlude
                num_landmarks_to_occlude = self.rng.randint(1, 5)
                landmark_indices = self.rng.choice(data.shape[1], num_landmarks_to_occlude, replace=False)
                occluded_data[frame_idx, landmark_indices] = 0
        
        return occluded_data
    
    def perspective_transform(self, data, perspective_factor=0.1):
        """Apply perspective transformation to simulate different viewing angles."""
        # Create perspective transformation matrix
        perspective_matrix = np.array([
            [1 + self.rng.uniform(-perspective_factor, perspective_factor), 
             self.rng.uniform(-perspective_factor, perspective_factor), 0],
            [self.rng.uniform(-perspective_factor, perspective_factor), 
             1 + self.rng.uniform(-perspective_factor, perspective_factor), 0],
            [self.rng.uniform(-perspective_factor, perspective_factor), 
             self.rng.uniform(-perspective_factor, perspective_factor), 1]
        ])
        
        transformed_data = data.copy()
        
        # Apply transformation to x, y coordinates (z remains unchanged)
        for frame_idx in range(data.shape[0]):
            for landmark_idx in range(0, data.shape[1], 3):
                if landmark_idx + 1 < data.shape[1]:
                    xyz = data[frame_idx, landmark_idx:landmark_idx+3]
                    # Apply perspective transformation to x, y
                    xy_homogeneous = np.array([xyz[0], xyz[1], 1])
                    transformed_xy = np.dot(perspective_matrix, xy_homogeneous)
                    transformed_data[frame_idx, landmark_idx:landmark_idx+2] = transformed_xy[:2]
        
        return transformed_data
    
    def mixup_augmentation(self, data1, data2, alpha=0.2):
        """Mixup augmentation between two sequences."""
        # Ensure same length
        min_length = min(data1.shape[0], data2.shape[0])
        data1 = data1[:min_length]
        data2 = data2[:min_length]
        
        # Create mixing weight
        lam = self.rng.beta(alpha, alpha)
        
        # Mix the sequences
        mixed_data = lam * data1 + (1 - lam) * data2
        
        return mixed_data
    
    def augment_sequence(self, data, metadata=None):
        """Apply a combination of augmentations to a sequence."""
        augmented_sequences = []
        augmentation_types = []
        
        for i in range(self.augmentation_factor):
            augmented_data = data.copy()
            applied_augmentations = []
            
            # Apply random combination of augmentations
            if self.rng.random() > 0.3:
                augmented_data = self.add_gaussian_noise(augmented_data)
                applied_augmentations.append('gaussian_noise')
            
            if self.rng.random() > 0.4:
                augmented_data = self.add_temporal_noise(augmented_data)
                applied_augmentations.append('temporal_noise')
            
            if self.rng.random() > 0.5:
                augmented_data = self.scale_landmarks(augmented_data)
                applied_augmentations.append('scale')
            
            if self.rng.random() > 0.5:
                augmented_data = self.translate_landmarks(augmented_data)
                applied_augmentations.append('translate')
            
            if self.rng.random() > 0.6:
                augmented_data = self.rotate_landmarks(augmented_data)
                applied_augmentations.append('rotate')
            
            if self.rng.random() > 0.7:
                augmented_data = self.temporal_warping(augmented_data)
                applied_augmentations.append('temporal_warp')
            
            if self.rng.random() > 0.8:
                augmented_data = self.partial_occlusion(augmented_data)
                applied_augmentations.append('occlusion')
            
            if self.rng.random() > 0.8:
                augmented_data = self.perspective_transform(augmented_data)
                applied_augmentations.append('perspective')
            
            augmented_sequences.append(augmented_data)
            augmentation_types.append(applied_augmentations)
        
        return augmented_sequences, augmentation_types
    
    def augment_dataset(self, input_dir, output_dir):
        """Augment all sequences in a directory."""
        logger.info(f"Starting augmentation from {input_dir} to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        total_original = 0
        total_augmented = 0
        
        # Process each sign directory
        for sign_name in os.listdir(input_dir):
            sign_dir = os.path.join(input_dir, sign_name)
            if not os.path.isdir(sign_dir):
                continue
            
            output_sign_dir = os.path.join(output_dir, sign_name)
            os.makedirs(output_sign_dir, exist_ok=True)
            
            logger.info(f"Augmenting sign: {sign_name}")
            
            # Get all landmark files for this sign
            landmark_files = [f for f in os.listdir(sign_dir) if f.endswith('_landmarks.npy')]
            
            for landmark_file in landmark_files:
                landmark_path = os.path.join(sign_dir, landmark_file)
                metadata_path = landmark_path.replace('_landmarks.npy', '_metadata.json')
                
                # Load original data
                original_data = np.load(landmark_path)
                total_original += 1
                
                # Load metadata if available
                metadata = None
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # Save original data
                original_output_path = os.path.join(output_sign_dir, landmark_file)
                np.save(original_output_path, original_data)
                
                if metadata:
                    original_metadata_path = os.path.join(output_sign_dir, 
                                                        landmark_file.replace('_landmarks.npy', '_metadata.json'))
                    with open(original_metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # Generate augmented versions
                augmented_sequences, augmentation_types = self.augment_sequence(original_data, metadata)
                
                base_name = landmark_file.replace('_landmarks.npy', '')
                
                for i, (augmented_data, aug_types) in enumerate(zip(augmented_sequences, augmentation_types)):
                    # Create augmented filename
                    aug_filename = f"{base_name}_aug_{i+1:03d}.npy"
                    aug_path = os.path.join(output_sign_dir, aug_filename)
                    
                    # Save augmented data
                    np.save(aug_path, augmented_data)
                    
                    # Save augmentation metadata
                    aug_metadata = {
                        'original_file': landmark_file,
                        'augmentation_types': aug_types,
                        'augmentation_index': i + 1,
                        'original_metadata': metadata
                    }
                    
                    aug_metadata_path = aug_path.replace('_landmarks.npy', '_metadata.json')
                    with open(aug_metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(aug_metadata, f, indent=2, ensure_ascii=False)
                    
                    total_augmented += 1
        
        logger.info(f"Augmentation complete!")
        logger.info(f"Original samples: {total_original}")
        logger.info(f"Augmented samples: {total_augmented}")
        logger.info(f"Total samples: {total_original + total_augmented}")

def main():
    """Main function to run advanced augmentation."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    
    # Define paths
    train_dir = os.path.join(data_path, 'train')
    augmented_train_dir = os.path.join(data_path, 'train_augmented')
    
    # Initialize augmenter
    augmenter = AdvancedAugmenter(augmentation_factor=15)  # 15 augmented versions per original
    
    # Run augmentation
    augmenter.augment_dataset(train_dir, augmented_train_dir)

if __name__ == '__main__':
    main() 