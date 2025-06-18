#!/usr/bin/env python3
"""
Landmark Visualization Script

This script provides tools to visualize extracted landmarks and augmented data
to verify quality and understand the data structure.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LandmarkVisualizer:
    def __init__(self):
        """Initialize the landmark visualizer."""
        self.colors = {
            'pose': (0, 255, 0),      # Green
            'face': (255, 0, 0),      # Blue
            'left_hand': (0, 0, 255), # Red
            'right_hand': (255, 255, 0) # Cyan
        }
        
    def load_landmarks(self, file_path):
        """Load landmarks from .npy file."""
        landmarks = np.load(file_path)
        return landmarks
    
    def load_metadata(self, file_path):
        """Load metadata from .json file."""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def visualize_frame(self, landmarks, frame_idx=0, title="Landmarks Visualization"):
        """Visualize landmarks for a single frame."""
        if frame_idx >= landmarks.shape[0]:
            logger.error(f"Frame index {frame_idx} out of range. Max frames: {landmarks.shape[0]}")
            return None
        
        # Create a blank image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        frame_landmarks = landmarks[frame_idx]
        
        # Define landmark groups
        pose_start, pose_end = 0, 33 * 4
        face_start, face_end = pose_end, pose_end + 468 * 3
        lh_start, lh_end = face_end, face_end + 21 * 3
        rh_start, rh_end = lh_end, lh_end + 21 * 3
        
        # Draw pose landmarks
        pose_landmarks = frame_landmarks[pose_start:pose_end].reshape(-1, 4)
        for i, (x, y, z, v) in enumerate(pose_landmarks):
            if v > 0.5:  # Only draw visible landmarks
                x_pixel = int(x * 640)
                y_pixel = int(y * 480)
                cv2.circle(img, (x_pixel, y_pixel), 3, self.colors['pose'], -1)
        
        # Draw face landmarks
        face_landmarks = frame_landmarks[face_start:face_end].reshape(-1, 3)
        for i, (x, y, z) in enumerate(face_landmarks):
            x_pixel = int(x * 640)
            y_pixel = int(y * 480)
            cv2.circle(img, (x_pixel, y_pixel), 2, self.colors['face'], -1)
        
        # Draw hand landmarks
        lh_landmarks = frame_landmarks[lh_start:lh_end].reshape(-1, 3)
        for i, (x, y, z) in enumerate(lh_landmarks):
            x_pixel = int(x * 640)
            y_pixel = int(y * 480)
            cv2.circle(img, (x_pixel, y_pixel), 4, self.colors['left_hand'], -1)
        
        rh_landmarks = frame_landmarks[rh_start:rh_end].reshape(-1, 3)
        for i, (x, y, z) in enumerate(rh_landmarks):
            x_pixel = int(x * 640)
            y_pixel = int(y * 480)
            cv2.circle(img, (x_pixel, y_pixel), 4, self.colors['right_hand'], -1)
        
        # Add title
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
    def create_animation(self, landmarks, output_path, fps=10):
        """Create a video animation of landmarks over time."""
        import cv2
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        logger.info(f"Création de l'animation vidéo : {output_path}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        for frame_idx in range(landmarks.shape[0]):
            img = self.visualize_frame(landmarks, frame_idx, "Landmarks Animation")
            if img is not None:
                out.write(img)
        
        out.release()
        logger.info(f"Animation saved to: {output_path}")
    
    def plot_confidence_over_time(self, landmarks, metadata, output_path=None):
        """Plot confidence scores over time."""
        if not metadata or 'frame_metadata' not in metadata:
            logger.warning("No frame metadata available for confidence plotting")
            return
        
        frame_metadata = metadata['frame_metadata']
        
        # Extract confidence scores
        pose_conf = [meta.get('pose_confidence', 0) for meta in frame_metadata]
        face_conf = [meta.get('face_confidence', 0) for meta in frame_metadata]
        lh_conf = [meta.get('left_hand_confidence', 0) for meta in frame_metadata]
        rh_conf = [meta.get('right_hand_confidence', 0) for meta in frame_metadata]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        frames = range(len(frame_metadata))
        
        plt.subplot(2, 2, 1)
        plt.plot(frames, pose_conf, 'g-', label='Pose')
        plt.title('Pose Confidence Over Time')
        plt.ylabel('Confidence')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(frames, face_conf, 'b-', label='Face')
        plt.title('Face Confidence Over Time')
        plt.ylabel('Confidence')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(frames, lh_conf, 'r-', label='Left Hand')
        plt.title('Left Hand Confidence Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(frames, rh_conf, 'c-', label='Right Hand')
        plt.title('Right Hand Confidence Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_original_vs_augmented(self, original_path, augmented_path, output_dir):
        """Compare original and augmented landmarks."""
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        comparison_path = os.path.join(output_dir, 'comparison.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        logger.info(f"Création de la comparaison vidéo : {comparison_path}")
        out = cv2.VideoWriter(comparison_path, fourcc, 10, (1280, 480))
        
        # Load data
        original_landmarks = self.load_landmarks(original_path)
        augmented_landmarks = self.load_landmarks(augmented_path)
        
        original_metadata_path = original_path.replace('.npy', '_metadata.json')
        augmented_metadata_path = augmented_path.replace('.npy', '_metadata.json')
        
        original_metadata = self.load_metadata(original_metadata_path)
        augmented_metadata = self.load_metadata(augmented_metadata_path)
        
        # Create comparison animation
        max_frames = min(original_landmarks.shape[0], augmented_landmarks.shape[0])
        
        for frame_idx in range(max_frames):
            # Create side-by-side comparison
            original_img = self.visualize_frame(original_landmarks, frame_idx, "Original")
            augmented_img = self.visualize_frame(augmented_landmarks, frame_idx, "Augmented")
            
            if original_img is not None and augmented_img is not None:
                combined = np.hstack([original_img, augmented_img])
                out.write(combined)
        
        out.release()
        logger.info(f"Comparison video saved to: {comparison_path}")
        
        # Plot confidence comparison
        if original_metadata and augmented_metadata:
            plt.figure(figsize=(15, 10))
            
            # Original confidence
            plt.subplot(2, 1, 1)
            if 'frame_metadata' in original_metadata:
                pose_conf = [meta.get('pose_confidence', 0) for meta in original_metadata['frame_metadata']]
                plt.plot(pose_conf, 'g-', label='Original Pose Confidence')
            plt.title('Original vs Augmented Confidence Comparison')
            plt.ylabel('Confidence')
            plt.legend()
            
            # Augmented confidence
            plt.subplot(2, 1, 2)
            if 'frame_metadata' in augmented_metadata:
                pose_conf = [meta.get('pose_confidence', 0) for meta in augmented_metadata['frame_metadata']]
                plt.plot(pose_conf, 'r-', label='Augmented Pose Confidence')
            plt.xlabel('Frame')
            plt.ylabel('Confidence')
            plt.legend()
            
            comparison_plot_path = os.path.join(output_dir, 'confidence_comparison.png')
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confidence comparison plot saved to: {comparison_plot_path}")

def visualize_dataset_sample(data_path, sign_name=None, source_name=None, output_dir=None):
    """Visualize a sample from the dataset."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    
    if data_path == 'processed':
        data_dir = os.path.join(project_root, 'data', 'processed')
    elif data_path == 'train':
        data_dir = os.path.join(project_root, 'data', 'train')
    elif data_path == 'test':
        data_dir = os.path.join(project_root, 'data', 'test')
    else:
        data_dir = data_path
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(project_root, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = LandmarkVisualizer()
    
    # Find a sample to visualize
    if sign_name is None:
        # Pick first available sign
        signs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not signs:
            logger.error("No signs found in data directory")
            return
        sign_name = signs[0]
        logger.info(f"Selected sign: {sign_name}")
    
    sign_dir = os.path.join(data_dir, sign_name)
    if not os.path.exists(sign_dir):
        logger.error(f"Sign directory not found: {sign_dir}")
        return
    
    # Find landmark files
    landmark_files = [f for f in os.listdir(sign_dir) if f.endswith('.npy') and not f.endswith('_metadata.json')]
    if not landmark_files:
        logger.error(f"No landmark files found in {sign_dir}")
        return
    
    if source_name is None:
        # Pick first available source
        source_name = landmark_files[0].replace('.npy', '')
        logger.info(f"Selected source: {source_name}")
    
    landmark_path = os.path.join(sign_dir, f"{source_name}.npy")
    metadata_path = os.path.join(sign_dir, f"{source_name}_metadata.json")
    
    if not os.path.exists(landmark_path):
        logger.error(f"Landmark file not found: {landmark_path}")
        return
    
    # Load data
    landmarks = visualizer.load_landmarks(landmark_path)
    metadata = visualizer.load_metadata(metadata_path)
    
    logger.info(f"Loaded landmarks: {landmarks.shape}")
    logger.info(f"Metadata: {metadata}")
    
    # Create visualizations
    # 1. First frame visualization
    first_frame = visualizer.visualize_frame(landmarks, 0, f"{sign_name} - {source_name}")
    if first_frame is not None:
        first_frame_path = os.path.join(output_dir, f"{sign_name}_{source_name}_frame0.png")
        cv2.imwrite(first_frame_path, first_frame)
        logger.info(f"First frame saved to: {first_frame_path}")
    
    # 2. Animation
    animation_path = os.path.join(output_dir, f"{sign_name}_{source_name}_animation.mp4")
    visualizer.create_animation(landmarks, animation_path)
    
    # 3. Confidence plot
    if metadata:
        confidence_path = os.path.join(output_dir, f"{sign_name}_{source_name}_confidence.png")
        visualizer.plot_confidence_over_time(landmarks, metadata, confidence_path)
    
    # 4. Check for augmented versions
    augmented_files = [f for f in landmark_files if 'aug_' in f]
    if augmented_files:
        logger.info(f"Found {len(augmented_files)} augmented versions")
        
        # Compare with first augmented version
        aug_file = augmented_files[0]
        aug_path = os.path.join(sign_dir, aug_file)
        
        comparison_dir = os.path.join(output_dir, f"{sign_name}_{source_name}_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        visualizer.compare_original_vs_augmented(landmark_path, aug_path, comparison_dir)

def main():
    """Main function for landmark visualization."""
    parser = argparse.ArgumentParser(description='Landmark Visualization Tool')
    parser.add_argument('--data-path', choices=['processed', 'train', 'test'], default='processed',
                       help='Which dataset to visualize')
    parser.add_argument('--sign-name', type=str, help='Specific sign to visualize')
    parser.add_argument('--source-name', type=str, help='Specific source to visualize')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    logger.info("🎨 Starting landmark visualization")
    logger.info(f"Arguments: {vars(args)}")
    
    visualize_dataset_sample(
        data_path=args.data_path,
        sign_name=args.sign_name,
        source_name=args.source_name,
        output_dir=args.output_dir
    )
    
    logger.info("✅ Visualization complete!")

if __name__ == '__main__':
    main() 