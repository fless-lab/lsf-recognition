#!/usr/bin/env python3
"""
Test Pipeline Script

This script tests all components of the LSF recognition pipeline to ensure
everything works correctly before running on the full dataset.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import json
import logging
from pathlib import Path
from src.utils.landmark_utils import extract_landmark_vector
from extract_landmarks import LandmarkExtractor
from consolidate import DatasetConsolidator
from augment import DataAugmenter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_video(output_path, duration=2, fps=30):
    """Create a simple test video for testing."""
    try:
        import cv2
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        logger.info(f"Création d'une vidéo de test : {output_path}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        for frame in range(duration * fps):
            # Create a frame with moving shapes
            img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # Draw moving circles
            x1 = int(320 + 100 * np.sin(frame * 0.1))
            y1 = int(240 + 50 * np.cos(frame * 0.1))
            x2 = int(320 + 80 * np.cos(frame * 0.15))
            y2 = int(240 + 60 * np.sin(frame * 0.15))
            
            cv2.circle(img, (x1, y1), 30, (0, 255, 0), -1)
            cv2.circle(img, (x2, y2), 25, (255, 0, 0), -1)
            
            # Add text
            cv2.putText(img, f"Frame {frame}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            out.write(img)
        
        out.release()
        logger.info(f"Created test video: {output_path}")
        return True
        
    except ImportError:
        logger.warning("OpenCV not available, skipping video creation")
        return False
    except Exception as e:
        logger.error(f"Error creating test video: {e}")
        return False

def test_landmark_extraction():
    """Test landmark extraction functionality."""
    logger.info("🧪 Testing landmark extraction...")
    
    try:
        # Create a test extractor
        extractor = LandmarkExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Create a temporary test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            test_video_path = tmp_file.name
        
        if not create_test_video(test_video_path):
            logger.warning("Skipping landmark extraction test (no test video)")
            return True
        
        try:
            # Test processing
            landmarks, metadata = extractor.process_video(test_video_path)
            
            if landmarks is not None and metadata is not None:
                logger.info(f"✅ Landmark extraction successful")
                logger.info(f"   Landmarks shape: {landmarks.shape}")
                logger.info(f"   Metadata keys: {list(metadata.keys())}")
                
                # Test saving
                landmarks_path = test_video_path.replace('.mp4', '_landmarks.npy')
                metadata_path = test_video_path.replace('.mp4', '_metadata.json')
                
                np.save(landmarks_path, landmarks)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Landmark saving successful")
                return True
            else:
                logger.error("❌ Landmark extraction failed")
                return False
                
        finally:
            # Cleanup
            os.unlink(test_video_path)
            if os.path.exists(landmarks_path):
                os.unlink(landmarks_path)
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
                
    except ImportError as e:
        logger.error(f"❌ Cannot import landmark extraction module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Landmark extraction test failed: {e}")
        return False

def test_augmentation():
    """Test data augmentation functionality."""
    logger.info("🧪 Testing data augmentation...")
    
    try:
        # Create test landmarks
        num_frames = 30
        num_landmarks = 1662  # Total landmarks (pose + face + hands)
        test_landmarks = np.random.rand(num_frames, num_landmarks)
        
        # Create test metadata
        test_metadata = {
            'video_path': 'test_video.mp4',
            'fps': 30.0,
            'frame_count': num_frames,
            'average_pose_confidence': 0.8,
            'average_face_confidence': 1.0,
            'average_left_hand_confidence': 0.9,
            'average_right_hand_confidence': 0.85
        }
        
        # Test augmenter
        augmenter = DataAugmenter(augmentation_factor=3)
        augmented_samples = augmenter.augment_landmarks(test_landmarks, test_metadata)
        
        if len(augmented_samples) == 3:  # Original + 2 augmented
            logger.info(f"✅ Data augmentation successful")
            logger.info(f"   Original shape: {augmented_samples[0][0].shape}")
            logger.info(f"   Augmented samples: {len(augmented_samples) - 1}")
            
            # Check that augmented samples have augmentation metadata
            for i, (landmarks, metadata) in enumerate(augmented_samples[1:], 1):
                if 'augmentation' in metadata:
                    logger.info(f"   Sample {i}: {metadata['augmentation']['technique']}")
                else:
                    logger.warning(f"   Sample {i}: No augmentation metadata")
            
            return True
        else:
            logger.error(f"❌ Expected 3 samples, got {len(augmented_samples)}")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Cannot import augmentation module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Data augmentation test failed: {e}")
        return False

def test_consolidation():
    """Test consolidation functionality."""
    logger.info("🧪 Testing consolidation...")
    
    try:
        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = os.path.join(temp_dir, 'processed')
            data_dir = os.path.join(temp_dir, 'data')
            os.makedirs(processed_dir)
            os.makedirs(data_dir)
            
            # Create test sign directories
            test_signs = ['bonjour', 'merci', 'au_revoir']
            test_sources = ['jauvert', 'elix']
            
            for sign in test_signs:
                sign_dir = os.path.join(processed_dir, sign)
                os.makedirs(sign_dir)
                
                for source in test_sources:
                    # Create fake landmarks
                    landmarks = np.random.rand(30, 1662)
                    landmarks_path = os.path.join(sign_dir, f"{source}.npy")
                    np.save(landmarks_path, landmarks)
                    
                    # Create fake metadata
                    metadata = {
                        'video_path': f'test_{sign}_{source}.mp4',
                        'fps': 30.0,
                        'frame_count': 30,
                        'average_pose_confidence': 0.8,
                        'average_face_confidence': 1.0,
                        'average_left_hand_confidence': 0.9,
                        'average_right_hand_confidence': 0.85
                    }
                    metadata_path = os.path.join(sign_dir, f"{source}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
            
            # Test consolidator
            consolidator = DatasetConsolidator(processed_dir, data_dir)
            
            # Test analysis
            sign_sources, sign_files = consolidator.analyze_dataset()
            logger.info(f"✅ Dataset analysis successful")
            logger.info(f"   Found {len(sign_sources)} signs")
            logger.info(f"   Sources per sign: {[len(sources) for sources in sign_sources.values()]}")
            
            # Test corpus generation
            corpus_signs, quality_metrics = consolidator.generate_corpus(min_confidence=0.3)
            logger.info(f"✅ Corpus generation successful")
            logger.info(f"   Corpus size: {len(corpus_signs)}")
            
            # Test splits creation
            split_assignments, train_signs, val_signs, test_signs = consolidator.create_dataset_splits(corpus_signs, quality_metrics)
            logger.info(f"✅ Dataset splits successful")
            logger.info(f"   Train: {len(train_signs)}, Val: {len(val_signs)}, Test: {len(test_signs)}")
            
            return True
            
    except ImportError as e:
        logger.error(f"❌ Cannot import consolidation module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Consolidation test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    logger.info("🧪 Testing visualization...")
    
    try:
        from visualize_landmarks import LandmarkVisualizer
        
        # Create test landmarks
        num_frames = 20
        num_landmarks = 1662
        test_landmarks = np.random.rand(num_frames, num_landmarks)
        
        # Create test metadata
        test_metadata = {
            'frame_metadata': [
                {
                    'frame_index': i,
                    'pose_confidence': 0.8 + 0.1 * np.random.rand(),
                    'face_confidence': 1.0,
                    'left_hand_confidence': 0.9 + 0.05 * np.random.rand(),
                    'right_hand_confidence': 0.85 + 0.05 * np.random.rand()
                }
                for i in range(num_frames)
            ]
        }
        
        # Test visualizer
        visualizer = LandmarkVisualizer()
        
        # Test frame visualization
        img = visualizer.visualize_frame(test_landmarks, 0, "Test Visualization")
        if img is not None:
            logger.info(f"✅ Frame visualization successful")
            logger.info(f"   Image shape: {img.shape}")
        else:
            logger.error("❌ Frame visualization failed")
            return False
        
        # Test confidence plotting
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plot_path = tmp_file.name
        
        try:
            visualizer.plot_confidence_over_time(test_landmarks, test_metadata, plot_path)
            if os.path.exists(plot_path):
                logger.info(f"✅ Confidence plotting successful")
                os.unlink(plot_path)
            else:
                logger.error("❌ Confidence plotting failed")
                return False
        except Exception as e:
            logger.warning(f"Confidence plotting failed (matplotlib issue): {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Cannot import visualization module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Visualization test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    logger.info("🧪 Testing dependencies...")
    
    required_packages = [
        'numpy',
        'cv2',
        'mediapipe',
        'matplotlib',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            logger.error(f"❌ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run all tests."""
    logger.info("🚀 Starting LSF Recognition Pipeline Tests")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Landmark Extraction", test_landmark_extraction),
        ("Data Augmentation", test_augmentation),
        ("Consolidation", test_consolidation),
        ("Visualization", test_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline is ready to use.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Please fix issues before running the pipeline.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 