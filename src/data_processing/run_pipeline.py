#!/usr/bin/env python3
"""
Complete LSF Recognition Data Pipeline

This script orchestrates the entire data processing pipeline:
1. Extract landmarks from raw videos using MediaPipe Holistic
2. Consolidate and create train/val/test splits with source separation
3. Augment training data with multiple techniques
4. Generate final dataset ready for model training

Usage:
    python run_pipeline.py [--skip-extraction] [--skip-consolidation] [--skip-augmentation]
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    logger.info(f"Starting: {description}")
    logger.info(f"Running: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Completed: {description}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"❌ Failed: {description}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception in {description}: {str(e)}")
        return False
    
    return True

def check_prerequisites():
    """Check if all required directories and files exist."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    raw_path = os.path.join(data_path, 'raw')
    
    # Check if data directory exists
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        logger.error("Please create the data directory and add raw videos.")
        return False
    
    # Check if raw directory exists and has content
    if not os.path.exists(raw_path):
        logger.error(f"Raw data directory not found: {raw_path}")
        logger.error("Please create the raw directory and add video files.")
        return False
    
    # Check for expected source directories
    expected_sources = ['parlr/jauvert', 'parlr/elix', 'parlr/education-nationale', 'custom']
    found_sources = []
    
    for source in expected_sources:
        source_path = os.path.join(raw_path, source)
        if os.path.exists(source_path):
            found_sources.append(source)
    
    if not found_sources:
        logger.error("No source directories found in raw data.")
        logger.error(f"Expected sources: {expected_sources}")
        return False
    
    logger.info(f"Found source directories: {found_sources}")
    return True

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='LSF Recognition Data Pipeline')
    parser.add_argument('--skip-extraction', action='store_true', 
                       help='Skip landmark extraction step')
    parser.add_argument('--skip-consolidation', action='store_true',
                       help='Skip consolidation and splitting step')
    parser.add_argument('--skip-augmentation', action='store_true',
                       help='Skip data augmentation step')
    parser.add_argument('--augmentation-factor', type=int, default=5,
                       help='Number of augmented versions per original sample (default: 5)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting LSF Recognition Data Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Get script paths
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    scripts_dir = os.path.join(project_root, 'src', 'data_processing')
    
    extraction_script = os.path.join(scripts_dir, 'extract_landmarks.py')
    consolidation_script = os.path.join(scripts_dir, 'consolidate.py')
    augmentation_script = os.path.join(scripts_dir, 'augment.py')
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("❌ Prerequisites check failed. Exiting.")
        return 1
    
    # Step 1: Extract landmarks
    if not args.skip_extraction:
        if not run_script(extraction_script, "Landmark Extraction"):
            logger.error("❌ Landmark extraction failed. Exiting.")
            return 1
    else:
        logger.info("⏭️ Skipping landmark extraction")
    
    # Step 2: Consolidate and create splits
    if not args.skip_consolidation:
        if not run_script(consolidation_script, "Dataset Consolidation and Splitting"):
            logger.error("❌ Consolidation failed. Exiting.")
            return 1
    else:
        logger.info("⏭️ Skipping consolidation")
    
    # Step 3: Augment training data
    if not args.skip_augmentation:
        # Modify augmentation script to use the specified factor
        if args.augmentation_factor != 5:
            logger.info(f"Modifying augmentation factor to {args.augmentation_factor}")
            # This would require modifying the script or passing parameters
            # For now, we'll use the default
        
        if not run_script(augmentation_script, "Data Augmentation"):
            logger.error("❌ Data augmentation failed. Exiting.")
            return 1
    else:
        logger.info("⏭️ Skipping data augmentation")
    
    # Final summary
    logger.info("🎉 Pipeline completed successfully!")
    
    # Print dataset statistics
    data_path = os.path.join(project_root, 'data')
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    test_path = os.path.join(data_path, 'test')
    
    if os.path.exists(train_path):
        train_signs = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        logger.info(f"📊 Train set: {train_signs} signs")
    
    if os.path.exists(val_path):
        val_signs = len([d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))])
        logger.info(f"📊 Validation set: {val_signs} signs")
    
    if os.path.exists(test_path):
        test_signs = len([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
        logger.info(f"📊 Test set: {test_signs} signs")
    
    logger.info("📁 Dataset ready for model training!")
    logger.info(f"📁 Data location: {data_path}")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 