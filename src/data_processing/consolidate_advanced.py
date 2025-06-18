import os
import shutil
import json
import numpy as np
from collections import defaultdict, Counter
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetConsolidator:
    def __init__(self, processed_path, consolidated_path):
        self.processed_path = processed_path
        self.consolidated_path = consolidated_path
        self.sign_sources = defaultdict(list)  # sign_name -> list of sources
        self.source_signs = defaultdict(list)  # source -> list of signs
        self.sign_files = defaultdict(list)    # sign_name -> list of file paths
        
    def analyze_dataset(self):
        """Analyze the processed dataset to understand the distribution."""
        logger.info("Analyzing dataset structure...")
        
        # Walk through processed directory
        for source_folder in os.listdir(self.processed_path):
            source_path = os.path.join(self.processed_path, source_folder)
            if not os.path.isdir(source_path):
                continue
                
            for sign_folder in os.listdir(source_path):
                sign_path = os.path.join(source_path, sign_folder)
                if not os.path.isdir(sign_path):
                    continue
                    
                # Check if this sign has landmark files
                landmark_files = [f for f in os.listdir(sign_path) if f.endswith('_landmarks.npy')]
                if landmark_files:
                    sign_name = sign_folder
                    source_name = source_folder
                    
                    # Track sign-source relationships
                    self.sign_sources[sign_name].append(source_name)
                    self.source_signs[source_name].append(sign_name)
                    
                    # Track files for this sign
                    for file in landmark_files:
                        file_path = os.path.join(sign_path, file)
                        self.sign_files[sign_name].append({
                            'path': file_path,
                            'source': source_name,
                            'filename': file
                        })
        
        # Analyze distribution
        logger.info(f"Found {len(self.sign_sources)} unique signs")
        logger.info(f"Found {len(self.source_signs)} sources")
        
        # Count signs by number of sources
        source_counts = Counter([len(sources) for sources in self.sign_sources.values()])
        logger.info("Sign distribution by number of sources:")
        for count, num_signs in source_counts.most_common():
            logger.info(f"  {count} source(s): {num_signs} signs")
            
        return self.sign_sources, self.source_signs, self.sign_files
    
    def generate_corpus(self, min_confidence=0.3):
        """Generate corpus with quality filtering."""
        logger.info("Generating corpus with quality filtering...")
        
        corpus_signs = []
        quality_metrics = {}
        
        for sign_name, files in self.sign_files.items():
            # Check quality of all files for this sign
            sign_quality = []
            
            for file_info in files:
                metadata_path = file_info['path'].replace('_landmarks.npy', '_metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Calculate quality score
                    avg_pose_conf = metadata.get('average_pose_confidence', 0)
                    avg_lh_conf = metadata.get('average_left_hand_confidence', 0)
                    avg_rh_conf = metadata.get('average_right_hand_confidence', 0)
                    
                    # Hand confidence is most important for sign language
                    hand_confidence = max(avg_lh_conf, avg_rh_conf)
                    quality_score = (avg_pose_conf + hand_confidence) / 2
                    
                    sign_quality.append(quality_score)
                else:
                    sign_quality.append(0.0)
            
            # Only include signs with at least one good quality sample
            if max(sign_quality) >= min_confidence:
                corpus_signs.append(sign_name)
                quality_metrics[sign_name] = {
                    'avg_quality': np.mean(sign_quality),
                    'max_quality': max(sign_quality),
                    'num_sources': len(self.sign_sources[sign_name]),
                    'num_files': len(files)
                }
        
        # Sort by quality and number of sources
        corpus_signs.sort(key=lambda x: (quality_metrics[x]['num_sources'], quality_metrics[x]['avg_quality']), reverse=True)
        
        logger.info(f"Generated corpus with {len(corpus_signs)} high-quality signs")
        
        return corpus_signs, quality_metrics
    
    def create_consolidated_structure(self, corpus_signs, quality_metrics):
        """Create consolidated directory structure with intelligent separation."""
        logger.info("Creating consolidated directory structure...")
        
        # Clean up previous consolidated data
        if os.path.exists(self.consolidated_path):
            shutil.rmtree(self.consolidated_path)
        os.makedirs(self.consolidated_path)
        
        # Create directories for each sign
        for sign_name in corpus_signs:
            sign_dir = os.path.join(self.consolidated_path, sign_name)
            os.makedirs(sign_dir, exist_ok=True)
            
            # Copy files with source information in filename
            for file_info in self.sign_files[sign_name]:
                source = file_info['source']
                original_filename = file_info['filename']
                base_name = original_filename.replace('_landmarks.npy', '')
                
                # Create new filename with source information
                new_filename = f"{base_name}_{source}.npy"
                new_path = os.path.join(sign_dir, new_filename)
                
                # Copy the file
                shutil.copy2(file_info['path'], new_path)
                
                # Also copy metadata
                metadata_path = file_info['path'].replace('_landmarks.npy', '_metadata.json')
                if os.path.exists(metadata_path):
                    new_metadata_path = os.path.join(sign_dir, f"{base_name}_{source}_metadata.json")
                    shutil.copy2(metadata_path, new_metadata_path)
        
        logger.info(f"Created consolidated structure for {len(corpus_signs)} signs")
    
    def create_dataset_splits(self, corpus_signs, test_ratio=0.2, val_ratio=0.1):
        """Create train/val/test splits with intelligent source separation."""
        logger.info("Creating dataset splits with source separation...")
        
        # Create split directories
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = os.path.join(self.consolidated_path, f'..', split)
            os.makedirs(split_dir, exist_ok=True)
        
        split_assignments = {}
        
        for sign_name in corpus_signs:
            sources = self.sign_sources[sign_name]
            num_sources = len(sources)
            
            if num_sources >= 2:
                # Multiple sources: separate by source
                if num_sources == 2:
                    # 2 sources: one for train, one for test
                    split_assignments[sign_name] = {
                        sources[0]: 'train',
                        sources[1]: 'test'
                    }
                elif num_sources == 3:
                    # 3 sources: train, val, test
                    split_assignments[sign_name] = {
                        sources[0]: 'train',
                        sources[1]: 'val',
                        sources[2]: 'test'
                    }
                else:
                    # More than 3 sources: distribute proportionally
                    train_sources = sources[:int(num_sources * (1 - test_ratio - val_ratio))]
                    val_sources = sources[int(num_sources * (1 - test_ratio - val_ratio)):int(num_sources * (1 - test_ratio))]
                    test_sources = sources[int(num_sources * (1 - test_ratio)):]
                    
                    split_assignments[sign_name] = {}
                    for source in train_sources:
                        split_assignments[sign_name][source] = 'train'
                    for source in val_sources:
                        split_assignments[sign_name][source] = 'val'
                    for source in test_sources:
                        split_assignments[sign_name][source] = 'test'
            else:
                # Single source: split files within the source
                split_assignments[sign_name] = {sources[0]: 'train'}  # Will be split later
        
        # Copy files to appropriate split directories
        for sign_name, source_splits in split_assignments.items():
            for source, split in source_splits.items():
                split_dir = os.path.join(self.consolidated_path, f'..', split, sign_name)
                os.makedirs(split_dir, exist_ok=True)
                
                # Copy files from this source
                for file_info in self.sign_files[sign_name]:
                    if file_info['source'] == source:
                        original_filename = file_info['filename']
                        base_name = original_filename.replace('_landmarks.npy', '')
                        new_filename = f"{base_name}_{source}.npy"
                        
                        dest_path = os.path.join(split_dir, new_filename)
                        shutil.copy2(file_info['path'], dest_path)
                        
                        # Copy metadata
                        metadata_path = file_info['path'].replace('_landmarks.npy', '_metadata.json')
                        if os.path.exists(metadata_path):
                            new_metadata_path = os.path.join(split_dir, f"{base_name}_{source}_metadata.json")
                            shutil.copy2(metadata_path, new_metadata_path)
        
        logger.info("Dataset splits created successfully")
        return split_assignments

def main():
    """Main function to consolidate the dataset with intelligent source separation."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    
    processed_path = os.path.join(data_path, 'processed')
    consolidated_path = os.path.join(data_path, 'consolidated')
    
    # Initialize consolidator
    consolidator = DatasetConsolidator(processed_path, consolidated_path)
    
    # Analyze dataset
    sign_sources, source_signs, sign_files = consolidator.analyze_dataset()
    
    # Generate corpus with quality filtering
    corpus_signs, quality_metrics = consolidator.generate_corpus(min_confidence=0.3)
    
    # Create consolidated structure
    consolidator.create_consolidated_structure(corpus_signs, quality_metrics)
    
    # Create dataset splits
    split_assignments = consolidator.create_dataset_splits(corpus_signs)
    
    # Save corpus and metadata
    corpus_file = os.path.join(data_path, 'corpus.txt')
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for sign in corpus_signs:
            f.write(f"{sign}\n")
    
    # Save quality metrics
    metrics_file = os.path.join(data_path, 'quality_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(quality_metrics, f, indent=2, ensure_ascii=False)
    
    # Save split assignments
    splits_file = os.path.join(data_path, 'split_assignments.json')
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(split_assignments, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Consolidation complete!")
    logger.info(f"Corpus saved to: {corpus_file}")
    logger.info(f"Quality metrics saved to: {metrics_file}")
    logger.info(f"Split assignments saved to: {splits_file}")

if __name__ == '__main__':
    main() 