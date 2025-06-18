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
    def __init__(self, processed_path, data_path):
        logger.info(f"Initialisation du consolidateur pour {processed_path} -> {data_path}")
        self.processed_path = processed_path
        self.data_path = data_path
        self.sign_sources = defaultdict(list)  # sign_name -> list of sources
        self.sign_files = defaultdict(list)    # sign_name -> list of file paths
        
    def analyze_dataset(self):
        logger.info("Analyse du dataset processed...")
        """Analyze the processed dataset to understand the distribution."""
        
        # Walk through processed directory
        for sign_folder in os.listdir(self.processed_path):
            sign_path = os.path.join(self.processed_path, sign_folder)
            if not os.path.isdir(sign_path):
                continue
                
            # Check if this sign has landmark files
            landmark_files = [f for f in os.listdir(sign_path) if f.endswith('.npy') and not f.endswith('_metadata.json')]
            if landmark_files:
                sign_name = sign_folder
                
                # Track sources for this sign
                for file in landmark_files:
                    source_name = file.replace('.npy', '')
                    if source_name not in self.sign_sources[sign_name]:
                        self.sign_sources[sign_name].append(source_name)
                    
                    file_path = os.path.join(sign_path, file)
                    self.sign_files[sign_name].append({
                        'path': file_path,
                        'source': source_name,
                        'filename': file
                    })
        
        # Analyze distribution
        logger.info(f"Analyse terminée : {len(self.sign_sources)} signes trouvés.")
        
        # Count signs by number of sources
        source_counts = Counter([len(sources) for sources in self.sign_sources.values()])
        logger.info("Sign distribution by number of sources:")
        for count, num_signs in source_counts.most_common():
            logger.info(f"  {count} source(s): {num_signs} signs")
            
        return self.sign_sources, self.sign_files
    
    def generate_corpus(self, min_confidence=0.3):
        logger.info(f"Génération du corpus avec filtrage qualité (min_conf={min_confidence})...")
        """Generate corpus with quality filtering."""
        
        corpus_signs = []
        quality_metrics = {}
        
        for sign_name, files in self.sign_files.items():
            # Check quality of all files for this sign
            sign_quality = []
            
            for file_info in files:
                metadata_path = file_info['path'].replace('.npy', '_metadata.json')
                
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
                    'num_files': len(files),
                    'sources': self.sign_sources[sign_name]
                }
        
        # Sort by quality and number of sources
        corpus_signs.sort(key=lambda x: (quality_metrics[x]['num_sources'], quality_metrics[x]['avg_quality']), reverse=True)
        
        logger.info(f"Corpus généré : {len(corpus_signs)} signes de qualité.")
        
        return corpus_signs, quality_metrics
    
    def create_dataset_splits(self, corpus_signs, quality_metrics):
        logger.info("Création des splits train/val/test...")
        """Create train/val/test splits with intelligent source separation."""
        
        # Create split directories
        train_path = os.path.join(self.data_path, 'train')
        val_path = os.path.join(self.data_path, 'val')
        test_path = os.path.join(self.data_path, 'test')
        
        for path in [train_path, val_path, test_path]:
            os.makedirs(path, exist_ok=True)
        
        split_assignments = {}
        train_signs = []
        test_signs = []
        val_signs = []
        
        for sign_name in corpus_signs:
            sources = self.sign_sources[sign_name]
            num_sources = len(sources)
            
            if num_sources >= 2:
                # Multiple sources: separate by source
                if num_sources == 2:
                    # 2 sources: one for train, one for test
                    train_source = sources[0]
                    test_source = sources[1]
                    
                    split_assignments[sign_name] = {
                        'train': [train_source],
                        'test': [test_source],
                        'val': []
                    }
                    
                    # Copy files to appropriate directories
                    self._copy_sign_to_split(sign_name, train_source, train_path)
                    self._copy_sign_to_split(sign_name, test_source, test_path)
                    
                    train_signs.append(sign_name)
                    test_signs.append(sign_name)
                    
                elif num_sources == 3:
                    # 3 sources: train, val, test
                    train_source = sources[0]
                    val_source = sources[1]
                    test_source = sources[2]
                    
                    split_assignments[sign_name] = {
                        'train': [train_source],
                        'val': [val_source],
                        'test': [test_source]
                    }
                    
                    # Copy files to appropriate directories
                    self._copy_sign_to_split(sign_name, train_source, train_path)
                    self._copy_sign_to_split(sign_name, val_source, val_path)
                    self._copy_sign_to_split(sign_name, test_source, test_path)
                    
                    train_signs.append(sign_name)
                    val_signs.append(sign_name)
                    test_signs.append(sign_name)
                    
            else:
                # Single source: all in train (no test for this sign)
                train_source = sources[0]
                
                split_assignments[sign_name] = {
                    'train': [train_source],
                    'val': [],
                    'test': []
                }
                
                # Copy file to train directory
                self._copy_sign_to_split(sign_name, train_source, train_path)
                
                train_signs.append(sign_name)
        
        logger.info(f"Splits terminés : train={len(train_signs)}, val={len(val_signs)}, test={len(test_signs)}")
        
        return split_assignments, train_signs, val_signs, test_signs
    
    def _copy_sign_to_split(self, sign_name, source_name, split_path):
        logger.info(f"Copie du signe {sign_name} ({source_name}) vers {split_path}")
        """Copy a sign from a specific source to a split directory."""
        sign_dir = os.path.join(split_path, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Copy landmark file
        source_file = os.path.join(self.processed_path, sign_name, f"{source_name}.npy")
        dest_file = os.path.join(sign_dir, f"{source_name}.npy")
        shutil.copy2(source_file, dest_file)
        
        # Copy metadata file
        source_metadata = os.path.join(self.processed_path, sign_name, f"{source_name}_metadata.json")
        dest_metadata = os.path.join(sign_dir, f"{source_name}_metadata.json")
        if os.path.exists(source_metadata):
            shutil.copy2(source_metadata, dest_metadata)

def main():
    logger.info("=== DÉBUT Consolidation & split ===")
    """Main function to consolidate the dataset with intelligent source separation."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    data_path = os.path.join(project_root, 'data')
    
    processed_path = os.path.join(data_path, 'processed')
    
    # Initialize consolidator
    consolidator = DatasetConsolidator(processed_path, data_path)
    
    # Analyze dataset
    sign_sources, sign_files = consolidator.analyze_dataset()
    
    # Generate corpus with quality filtering
    corpus_signs, quality_metrics = consolidator.generate_corpus(min_confidence=0.3)
    
    # Create dataset splits
    split_assignments, train_signs, val_signs, test_signs = consolidator.create_dataset_splits(corpus_signs, quality_metrics)
    
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
    
    # Save split statistics
    split_stats = {
        'train_signs': train_signs,
        'val_signs': val_signs,
        'test_signs': test_signs,
        'total_signs': len(corpus_signs)
    } 