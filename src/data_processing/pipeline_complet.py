#!/usr/bin/env python3
"""
Pipeline complet pour le traitement des données LSF.
Ce script orchestre tout le processus : extraction, consolidation, augmentation et préparation.
"""

import os
import sys
import logging
import time
import json
import numpy as np
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from extract_landmarks_advanced import LandmarkExtractor
from consolidate_advanced import DatasetConsolidator
from augment_advanced import AdvancedAugmenter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSFDataPipeline:
    def __init__(self, project_root):
        self.project_root = project_root
        self.data_path = os.path.join(project_root, 'data')
        
        # Define all paths
        self.raw_path = os.path.join(self.data_path, 'raw')
        self.processed_path = os.path.join(self.data_path, 'processed')
        self.consolidated_path = os.path.join(self.data_path, 'consolidated')
        self.train_path = os.path.join(self.data_path, 'train')
        self.val_path = os.path.join(self.data_path, 'val')
        self.test_path = os.path.join(self.data_path, 'test')
        self.augmented_train_path = os.path.join(self.data_path, 'train_augmented')
        
        # Create necessary directories
        for path in [self.processed_path, self.consolidated_path, 
                    self.train_path, self.val_path, self.test_path, 
                    self.augmented_train_path]:
            os.makedirs(path, exist_ok=True)
    
    def step1_extract_landmarks(self, force_reprocess=False):
        """Étape 1: Extraction des landmarks depuis les vidéos."""
        logger.info("=" * 60)
        logger.info("ÉTAPE 1: EXTRACTION DES LANDMARKS")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Initialize extractor
        extractor = LandmarkExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Process all sources
        sources = ['parlr/jauvert', 'parlr/elix', 'parlr/education-nationale', 'custom']
        
        total_processed = 0
        total_errors = 0
        
        for source in sources:
            source_path = os.path.join(self.raw_path, source)
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
                output_dir = os.path.join(self.processed_path, source.replace('/', '_'), sign_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create output filenames
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                landmarks_file = os.path.join(output_dir, f"{base_name}_landmarks.npy")
                metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
                
                # Skip if already processed and not forcing reprocess
                if not force_reprocess and os.path.exists(landmarks_file) and os.path.exists(metadata_file):
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
        
        elapsed_time = time.time() - start_time
        logger.info(f"Extraction complete in {elapsed_time:.2f} seconds")
        logger.info(f"Processed: {total_processed}, Errors: {total_errors}")
        
        return total_processed, total_errors
    
    def step2_consolidate_and_split(self):
        """Étape 2: Consolidation et séparation intelligente des données."""
        logger.info("=" * 60)
        logger.info("ÉTAPE 2: CONSOLIDATION ET SÉPARATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Initialize consolidator
        consolidator = DatasetConsolidator(self.processed_path, self.consolidated_path)
        
        # Analyze dataset
        logger.info("Analyzing dataset structure...")
        sign_sources, source_signs, sign_files = consolidator.analyze_dataset()
        
        # Generate corpus with quality filtering
        logger.info("Generating corpus with quality filtering...")
        corpus_signs, quality_metrics = consolidator.generate_corpus(min_confidence=0.3)
        
        # Create consolidated structure
        logger.info("Creating consolidated structure...")
        consolidator.create_consolidated_structure(corpus_signs, quality_metrics)
        
        # Create dataset splits
        logger.info("Creating dataset splits with source separation...")
        split_assignments = consolidator.create_dataset_splits(corpus_signs)
        
        # Save corpus and metadata
        corpus_file = os.path.join(self.data_path, 'corpus.txt')
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for sign in corpus_signs:
                f.write(f"{sign}\n")
        
        # Save quality metrics
        metrics_file = os.path.join(self.data_path, 'quality_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(quality_metrics, f, indent=2, ensure_ascii=False)
        
        # Save split assignments
        splits_file = os.path.join(self.data_path, 'split_assignments.json')
        with open(splits_file, 'w', encoding='utf-8') as f:
            json.dump(split_assignments, f, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Consolidation complete in {elapsed_time:.2f} seconds")
        logger.info(f"Corpus saved to: {corpus_file}")
        logger.info(f"Quality metrics saved to: {metrics_file}")
        logger.info(f"Split assignments saved to: {splits_file}")
        
        return corpus_signs, quality_metrics, split_assignments
    
    def step3_augment_data(self, augmentation_factor=15):
        """Étape 3: Augmentation avancée des données d'entraînement."""
        logger.info("=" * 60)
        logger.info("ÉTAPE 3: AUGMENTATION AVANCÉE")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Initialize augmenter
        augmenter = AdvancedAugmenter(augmentation_factor=augmentation_factor)
        
        # Run augmentation on training data
        logger.info(f"Starting augmentation with factor {augmentation_factor}...")
        augmenter.augment_dataset(self.train_path, self.augmented_train_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Augmentation complete in {elapsed_time:.2f} seconds")
        
        return True
    
    def step4_generate_final_dataset(self):
        """Étape 4: Génération du dataset final pour l'entraînement."""
        logger.info("=" * 60)
        logger.info("ÉTAPE 4: GÉNÉRATION DU DATASET FINAL")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Load corpus
        corpus_file = os.path.join(self.data_path, 'corpus.txt')
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_signs = [line.strip() for line in f.readlines()]
        
        # Create final dataset structure
        final_train_path = os.path.join(self.data_path, 'final_train')
        final_val_path = os.path.join(self.data_path, 'final_val')
        final_test_path = os.path.join(self.data_path, 'final_test')
        
        for path in [final_train_path, final_val_path, final_test_path]:
            os.makedirs(path, exist_ok=True)
        
        # Copy augmented training data
        logger.info("Copying augmented training data...")
        import shutil
        for sign_name in corpus_signs:
            # Copy from augmented train
            aug_sign_dir = os.path.join(self.augmented_train_path, sign_name)
            if os.path.exists(aug_sign_dir):
                final_sign_dir = os.path.join(final_train_path, sign_name)
                if os.path.exists(final_sign_dir):
                    shutil.rmtree(final_sign_dir)
                shutil.copytree(aug_sign_dir, final_sign_dir)
        
        # Copy validation data
        logger.info("Copying validation data...")
        for sign_name in corpus_signs:
            val_sign_dir = os.path.join(self.val_path, sign_name)
            if os.path.exists(val_sign_dir):
                final_sign_dir = os.path.join(final_val_path, sign_name)
                if os.path.exists(final_sign_dir):
                    shutil.rmtree(final_sign_dir)
                shutil.copytree(val_sign_dir, final_sign_dir)
        
        # Copy test data
        logger.info("Copying test data...")
        for sign_name in corpus_signs:
            test_sign_dir = os.path.join(self.test_path, sign_name)
            if os.path.exists(test_sign_dir):
                final_sign_dir = os.path.join(final_test_path, sign_name)
                if os.path.exists(final_sign_dir):
                    shutil.rmtree(final_sign_dir)
                shutil.copytree(test_sign_dir, final_sign_dir)
        
        # Generate dataset statistics
        stats = self.generate_dataset_statistics(corpus_signs)
        
        # Save statistics
        stats_file = os.path.join(self.data_path, 'dataset_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Final dataset generation complete in {elapsed_time:.2f} seconds")
        logger.info(f"Dataset statistics saved to: {stats_file}")
        
        return stats
    
    def generate_dataset_statistics(self, corpus_signs):
        """Generate comprehensive dataset statistics."""
        stats = {
            'corpus_size': len(corpus_signs),
            'corpus_signs': corpus_signs,
            'splits': {}
        }
        
        splits = {
            'train': self.data_path + '/final_train',
            'val': self.data_path + '/final_val', 
            'test': self.data_path + '/final_test'
        }
        
        for split_name, split_path in splits.items():
            split_stats = {
                'num_signs': 0,
                'total_samples': 0,
                'samples_per_sign': {}
            }
            
            if os.path.exists(split_path):
                for sign_name in corpus_signs:
                    sign_dir = os.path.join(split_path, sign_name)
                    if os.path.exists(sign_dir):
                        num_samples = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
                        if num_samples > 0:
                            split_stats['num_signs'] += 1
                            split_stats['total_samples'] += num_samples
                            split_stats['samples_per_sign'][sign_name] = num_samples
            
            stats['splits'][split_name] = split_stats
        
        return stats
    
    def run_complete_pipeline(self, force_reprocess=False, augmentation_factor=15):
        """Exécute le pipeline complet."""
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET LSF")
        logger.info("=" * 80)
        
        total_start_time = time.time()
        
        try:
            # Étape 1: Extraction des landmarks
            processed, errors = self.step1_extract_landmarks(force_reprocess)
            
            # Étape 2: Consolidation et séparation
            corpus_signs, quality_metrics, split_assignments = self.step2_consolidate_and_split()
            
            # Étape 3: Augmentation
            self.step3_augment_data(augmentation_factor)
            
            # Étape 4: Dataset final
            stats = self.step4_generate_final_dataset()
            
            total_elapsed = time.time() - total_start_time
            
            logger.info("=" * 80)
            logger.info("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
            logger.info("=" * 80)
            logger.info(f"Temps total: {total_elapsed:.2f} secondes")
            logger.info(f"Signes dans le corpus: {len(corpus_signs)}")
            logger.info(f"Échantillons traités: {processed}")
            logger.info(f"Erreurs: {errors}")
            logger.info(f"Facteur d'augmentation: {augmentation_factor}")
            
            # Afficher les statistiques finales
            logger.info("\n📊 STATISTIQUES FINALES:")
            for split_name, split_stats in stats['splits'].items():
                logger.info(f"  {split_name.upper()}: {split_stats['num_signs']} signes, {split_stats['total_samples']} échantillons")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERREUR DANS LE PIPELINE: {str(e)}")
            return False

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline complet pour le traitement des données LSF')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Forcer le retraitement des vidéos déjà traitées')
    parser.add_argument('--augmentation-factor', type=int, default=15,
                       help='Facteur d\'augmentation (nombre de versions par échantillon original)')
    parser.add_argument('--project-root', type=str, default=None,
                       help='Racine du projet (défaut: détection automatique)')
    
    args = parser.parse_args()
    
    # Detect project root if not provided
    if args.project_root is None:
        script_path = os.path.abspath(__file__)
        args.project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    
    # Initialize and run pipeline
    pipeline = LSFDataPipeline(args.project_root)
    success = pipeline.run_complete_pipeline(
        force_reprocess=args.force_reprocess,
        augmentation_factor=args.augmentation_factor
    )
    
    if success:
        logger.info("🎉 Pipeline terminé avec succès!")
        sys.exit(0)
    else:
        logger.error("💥 Pipeline échoué!")
        sys.exit(1)

if __name__ == '__main__':
    main() 