#!/usr/bin/env python3
"""
Script de lancement simple pour le pipeline LSF.
Usage: python run_pipeline.py [--force-reprocess] [--augmentation-factor N]
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_processing'))

try:
    from pipeline_complet import LSFDataPipeline
except ImportError:
    print("❌ Erreur: Impossible d'importer le pipeline. Vérifiez que tous les fichiers sont présents.")
    sys.exit(1)

def main():
    """Lance le pipeline complet avec des paramètres par défaut."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline LSF - Traitement complet des données')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Forcer le retraitement des vidéos déjà traitées')
    parser.add_argument('--augmentation-factor', type=int, default=15,
                       help='Facteur d\'augmentation (défaut: 15)')
    
    args = parser.parse_args()
    
    # Détecter la racine du projet
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 Lancement du pipeline LSF...")
    print(f"Racine du projet: {project_root}")
    print(f"Force reprocess: {args.force_reprocess}")
    print(f"Facteur d'augmentation: {args.augmentation_factor}")
    print("=" * 60)
    
    # Initialiser et lancer le pipeline
    pipeline = LSFDataPipeline(project_root)
    success = pipeline.run_complete_pipeline(
        force_reprocess=args.force_reprocess,
        augmentation_factor=args.augmentation_factor
    )
    
    if success:
        print("\n🎉 Pipeline terminé avec succès!")
        print("📁 Les données finales sont dans data/final_train, data/final_val, data/final_test")
        print("📄 Le corpus est dans data/corpus.txt")
        return 0
    else:
        print("\n💥 Pipeline échoué!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 