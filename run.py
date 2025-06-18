#!/usr/bin/env python3
"""
Script de lancement principal pour le pipeline LSF.
Usage : python run.py [--force-reprocess] [--augmentation-factor N] [--skip-extraction] [--skip-consolidation] [--skip-augmentation]
"""

import sys
import os

# Ajouter le dossier data_processing au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_processing'))

def main():
    """Lance le pipeline complet avec les paramètres utilisateur."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline LSF - Traitement complet des données')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Forcer le retraitement des vidéos déjà traitées')
    parser.add_argument('--augmentation-factor', type=int, default=5,
                       help='Facteur d\'augmentation (défaut : 5)')
    parser.add_argument('--skip-extraction', action='store_true', help='Sauter l\'étape d\'extraction des landmarks')
    parser.add_argument('--skip-consolidation', action='store_true', help='Sauter l\'étape de consolidation/split')
    parser.add_argument('--skip-augmentation', action='store_true', help='Sauter l\'étape d\'augmentation des données')
    
    args = parser.parse_args()
    
    # Affichage utilisateur AVANT d'appeler le pipeline
    project_root = os.path.dirname(os.path.abspath(__file__))
    print("\n================ LSF PIPELINE ==================")
    print("🚀 Lancement du pipeline LSF...")
    print(f"Racine du projet : {project_root}")
    print(f"Force reprocess : {args.force_reprocess}")
    print(f"Facteur d'augmentation : {args.augmentation_factor}")
    print(f"Skip extraction : {args.skip_extraction}")
    print(f"Skip consolidation : {args.skip_consolidation}")
    print(f"Skip augmentation : {args.skip_augmentation}")
    print("=" * 48)

    # Appel du vrai orchestrateur (qui gère les arguments via sys.argv)
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from main import main as pipeline_main
    exit_code = pipeline_main()

    # Affichage utilisateur APRÈS le pipeline
    if exit_code == 0:
        print("\n🎉 Pipeline terminé avec succès !")
        print("📁 Les données finales sont dans data/train, data/val, data/test")
        print("📄 Le corpus est dans data/corpus.txt")
    else:
        print("\n💥 Le pipeline a rencontré une erreur (voir logs).")
    print("=" * 48)
    return exit_code

if __name__ == '__main__':
    sys.exit(main()) 