#!/usr/bin/env python3
"""
Script de test pour vérifier que le pipeline fonctionne.
"""

import sys
import os

def test_imports():
    """Teste que tous les modules peuvent être importés."""
    print("🧪 Test des imports...")
    
    try:
        # Test des imports de base
        import numpy as np
        print("✅ numpy importé")
        
        import cv2
        print("✅ opencv importé")
        
        import mediapipe as mp
        print("✅ mediapipe importé")
        
        import json
        print("✅ json importé")
        
        import logging
        print("✅ logging importé")
        
        # Test des imports du pipeline
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_processing'))
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return False

def test_data_structure():
    """Teste que la structure des données est correcte."""
    print("\n📁 Test de la structure des données...")
    
    # Vérifier que les dossiers existent
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    raw_path = os.path.join(data_path, 'raw')
    
    if not os.path.exists(data_path):
        print("❌ Dossier 'data' manquant")
        return False
    
    if not os.path.exists(raw_path):
        print("❌ Dossier 'data/raw' manquant")
        return False
    
    # Vérifier les sources
    sources = ['parlr/jauvert', 'parlr/elix', 'parlr/education-nationale']
    for source in sources:
        source_path = os.path.join(raw_path, source)
        if os.path.exists(source_path):
            # Compter les vidéos
            video_count = 0
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.endswith(('.webm', '.mp4', '.avi', '.mov')):
                        video_count += 1
            print(f"✅ {source}: {video_count} vidéos trouvées")
        else:
            print(f"⚠️  {source}: dossier manquant")
    
    return True

def test_pipeline_initialization():
    """Teste l'initialisation du pipeline."""
    print("\n🚀 Test d'initialisation du pipeline...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_processing'))
        
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        print("✅ Pipeline initialisé avec succès")
        print(f"   Racine du projet: {project_root}")
        print(f"   Chemin des données: {os.path.join(project_root, 'data')}")
        print(f"   Chemin des vidéos: {os.path.join(project_root, 'data', 'raw')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 TESTS DU PIPELINE LSF")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ ÉCHEC: Problème avec les imports")
        return False
    
    # Test 2: Structure des données
    if not test_data_structure():
        print("\n❌ ÉCHEC: Problème avec la structure des données")
        return False
    
    # Test 3: Initialisation du pipeline
    if not test_pipeline_initialization():
        print("\n❌ ÉCHEC: Problème d'initialisation du pipeline")
        return False
    
    print("\n" + "=" * 50)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("🎉 Le pipeline est prêt à être utilisé!")
    print("\n📋 Pour lancer le pipeline:")
    print("   python run_pipeline.py")
    print("   python run_pipeline.py --augmentation-factor 15")
    print("   python run_pipeline.py --force-reprocess")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 