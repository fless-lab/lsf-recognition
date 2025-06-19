#!/usr/bin/env python3
"""
Script de test pour vérifier que le training fonctionne sans planter.
"""

import os
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Teste les imports nécessaires."""
    try:
        logger.info("Testing imports...")
        import numpy as np
        import tensorflow as tf
        import psutil
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import error: {e}")
        return False

def test_memory():
    """Teste la surveillance mémoire."""
    try:
        logger.info("Testing memory monitoring...")
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"✓ Current memory usage: {memory_mb:.1f} MB")
        return True
    except Exception as e:
        logger.error(f"✗ Memory monitoring error: {e}")
        return False

def test_data_loading():
    """Teste le chargement des données."""
    try:
        logger.info("Testing data loading...")
        import numpy as np
        
        # Trouver le répertoire de données
        script_path = os.path.abspath(__file__)
        project_root = os.path.dirname(script_path)
        data_path = os.path.join(project_root, 'data', 'train')
        
        if not os.path.exists(data_path):
            logger.error(f"✗ Data directory not found: {data_path}")
            return False
        
        # Compter les fichiers
        total_files = 0
        for sign_folder in os.listdir(data_path):
            sign_path = os.path.join(data_path, sign_folder)
            if os.path.isdir(sign_path):
                npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
                total_files += len(npy_files)
        
        logger.info(f"✓ Found {total_files} .npy files in training data")
        
        # Tester le chargement d'un seul fichier
        for sign_folder in os.listdir(data_path):
            sign_path = os.path.join(data_path, sign_folder)
            if os.path.isdir(sign_path):
                npy_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
                if npy_files:
                    test_file = os.path.join(sign_path, npy_files[0])
                    data = np.load(test_file)
                    logger.info(f"✓ Successfully loaded {test_file} with shape {data.shape}")
                    break
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loading error: {e}")
        return False

def test_model_creation():
    """Teste la création du modèle."""
    try:
        logger.info("Testing model creation...")
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
        
        # Créer un petit modèle de test
        model = Sequential([
            Masking(mask_value=0., input_shape=(100, 33)),
            LSTM(16, return_sequences=True),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("✓ Model created successfully")
        logger.info(f"✓ Model parameters: {model.count_params()}")
        return True
    except Exception as e:
        logger.error(f"✗ Model creation error: {e}")
        return False

def main():
    """Fonction principale de test."""
    logger.info("Starting training tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Memory monitoring", test_memory),
        ("Data loading", test_data_loading),
        ("Model creation", test_model_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✓ {test_name} PASSED")
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("✓ All tests passed! Training should work.")
        return True
    else:
        logger.error("✗ Some tests failed. Check the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 