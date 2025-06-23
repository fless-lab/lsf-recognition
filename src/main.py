#!/usr/bin/env python3
"""
Pipeline principal LSF Recognition (robuste, modulaire, extensible)
Usage :
    python src/main.py --model-type siamese --augmentation-factor 0 --until demo
    python src/main.py --model-type classic --until eval
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

# Modèles qui n'ont PAS besoin d'augmentation
NO_AUGMENT_MODELS = {'siamese', 'prototypical', 'metric', 'matching'}
# Mapping model type -> (train_script, eval_script, demo_script)
MODEL_SCRIPTS = {
    'classic': {
        'train': 'src/models/classic/train_model.py',
        'eval': 'src/models/classic/evaluate_model.py',
        'demo': None
    },
    'siamese': {
        'train': 'src/models/fewshot/siamese/train_siamese.py',
        'eval': 'src/models/fewshot/siamese/eval_oneshot_cross_source.py',
        'demo': os.path.abspath('demo/app_streamlit.py')
    },
    'prototypical': {
        'train': 'src/models/fewshot/prototypical/train_prototypical.py',
        'eval': None,
        'demo': None
    },
    'metric': {
        'train': 'src/models/fewshot/metric/train_metric.py',
        'eval': None,
        'demo': None
    },
    'matching': {
        'train': 'src/models/fewshot/matching/train_matching.py',
        'eval': None,
        'demo': None
    }
}

STEPS = ['extraction', 'consolidation', 'augmentation', 'training', 'eval', 'demo']

# Correction : rendre tous les chemins de scripts absolus
for model, scripts in MODEL_SCRIPTS.items():
    for key, rel_path in scripts.items():
        if rel_path:
            scripts[key] = os.path.abspath(rel_path)

def run_script(script_path, description, extra_args=None):
    """Run a Python script and handle errors."""
    logger.info(f"Starting: {description}")
    logger.info(f"Running: {script_path}")
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd += extra_args
    # Ajout du PYTHONPATH pour garantir les imports
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = project_root + (':' + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(script_path), env=env)
        if result.returncode == 0:
            logger.info(f"✅ Completed: {description}")
        else:
            logger.error(f"❌ Failed: {description}")
        return result.returncode == 0
    except Exception as e:
        logger.error(f"❌ Exception in {description}: {str(e)}")
        return False


def check_prerequisites():
    """Check if all required directories and files exist."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    data_path = os.path.join(project_root, 'data')
    raw_path = os.path.join(data_path, 'raw')
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        return False
    if not os.path.exists(raw_path):
        logger.error(f"Raw data directory not found: {raw_path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='LSF Recognition Data Pipeline (robuste)')
    parser.add_argument('--model-type', type=str, default='siamese', choices=list(MODEL_SCRIPTS.keys()),
                        help='Type de modèle à entraîner (classic, siamese, prototypical, metric, matching)')
    parser.add_argument('--until', type=str, default='demo', choices=STEPS,
                        help='Arrêter le pipeline à cette étape (extraction, consolidation, augmentation, training, eval, demo)')
    parser.add_argument('--skip-extraction', action='store_true', help='Sauter l\'étape d\'extraction des landmarks')
    parser.add_argument('--skip-consolidation', action='store_true', help='Sauter l\'étape de consolidation/split')
    parser.add_argument('--skip-augmentation', action='store_true', help='Sauter l\'étape d\'augmentation des données')
    parser.add_argument('--augmentation-factor', type=int, default=5, help='Facteur d\'augmentation (défaut : 5)')
    parser.add_argument('--no-demo', action='store_true', help='Ne pas lancer la démo Streamlit à la fin')
    parser.add_argument('--force-reprocess', action='store_true', help='Forcer le retraitement des vidéos déjà traitées')
    args, unknown = parser.parse_known_args()

    logger.info("🚀 Lancement du pipeline LSF Recognition (robuste)")
    logger.info(f"Arguments: {vars(args)}")

    # Vérification des prérequis
    if not check_prerequisites():
        logger.error("❌ Prérequis manquants. Arrêt.")
        return 1

    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    scripts_dir = os.path.join(project_root, 'src', 'data_processing')

    extraction_script = os.path.join(scripts_dir, 'extract_landmarks.py')
    consolidation_script = os.path.join(scripts_dir, 'consolidate.py')
    augmentation_script = os.path.join(scripts_dir, 'augment.py')

    # Orchestration dynamique
    step_idx = STEPS.index(args.until)
    steps_to_run = STEPS[:step_idx+1]

    # Extraction
    if 'extraction' in steps_to_run and not args.skip_extraction:
        if not run_script(extraction_script, "Landmark Extraction", extra_args=(['--force-reprocess'] if args.force_reprocess else [])):
            logger.error("❌ Landmark extraction failed. Exiting.")
            return 1
    else:
        logger.info("⏭️ Skipping landmark extraction")
    if args.until == 'extraction':
        return 0

    # Consolidation
    if 'consolidation' in steps_to_run and not args.skip_consolidation:
        if not run_script(consolidation_script, "Dataset Consolidation and Splitting"):
            logger.error("❌ Consolidation failed. Exiting.")
            return 1
    else:
        logger.info("⏭️ Skipping consolidation")
    if args.until == 'consolidation':
        return 0

    # Augmentation (skippée si modèle few-shot ou flag)
    if 'augmentation' in steps_to_run:
        if args.model_type in NO_AUGMENT_MODELS or args.skip_augmentation or args.augmentation_factor == 0:
            logger.info(f"⏭️ Skipping augmentation for model type {args.model_type}")
        else:
            if not run_script(augmentation_script, "Data Augmentation", extra_args=[f'--augmentation-factor={args.augmentation_factor}']):
                logger.error("❌ Data augmentation failed. Exiting.")
                return 1
    if args.until == 'augmentation':
        return 0

    # Entraînement
    train_script = MODEL_SCRIPTS[args.model_type]['train']
    if 'training' in steps_to_run:
        if not run_script(train_script, f"Model Training ({args.model_type})"):
            logger.error("❌ Model training failed. Exiting.")
            return 1
    if args.until == 'training':
        return 0

    # Évaluation
    eval_script = MODEL_SCRIPTS[args.model_type]['eval']
    if 'eval' in steps_to_run and eval_script:
        if not run_script(eval_script, f"Model Evaluation ({args.model_type})"):
            logger.error("❌ Model evaluation failed. Exiting.")
            return 1
    elif 'eval' in steps_to_run:
        logger.info(f"⏭️ No evaluation script for model type {args.model_type}")
    if args.until == 'eval':
        return 0

    # Démo Streamlit
    demo_script = MODEL_SCRIPTS[args.model_type]['demo']
    if 'demo' in steps_to_run and demo_script and not args.no_demo:
        logger.info(f"🚀 Lancement de la démo Streamlit pour le modèle {args.model_type}...")
        # Lancer Streamlit en sous-processus
        try:
            subprocess.run([sys.executable, demo_script])
        except Exception as e:
            logger.error(f"❌ Erreur lors du lancement de la démo : {e}")
    elif 'demo' in steps_to_run:
        logger.info(f"⏭️ No demo available for model type {args.model_type} or demo skipped.")

    logger.info("🎉 Pipeline complet terminé !")
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 