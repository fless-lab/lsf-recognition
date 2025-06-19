# Few-Shot Learning Models for LSF Recognition

Ce dossier contient des architectures et scripts pour l'apprentissage few-shot/one-shot appliqué à la reconnaissance de la LSF, adaptés aux cas où il n'y a qu'un ou très peu d'exemples par classe.

## Sous-dossiers et scripts

- `siamese/` : Réseau siamois (similarité entre séquences)
- `prototypical/` : Prototypical Networks (prototypes par classe)
- `matching/` : Matching Networks (comparaison par attention)
- `metric/` : Metric Learning (triplet loss, etc.)

Chaque sous-dossier contient :
- Un script d'entraînement complet (`train_*.py`)
- Un script d'inférence/démo (`infer_*.py`)
- Un README spécifique
- Les checkpoints de modèles sauvegardés

## Utilisation rapide

1. Placez vos données dans `data/processed` (format : un .npy par séquence, par classe)
2. Lancez le script d'entraînement de l'architecture souhaitée
3. Utilisez le script d'inférence pour tester la reconnaissance sur de nouveaux exemples

Voir chaque sous-dossier pour plus de détails et d'exemples de code. 