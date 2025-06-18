# Pipeline de Reconnaissance LSF (Langue des Signes Française)

Ce pipeline complet permet de traiter des vidéos de langue des signes française pour créer un dataset optimisé pour l'entraînement de modèles de reconnaissance.

## 🎯 Objectif

Transformer des vidéos brutes de signes LSF en un dataset structuré avec :
- Extraction de landmarks 3D (pose, visage, mains) via MediaPipe Holistic
- Séparation intelligente train/validation/test par source
- Augmentation de données sophistiquée
- Structure optimisée pour l'entraînement

## 📁 Structure du Projet

```
lsf-recognition/
├── data/
│   ├── raw/                          # Vidéos brutes
│   │   ├── parlr/jauvert/           # Source 1
│   │   ├── parlr/elix/              # Source 2
│   │   ├── parlr/education-nationale/ # Source 3
│   │   └── custom/                   # Source 4
│   ├── processed/                    # Landmarks extraits
│   │   ├── bonjour/
│   │   │   ├── jauvert.npy
│   │   │   ├── jauvert_metadata.json
│   │   │   ├── elix.npy
│   │   │   └── elix_metadata.json
│   │   └── ...
│   ├── train/                        # Données d'entraînement (avec augmentation)
│   ├── val/                          # Données de validation
│   ├── test/                         # Données de test
│   └── corpus.txt                    # Liste des signes
├── src/
│   └── data_processing/
│       ├── extract_landmarks.py
│       ├── consolidate.py
│       ├── augment.py
│       ├── visualize_landmarks.py
│       └── run_pipeline.py
└── visualizations/                   # Visualisations générées
```

## 🚀 Utilisation Rapide

### 1. Préparation des données

Placez vos vidéos dans la structure suivante :
```
data/raw/
├── parlr/jauvert/
│   ├── bonjour.webm
│   ├── merci.webm
│   └── ...
├── parlr/elix/
│   ├── bonjour.webm
│   ├── au_revoir.webm
│   └── ...
└── ...
```

### 2. Exécution du pipeline complet

```bash
# Pipeline complet
python src/data_processing/run_pipeline.py

# Ou par étapes
python src/data_processing/run_pipeline.py --skip-extraction      # Si landmarks déjà extraits
python src/data_processing/run_pipeline.py --skip-consolidation   # Si splits déjà créés
python src/data_processing/run_pipeline.py --skip-augmentation    # Si pas d'augmentation
```

### 3. Visualisation des résultats

```bash
# Visualiser un échantillon
python src/data_processing/visualize_landmarks.py --data-path processed

# Visualiser des données spécifiques
python src/data_processing/visualize_landmarks.py --data-path train --sign-name bonjour --source-name jauvert
```

## 🔧 Étapes du Pipeline

### 1. Extraction des Landmarks (`extract_landmarks.py`)

**Objectif** : Extraire les landmarks 3D de chaque vidéo

**Fonctionnalités** :
- Utilise MediaPipe Holistic pour détecter pose, visage, mains
- Sauvegarde landmarks + métadonnées (confiance, FPS, etc.)
- Structure : `processed/{sign}/{source}.npy`
- Analyse automatique de la distribution des sources

**Format des landmarks** :
```python
# Shape: (num_frames, total_landmarks)
# Structure:
# - Pose: 33 points × 4 (x, y, z, visibility)
# - Face: 468 points × 3 (x, y, z)
# - Main gauche: 21 points × 3 (x, y, z)
# - Main droite: 21 points × 3 (x, y, z)
# Total: 33*4 + 468*3 + 21*3 + 21*3 = 1662 dimensions
```

### 2. Consolidation et Splits (`consolidate.py`)

**Objectif** : Créer les splits train/val/test avec séparation par source

**Stratégie de séparation** :
- **Signes multi-sources** :
  - 2 sources : 1 train, 1 test
  - 3 sources : 1 train, 1 val, 1 test
- **Signes mono-source** : Tout en train (pas de test)
- **Filtrage qualité** : Seuil de confiance minimum

**Avantages** :
- Évite le data leakage entre sources
- Test sur sources non vues pendant l'entraînement
- Évaluation plus réaliste de la généralisation

### 3. Augmentation des Données (`augment.py`)

**Objectif** : Générer des versions augmentées pour l'entraînement

**Techniques d'augmentation** :
- **Spatiale** : Rotation, échelle, translation
- **Temporelle** : Variation de vitesse, suppression de frames
- **Occlusion** : Simulation d'occlusion partielle
- **Perspective** : Changement de point de vue
- **Mixup** : Mélange avec version bruitée

**Configuration** :
- 5 versions augmentées par original (configurable)
- Seulement sur les données d'entraînement
- Métadonnées préservées avec info d'augmentation

## 📊 Métadonnées et Qualité

### Métadonnées par vidéo
```json
{
  "video_path": "path/to/video.webm",
  "fps": 30.0,
  "frame_count": 90,
  "width": 1920,
  "height": 1080,
  "extracted_frames": 90,
  "average_pose_confidence": 0.85,
  "average_face_confidence": 1.0,
  "average_left_hand_confidence": 0.92,
  "average_right_hand_confidence": 0.88,
  "frame_metadata": [...]
}
```

### Métadonnées par frame
```json
{
  "frame_index": 0,
  "pose_confidence": 0.87,
  "face_confidence": 1.0,
  "left_hand_confidence": 0.95,
  "right_hand_confidence": 0.91
}
```

## 🎨 Visualisation

Le script `visualize_landmarks.py` permet de :

1. **Visualiser un frame** : Points colorés par type (pose, visage, mains)
2. **Créer des animations** : Vidéos MP4 des landmarks
3. **Analyser la confiance** : Graphiques de confiance dans le temps
4. **Comparer original vs augmenté** : Side-by-side des versions

### Utilisation
```bash
# Visualiser un échantillon automatique
python src/data_processing/visualize_landmarks.py

# Visualiser des données spécifiques
python src/data_processing/visualize_landmarks.py \
  --data-path train \
  --sign-name bonjour \
  --source-name jauvert \
  --output-dir ./my_visualizations
```

## ⚙️ Configuration

### Paramètres d'extraction
```python
# Dans extract_landmarks.py
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1  # 0, 1, ou 2
```

### Paramètres de consolidation
```python
# Dans consolidate.py
min_confidence = 0.3  # Seuil de qualité minimum
```

### Paramètres d'augmentation
```python
# Dans augment.py
augmentation_factor = 5  # Nombre de versions augmentées
```

## 📈 Statistiques du Dataset

Le pipeline génère automatiquement des statistiques :

- **Distribution des sources** : Nombre de signes par source
- **Qualité des landmarks** : Scores de confiance moyens
- **Splits** : Nombre de signes par split
- **Augmentation** : Nombre de versions générées

## 🔍 Dépannage

### Problèmes courants

1. **MediaPipe non installé** :
   ```bash
   pip install mediapipe
   ```

2. **Vidéos non trouvées** :
   - Vérifiez la structure des dossiers
   - Formats supportés : `.webm`, `.mp4`, `.avi`, `.mov`

3. **Mémoire insuffisante** :
   - Réduisez `model_complexity` dans MediaPipe
   - Traitez par petits lots

4. **Erreurs de landmarks** :
   - Vérifiez la qualité des vidéos
   - Augmentez `min_detection_confidence`

### Logs et monitoring

- **Logs** : `pipeline.log` dans le répertoire d'exécution
- **Progression** : Affichage en temps réel
- **Erreurs** : Détails complets dans les logs

## 🎯 Prochaines Étapes

1. **Entraînement du modèle** : Utiliser les données générées
2. **Évaluation** : Tester sur le split test
3. **Amélioration** : Ajuster les paramètres selon les résultats
4. **Déploiement** : Intégrer dans une application

## 📝 Notes Techniques

- **Performance** : Extraction ~1-2s par vidéo (selon la durée)
- **Stockage** : ~1-5MB par vidéo (selon le nombre de frames)
- **Compatibilité** : Python 3.8+, Linux/macOS/Windows
- **Dépendances** : Voir `requirements.txt`

---

**Pipeline développé pour la reconnaissance LSF avec focus sur la robustesse et la généralisation.** 