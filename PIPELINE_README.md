# 🚀 Pipeline Complet LSF - Guide d'Utilisation

## 📋 Vue d'ensemble

Ce pipeline complet traite automatiquement vos vidéos LSF pour créer un dataset robuste et généralisable pour l'entraînement d'un modèle de reconnaissance de langue des signes française.

## 🎯 Fonctionnalités

### ✅ **Extraction Avancée des Landmarks**
- Utilise MediaPipe Holistic pour extraire les landmarks du corps, visage et mains
- Préserve les métadonnées (confiance, FPS, dimensions)
- Gestion intelligente des erreurs et reprise automatique

### ✅ **Consolidation Intelligente**
- Analyse automatique des sources multiples (jauvert, elix, education-nationale)
- Séparation par source pour éviter le "data leakage"
- Filtrage par qualité des landmarks extraits
- Génération automatique du corpus

### ✅ **Augmentation Sophistiquée**
- **Spatiale** : rotation, translation, mise à l'échelle, perspective
- **Temporelle** : warping temporel, bruit temporel
- **Occlusion** : simulation d'occlusions partielles
- **Mixup** : mélange entre séquences similaires
- **15 versions augmentées** par échantillon original

### ✅ **Séparation Optimale Train/Val/Test**
- **Signes multi-sources** : séparation stricte par source
- **Signes uniques** : augmentation maximale pour enrichir le train
- **Évite le data leakage** pour une évaluation fiable

## 🛠️ Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### Lancement Simple
```bash
# Pipeline complet avec paramètres par défaut
python run_pipeline.py

# Avec facteur d'augmentation personnalisé
python run_pipeline.py --augmentation-factor 20

# Forcer le retraitement des vidéos déjà traitées
python run_pipeline.py --force-reprocess
```

### Lancement Avancé
```bash
# Utiliser le script principal directement
python src/data_processing/pipeline_complet.py --augmentation-factor 15 --force-reprocess
```

## 📁 Structure des Données

### Entrée (Vidéos Sources)
```
data/raw/
├── parlr/
│   ├── jauvert/          # ~100 vidéos .webm
│   ├── elix/             # ~100 vidéos .webm
│   └── education-nationale/  # ~100 vidéos .webm
└── custom/               # Vidéos personnalisées
```

### Sortie (Dataset Final)
```
data/
├── corpus.txt                    # Liste des signes
├── quality_metrics.json          # Métriques de qualité
├── split_assignments.json        # Assignations train/val/test
├── dataset_statistics.json       # Statistiques finales
├── final_train/                  # Données d'entraînement augmentées
├── final_val/                    # Données de validation
└── final_test/                   # Données de test
```

## ⚙️ Paramètres Configurables

### Facteur d'Augmentation
- **Défaut** : 15 versions par échantillon original
- **Recommandé** : 10-20 selon la puissance de calcul disponible
- **Impact** : Plus d'augmentation = meilleure généralisation mais temps de traitement plus long

### Seuil de Qualité
- **Défaut** : 0.3 (30% de confiance minimale)
- **Impact** : Filtre les landmarks de mauvaise qualité

### Séparation Train/Val/Test
- **Train** : 70% des sources + données augmentées
- **Validation** : 10% des sources
- **Test** : 20% des sources

## 📊 Métriques de Qualité

Le pipeline génère automatiquement :

1. **Corpus** : Liste des signes de haute qualité
2. **Métriques par signe** :
   - Confiance moyenne des landmarks
   - Nombre de sources
   - Nombre d'échantillons
3. **Statistiques par split** :
   - Nombre de signes
   - Nombre total d'échantillons
   - Distribution par signe

## 🔧 Techniques d'Augmentation

### Spatiales
- **Bruit gaussien** : ±0.01 sur les coordonnées
- **Translation** : ±0.05 en x, y, z
- **Rotation** : ±15° autour de l'axe Y
- **Mise à l'échelle** : ±10% uniforme
- **Perspective** : Simulation d'angles de vue

### Temporelles
- **Warping temporel** : ±20% de variation de vitesse
- **Bruit temporel** : Variations fluides dans le temps
- **Occlusion partielle** : 30% de chance d'occlusion par frame

### Avancées
- **Mixup** : Mélange entre séquences similaires
- **Combinaisons aléatoires** : Application probabiliste des augmentations

## ⚡ Performance

### Temps Estimés (sur CPU standard)
- **Extraction** : ~2-3 secondes par vidéo
- **Consolidation** : ~30 secondes
- **Augmentation** : ~1-2 secondes par échantillon
- **Total** : ~2-4 heures pour 1000+ vidéos

### Optimisations
- **Reprise automatique** : Le pipeline reprend où il s'est arrêté
- **Traitement parallèle** : Possibilité d'ajouter du multiprocessing
- **Mémoire optimisée** : Traitement par batch pour éviter l'overflow

## 🐛 Dépannage

### Erreurs Courantes

1. **"Could not open video"**
   - Vérifiez que les vidéos sont dans le bon format (.webm, .mp4, .avi, .mov)
   - Vérifiez les permissions de lecture

2. **"No landmarks extracted"**
   - Vidéo de mauvaise qualité ou trop courte
   - Personne non visible ou trop loin de la caméra

3. **"Import error"**
   - Vérifiez que toutes les dépendances sont installées
   - Vérifiez que tous les fichiers du pipeline sont présents

### Logs
- **pipeline.log** : Logs détaillés du traitement
- **Console** : Progression en temps réel

## 🎯 Résultat Final

Après exécution du pipeline, vous aurez :

1. **Dataset robuste** : ~15,000+ échantillons augmentés
2. **Séparation propre** : Train/val/test sans data leakage
3. **Corpus optimisé** : Signes de haute qualité uniquement
4. **Métadonnées complètes** : Pour analyse et debug

## 🚀 Prochaines Étapes

1. **Entraînement du modèle** : Utiliser `data/final_train/` et `data/final_val/`
2. **Évaluation** : Tester sur `data/final_test/`
3. **Analyse** : Consulter `dataset_statistics.json` pour les métriques

---

**🎉 Votre dataset est maintenant prêt pour entraîner un modèle robuste et généralisable !** 