# Recycle-moi Backend

Backend pour la classification de déchets par deep learning.

## 📊 Résultats

- **Modèle** : ResNet18 fine-tuné
- **Test Accuracy** : 83.55%
- **Classes** : 7 catégories de déchets (cardboard, e-waste, glass, medical, metal, paper, plastic)
- **Dataset** : 17,853 images

## 🚀 Installation

### Prérequis

- Python 3.11+
- CUDA 12.1+ (pour GPU)
- 6GB+ RAM GPU (recommandé)

### Setup
```bash
# Créer environnement conda
conda create -n recyclemoi python=3.11 -y
conda activate recyclemoi

# Installer dépendances
pip install -r requirements.txt
```

## 📁 Structure du Projet
```
backend-api/
├── src/
│   ├── config/          # Configuration
│   ├── data/            # Dataset et transformations
│   ├── models/          # Architectures de modèles
│   ├── training/        # Entraînement et évaluation
│   └── utils/           # Utilitaires
├── scripts/             # Scripts d'exécution
├── checkpoints/         # Modèles sauvegardés
├── logs/                # Logs d'entraînement
└── tests/               # Tests unitaires
```

## 🎓 Entraînement

### Entraînement basique
```bash
python scripts/train.py
```

### Avec options personnalisées
```bash
python scripts/train.py \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.0001 \
  --checkpoint-dir checkpoints/v2.0
```

### Options disponibles

- `--epochs` : Nombre d'epochs
- `--batch-size` : Taille du batch
- `--lr` : Learning rate
- `--device` : cuda ou cpu
- `--no-pretrained` : Ne pas utiliser les poids ImageNet
- `--freeze-backbone` : Geler les couches convolutionnelles
- `--seed` : Seed pour reproductibilité

## 🧪 Évaluation

### Évaluation basique
```bash
python scripts/evaluate.py --checkpoint checkpoints/v1.0/best_model.pth
```

### Avec matrice de confusion
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/v1.0/best_model.pth \
  --save-confusion-matrix \
  --output-dir results/v1.0
```

## ⚙️ Configuration

Tous les hyperparamètres sont dans `src/config/config.yaml`.

Sections principales :
- `data` : Dataset, classes, normalisation
- `model` : Architecture, poids pré-entraînés
- `training` : Batch size, learning rate, epochs
- `hardware` : Device, num_workers

## 📊 Métriques

Le modèle génère :
- Accuracy globale
- Rapport de classification (precision, recall, f1-score par classe)
- Matrice de confusion
- Métriques par classe

## 🔧 Développement

### Lancer les tests
```bash
pytest tests/
```

### Structure d'un nouveau module

1. Créer le fichier dans `src/`
2. Ajouter `__init__.py` si nouveau dossier
3. Importer dans les scripts si nécessaire
4. Documenter avec docstrings


## 👨‍💻 Auteur

Bahani vissiley thierry - [LinkedIn](https://www.linkedin.com/in/bahanivissiley)

## 📄 License

MIT License