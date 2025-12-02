# Recycle-moi Model v1.0

Premier modèle production-ready. ResNet18 avec Transfer Learning + Fine-tuning. Entraîné sur 17,853 images de déchets.

## Métriques

- **Test Accuracy**: 83.55%
- **Validation Accuracy**: 84.95%
- **Architecture**: ResNet18 (Transfer Learning + Fine-tuning)
- **Dataset**: 17,853 images (7 classes)

## Métriques par Classe

- **cardboard**: 79.34%
- **e-waste**: 92.05%
- **glass**: 82.28%
- **medical**: 85.28%
- **metal**: 84.62%
- **paper**: 81.85%
- **plastic**: 78.36%


## Fichiers

- `best_model.pth`: Poids du modèle
- `metadata.json`: Métadonnées complètes
- `config.yaml`: Configuration utilisée
- `classification_report.txt`: Rapport détaillé
- `confusion_matrix.png`: Matrice de confusion

## Utilisation
```python
from src.models.resnet import load_model

model = load_model('checkpoints/v1.0/best_model.pth')
```

## Date de création

2025-12-02 13:42:29
