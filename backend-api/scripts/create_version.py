"""
Script pour créer une version du modèle avec métadonnées complètes
Usage: python scripts/create_version.py --checkpoint checkpoints/v1.0/best_model.pth --version 1.0
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import argparse
from datetime import datetime
import json
import shutil

from src.config import config
from src.models.resnet import load_model, create_resnet18
from src.utils.helpers import count_parameters

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Créer une version du modèle avec métadonnées')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--version', type=str, required=True,
                       help='Numéro de version (ex: 1.0, 1.1, 2.0)')
    parser.add_argument('--description', type=str, default='',
                       help='Description de cette version')
    parser.add_argument('--test-acc', type=float, required=True,
                       help='Test accuracy de ce modèle')
    parser.add_argument('--valid-acc', type=float, required=True,
                       help='Validation accuracy de ce modèle')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Dossier de sortie (défaut: checkpoints/v{version})')
    
    return parser.parse_args()

def create_metadata(args, params_info, checkpoint_info):
    """
    Crée le dictionnaire de métadonnées complet
    
    Args:
        args: Arguments du script
        params_info: Info sur les paramètres du modèle
        checkpoint_info: Info depuis le checkpoint
        
    Returns:
        Dictionnaire de métadonnées
    """
    metadata = {
        # Version
        'version': args.version,
        'created_at': datetime.now().isoformat(),
        'description': args.description,
        
        # Project info
        'project': {
            'name': config.get('project.name', 'Recycle-moi'),
            'version': config.get('project.version', '1.0.0'),
            'description': config.get('project.description', '')
        },
        
        # Model architecture
        'model': {
            'architecture': 'resnet18',
            'pretrained': True,
            'freeze_backbone': False,
            'num_classes': config.get('data.num_classes', 7),
            'parameters': {
                'total': params_info['total'],
                'trainable': params_info['trainable'],
                'frozen': params_info['frozen']
            }
        },
        
        # Dataset info
        'data': {
            'classes': config.get('data.classes', []),
            'num_classes': config.get('data.num_classes', 7),
            'normalization': {
                'mean': config.get('data.mean', []),
                'std': config.get('data.std', [])
            },
            'image_size': 224,
            'dataset_size': {
                'train': 14279,
                'valid': 1781,
                'test': 1793,
                'total': 17853
            }
        },
        
        # Training info
        'training': {
            'optimizer': 'Adam',
            'learning_rate': checkpoint_info.get('learning_rate', 0.001),
            'batch_size': config.get('training.batch_size', 32),
            'num_epochs': checkpoint_info.get('num_epochs', 'N/A'),
            'framework': 'PyTorch',
            'pytorch_version': torch.__version__
        },
        
        # Results
        'results': {
            'test_accuracy': args.test_acc,
            'validation_accuracy': args.valid_acc,
            'per_class_accuracy': checkpoint_info.get('per_class_accuracy', {}),
            'metrics': {
                'test_loss': checkpoint_info.get('test_loss', 'N/A'),
                'valid_loss': checkpoint_info.get('valid_loss', 'N/A')
            }
        },
        
        # Hardware
        'hardware': {
            'device': 'NVIDIA GeForce GTX 1060',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'training_time': checkpoint_info.get('training_time', 'N/A')
        },
        
        # Files
        'files': {
            'model_file': 'best_model.pth',
            'config_file': 'config.yaml',
            'metadata_file': 'metadata.json'
        }
    }
    
    return metadata

def main():
    """Fonction principale"""
    
    args = parse_args()
    
    print("📦 CRÉATION DE LA VERSION DU MODÈLE")
    print("=" * 60)
    
    # Vérifier checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint non trouvé : {checkpoint_path}")
        return
    
    print(f"✅ Checkpoint : {checkpoint_path}")
    print(f"✅ Version : {args.version}")
    
    # Créer dossier de sortie
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('checkpoints') / f'v{args.version}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossier de sortie : {output_dir}")
    
    # Charger le modèle pour obtenir les infos
    print("\n🏗️  Analyse du modèle...")
    model = create_resnet18()
    params_info = count_parameters(model)
    
    print(f"   Paramètres totaux : {params_info['total']:,}")
    print(f"   Paramètres entraînables : {params_info['trainable']:,}")
    
    # Charger per_class_metrics si disponible
    per_class_file = Path('results/per_class_metrics.json')
    per_class_accuracy = {}
    if per_class_file.exists():
        with open(per_class_file, 'r') as f:
            per_class_data = json.load(f)
            per_class_accuracy = {k: v['accuracy'] for k, v in per_class_data.items()}
    
    # Info checkpoint
    checkpoint_info = {
        'learning_rate': 0.001,
        'num_epochs': 'Transfer Learning (10) + Fine-tuning (5)',
        'per_class_accuracy': per_class_accuracy,
        'training_time': '3h42min'
    }
    
    # Créer métadonnées
    print("\n📝 Génération des métadonnées...")
    metadata = create_metadata(args, params_info, checkpoint_info)
    
    # Sauvegarder métadonnées
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Métadonnées : {metadata_path}")
    
    # Copier le modèle
    model_dest = output_dir / 'best_model.pth'
    if not model_dest.exists() or model_dest != checkpoint_path:
        print(f"\n📦 Copie du modèle...")
        shutil.copy2(checkpoint_path, model_dest)
        print(f"✅ Modèle copié : {model_dest}")
    
    # Copier la config
    config_src = Path('src/config/config.yaml')
    config_dest = output_dir / 'config.yaml'
    if config_src.exists():
        shutil.copy2(config_src, config_dest)
        print(f"✅ Config copiée : {config_dest}")
    
    # Copier les résultats si disponibles
    results_files = [
        ('results/classification_report.txt', 'classification_report.txt'),
        ('results/per_class_metrics.json', 'per_class_metrics.json'),
        ('results/confusion_matrix.png', 'confusion_matrix.png')
    ]
    
    print(f"\n📊 Copie des résultats...")
    for src, dest in results_files:
        src_path = Path(src)
        if src_path.exists():
            dest_path = output_dir / dest
            shutil.copy2(src_path, dest_path)
            print(f"✅ {dest}")
    
    # Créer README pour cette version
    readme_content = f"""# Recycle-moi Model v{args.version}

{args.description}

## Métriques

- **Test Accuracy**: {args.test_acc:.2f}%
- **Validation Accuracy**: {args.valid_acc:.2f}%
- **Architecture**: ResNet18 (Transfer Learning + Fine-tuning)
- **Dataset**: 17,853 images (7 classes)

## Métriques par Classe

"""
    
    if per_class_accuracy:
        for class_name, acc in per_class_accuracy.items():
            readme_content += f"- **{class_name}**: {acc:.2f}%\n"
    
    readme_content += f"""

## Fichiers

- `best_model.pth`: Poids du modèle
- `metadata.json`: Métadonnées complètes
- `config.yaml`: Configuration utilisée
- `classification_report.txt`: Rapport détaillé
- `confusion_matrix.png`: Matrice de confusion

## Utilisation
```python
from src.models.resnet import load_model

model = load_model('checkpoints/v{args.version}/best_model.pth')
```

## Date de création

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ README : {readme_path}")
    
    # Résumé final
    print("\n" + "=" * 60)
    print("✅ VERSION CRÉÉE AVEC SUCCÈS!")
    print("=" * 60)
    print(f"📁 Dossier : {output_dir}")
    print(f"📝 Version : {args.version}")
    print(f"🏆 Test Accuracy : {args.test_acc:.2f}%")
    print(f"📊 Validation Accuracy : {args.valid_acc:.2f}%")
    print("\nFichiers créés:")
    for file in output_dir.iterdir():
        print(f"   - {file.name}")
    print("=" * 60)

if __name__ == '__main__':
    main()