"""
Script d'entraînement pour Recycle-moi
Usage: python scripts/train.py
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse

from src.config import config
from src.data.dataset import create_dataloaders
from src.models.resnet import create_resnet18
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, count_parameters, save_metadata
from src.utils.logger import setup_logger

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Entraînement du modèle Recycle-moi')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Nombre d\'epochs (défaut: depuis config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Taille du batch (défaut: depuis config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (défaut: depuis config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda ou cpu, défaut: depuis config)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Dossier pour sauvegarder les checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité (défaut: 42)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Ne pas utiliser les poids pré-entraînés')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Geler les couches convolutionnelles')
    
    return parser.parse_args()

def main():
    """Fonction principale d'entraînement"""
    
    # Parser arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("🚀 Démarrage de l'entraînement Recycle-moi")
    
    # Fixer la seed
    set_seed(args.seed)
    
    # Device
    device = args.device if args.device else config.get('hardware.device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    
    # ==================== DATA ====================
    logger.info("📊 Chargement des données...")
    
    train_loader, valid_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size
    )
    
    logger.info(f"   Train: {len(train_loader)} batches ({len(train_loader.dataset)} images)")
    logger.info(f"   Valid: {len(valid_loader)} batches ({len(valid_loader.dataset)} images)")
    logger.info(f"   Test: {len(test_loader)} batches ({len(test_loader.dataset)} images)")
    
    # ==================== MODEL ====================
    logger.info("🏗️  Création du modèle...")
    
    model = create_resnet18(
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone
    )
    
    # Compter paramètres
    params_info = count_parameters(model)
    logger.info(f"   Paramètres totaux: {params_info['total']:,}")
    logger.info(f"   Paramètres entraînables: {params_info['trainable']:,}")
    logger.info(f"   Paramètres gelés: {params_info['frozen']:,}")
    
    # ==================== TRAINING SETUP ====================
    logger.info("⚙️  Configuration de l'entraînement...")
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    lr = args.lr if args.lr else config.get('training.learning_rate', 0.001)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.get('training.weight_decay', 0.0001)
    )
    
    # Scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config.get('training.scheduler.step_size', 3),
        gamma=config.get('training.scheduler.gamma', 0.5)
    )
    
    logger.info(f"   Learning rate: {lr}")
    logger.info(f"   Optimizer: Adam")
    logger.info(f"   Scheduler: StepLR")
    
    # ==================== TRAINER ====================
    logger.info("🎓 Initialisation du Trainer...")
    
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config.get('paths.checkpoints')
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        checkpoint_dir=checkpoint_dir
    )
    
    # ==================== TRAINING ====================
    num_epochs = args.epochs if args.epochs else config.get('training.num_epochs', 10)
    
    logger.info(f"🚀 Début de l'entraînement ({num_epochs} epochs)...")
    
    history = trainer.train(
        num_epochs=num_epochs,
        save_best=True,
        early_stopping_patience=config.get('training.patience', 5)
    )
    
    # ==================== RESULTS ====================
    logger.info("📊 Sauvegarde des résultats...")
    
    # Sauvegarder l'historique
    import json
    history_path = Path(checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"   Historique: {history_path}")
    
    # Sauvegarder les métadonnées
    metadata_path = Path(checkpoint_dir) / 'metadata.json'
    save_metadata(
        save_path=metadata_path,
        model_info={
            'architecture': 'resnet18',
            'pretrained': not args.no_pretrained,
            'freeze_backbone': args.freeze_backbone,
            'num_classes': config.get('data.num_classes'),
            'parameters': params_info
        },
        training_info={
            'num_epochs': num_epochs,
            'batch_size': args.batch_size if args.batch_size else config.get('training.batch_size'),
            'learning_rate': lr,
            'optimizer': 'Adam',
            'seed': args.seed
        },
        results={
            'best_valid_acc': trainer.best_valid_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_valid_acc': history['valid_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'final_valid_loss': history['valid_loss'][-1]
        }
    )
    
    logger.info(f"   Métadonnées: {metadata_path}")
    logger.info(f"   Meilleur modèle: {trainer.best_model_path}")
    
    logger.info("✅ Entraînement terminé avec succès!")
    logger.info(f"🏆 Meilleure validation accuracy: {trainer.best_valid_acc:.2f}%")

if __name__ == '__main__':
    main()